# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1 — React UI build
# ============================================================================
FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# ============================================================================
# Stage 2 — SageAttention wheel build (needs the CUDA *devel* toolchain,
# which the final image doesn't ship — building the wheel here keeps
# ~5 GB of nvcc/headers out of the runtime image)
# ============================================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS sage-build

# CUDA compute capabilities to compile SageAttention for:
#   8.0 A100 | 8.6 RTX 30xx | 8.9 RTX 40xx | 9.0 H100 | 12.0 RTX 50xx
# Example override: --build-arg CUDA_ARCHITECTURES="8.6;8.9;12.0"
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip git ninja-build && \
    apt-get clean

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
# ponytail: 2 parallel nvcc jobs — each eats ~8 GB RAM compiling the sage
# kernels; raise if the build machine has >48 GB.
ENV MAX_JOBS="2"

# SageAttention's setup.py detects GPUs at build time — there are none in a
# Docker build, so patch it to use the arch list from the env var instead.
COPY <<EOF /tmp/patch_setup.py
import os
with open('setup.py', 'r') as f:
    content = f.read()

arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
arch_set = '{' + ', '.join([f'"{arch}"' for arch in arch_list.split(';')]) + '}'

old_section = '''compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")'''

new_section = 'compute_capabilities = ' + arch_set + '''
print(f"Manually set compute capabilities: {compute_capabilities}")'''

content = content.replace(old_section, new_section)

with open('setup.py', 'w') as f:
    f.write(content)
EOF

RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    python3 /tmp/patch_setup.py && \
    pip wheel --no-build-isolation --no-deps -w /wheels .

# ============================================================================
# Stage 3 — runtime image (build with --target runtime to skip SageAttention;
# the app then falls back to sdpa attention)
# ============================================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# build-essential: insightface ships as sdist on Linux (needs a C++
# compiler) and torch.compile needs a C compiler at runtime too.
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential git wget curl libgl1 libglib2.0-0 ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch first with the exact cu128 builds, so requirements.txt doesn't pull
# generic versions. Changing CUDA here also means changing the FROM images.
RUN pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY VERSION ./VERSION
COPY app/ ./app/
COPY --from=ui-build /ui/dist ./ui/dist

# GPL-3.0 seed-vc voice-conversion component — distributed from its own
# repository under its own license, never vendored into this repo.
RUN git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc app/postprocessing/seedvc

# ponytail: runs as root — sidesteps volume-ownership pain on first run;
# switch to a fixed-uid user + entrypoint chown if that ever matters.
ENV SERVER_NAME=0.0.0.0
ENV SERVER_PORT=7860
EXPOSE 7860

WORKDIR /workspace/app
# --settings stays at the default app/settings (wgp only auto-creates that
# relative path); persistence comes from the volume mounts in compose.
CMD ["python3", "launch.py", "--config", "/data/config"]

# ============================================================================
# Stage 4 (default) — runtime + SageAttention kernels
# ============================================================================
FROM runtime
RUN --mount=type=bind,from=sage-build,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*.whl
