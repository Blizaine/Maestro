// Native Hunyuan3D support for Maestro.
//
// The official 2.0/2.1 sources and their Python environment stay inside
// app/services/hunyuan3d. Model weights are intentionally not downloaded here:
// Hugging Face downloads only the variant selected in Studio on first use.
module.exports = {
  run: [
    {
      when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2')}}",
      method: "shell.run",
      params: {
        message: "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 app/services/hunyuan3d/vendor/Hunyuan3D-2"
      }
    },
    {
      when: "{{exists('app/services/hunyuan3d/vendor/Hunyuan3D-2/.git')}}",
      method: "shell.run",
      params: {
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2",
        message: "git pull --ff-only"
      }
    },
    {
      when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1')}}",
      method: "shell.run",
      params: {
        message: "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 app/services/hunyuan3d/vendor/Hunyuan3D-2.1"
      }
    },
    {
      when: "{{exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1/.git')}}",
      method: "shell.run",
      params: {
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1",
        message: "git pull --ff-only"
      }
    },
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/hunyuan3d/env",
          python: "3.10"
        },
        message: [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
          "uv pip install -r app/services/hunyuan3d/requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/hunyuan3d/env",
          python: "3.10"
        },
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer",
        message: "uv pip install -e ."
      }
    },
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/hunyuan3d/env",
          python: "3.10"
        },
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer",
        message: "uv pip install -e ."
      }
    },
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/hunyuan3d/env",
          python: "3.10"
        },
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer",
        message: "uv pip install -e ."
      }
    },
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/hunyuan3d/env",
          python: "3.10"
        },
        shell: "{{which('bash')}}",
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer",
        message: "bash compile_mesh_painter.sh"
      }
    },
    {
      when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth')}}",
      method: "fs.download",
      params: {
        url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
      }
    },
    {
      method: "notify",
      params: {
        html: "Hunyuan3D support installed. Open Maestro's 3D section; model weights download on first use."
      }
    }
  ]
}
