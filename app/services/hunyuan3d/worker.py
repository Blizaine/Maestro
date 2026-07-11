"""Short-lived Hunyuan3D inference worker used by Maestro.

The worker intentionally exits after one generation.  That makes VRAM release
deterministic and prevents Hunyuan's pinned dependency stack from interacting
with Maestro's main process.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V2_ROOT = HERE / "vendor" / "Hunyuan3D-2"
V21_ROOT = HERE / "vendor" / "Hunyuan3D-2.1"
sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V21_ROOT))
sys.path.insert(0, str(V21_ROOT / "hy3dshape"))
sys.path.insert(0, str(V21_ROOT / "hy3dpaint"))


def event(phase: str, progress: float, message: str) -> None:
    print("MAESTRO_EVENT " + json.dumps({"phase": phase, "progress": progress, "message": message}), flush=True)


def supported_call(callable_obj, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(**accepted)


def load_pil(path: str):
    from PIL import Image
    return Image.open(path).convert("RGBA")


def make_text_image(prompt: str, output_dir: Path, device: str) -> tuple[Any, str]:
    event("text_to_image", 0.08, "Loading HunyuanDiT text-to-image conditioner")
    from huggingface_hub import hf_hub_download
    from hy3dgen.text2image import HunyuanDiTPipeline

    repo_id = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
    pipeline = None
    for attempt in range(1, 4):
        try:
            # Resolve the public manifest before Diffusers loads the snapshot.
            # This seeds the revision cache and avoids the misleading generic
            # "model_index.json is missing" wrapper on metadata failures.
            hf_hub_download(repo_id, "model_index.json", token=False)
            pipeline = HunyuanDiTPipeline(
                repo_id,
                device=device,
            )
            break
        except OSError:
            if attempt == 3:
                raise
            delay = attempt * 5
            event("text_to_image", 0.08, f"Hugging Face metadata unavailable; retrying in {delay}s ({attempt}/3)")
            time.sleep(delay)
    assert pipeline is not None
    image = pipeline(prompt)
    path = output_dir / "text_condition.png"
    image.save(path)
    del pipeline
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return image.convert("RGBA"), str(path)


def prepare_images(request: dict[str, Any], engine: str, temp_dir: Path):
    settings = request["settings"]
    image_paths = request.get("images") or {}
    if image_paths:
        images = {name: load_pil(path) for name, path in image_paths.items()}
        source_path = image_paths.get("front") or next(iter(image_paths.values()))
    else:
        images = {}
        image, source_path = make_text_image(settings["prompt"], temp_dir, "cuda")
        images["front"] = image

    if settings.get("remove_background", True):
        event("preprocess", 0.16, "Removing image backgrounds")
        if engine == "v21":
            from hy3dshape.rembg import BackgroundRemover
        else:
            from hy3dgen.rembg import BackgroundRemover
        remover = BackgroundRemover()
        images = {name: remover(image) for name, image in images.items()}

    for name, image in images.items():
        normalized_path = temp_dir / f"{name}.png"
        image.save(normalized_path)
        if name == "front":
            source_path = str(normalized_path)
    return images, source_path


def load_v2_pipeline(model: dict[str, Any], settings: dict[str, Any]):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    event("loading_model", 0.24, f"Loading {model['label']}")
    try:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model["repo"],
            subfolder=model["subfolder"],
            use_safetensors=True,
            device="cuda",
        )
    except TypeError:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model["repo"], subfolder=model["subfolder"])
    configure_pipeline(pipeline, settings)
    return pipeline


def load_v21_pipeline(model: dict[str, Any], settings: dict[str, Any]):
    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception:
        pass
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

    event("loading_model", 0.24, f"Loading {model['label']}")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model["repo"])
    configure_pipeline(pipeline, settings)
    return pipeline


def configure_pipeline(pipeline, settings: dict[str, Any]) -> None:
    if settings.get("flashvdm") and hasattr(pipeline, "enable_flashvdm"):
        try:
            pipeline.enable_flashvdm(mc_algo=settings.get("mc_algo", "dmc"))
        except TypeError:
            pipeline.enable_flashvdm()
    if settings.get("compile"):
        if hasattr(pipeline, "compile"):
            pipeline.compile()
        elif hasattr(pipeline, "model"):
            import torch
            pipeline.model = torch.compile(pipeline.model)
    if settings.get("cpu_offload") and hasattr(pipeline, "enable_model_cpu_offload"):
        # Hunyuan3D 2.0/2.1 copied Diffusers' offload implementation but their
        # standalone pipeline never defines the `components` mapping it reads.
        # Build it after FlashVDM/compile so Accelerate hooks the final modules.
        if not isinstance(getattr(pipeline, "components", None), dict):
            pipeline.components = {
                name: getattr(pipeline, name)
                for name in ("conditioner", "model", "vae", "scheduler", "image_processor")
                if hasattr(pipeline, name)
            }
        pipeline.enable_model_cpu_offload()
        # The standalone Hunyuan pipeline stores `self.device` separately from
        # Accelerate's execution hooks and leaves it as CPU after offloading.
        # Its __call__ then uses self.device for latents, so restore the logical
        # execution device without moving the parked modules back to VRAM.
        import torch
        pipeline.device = torch.device("cuda")


def generate_mesh(request: dict[str, Any], images: dict[str, Any]):
    import torch

    model = request["model"]
    settings = request["settings"]
    pipeline = load_v21_pipeline(model, settings) if model["engine"] == "v21" else load_v2_pipeline(model, settings)
    image_input = images if model.get("multiview") else images.get("front", next(iter(images.values())))
    generator = torch.Generator(device="cuda").manual_seed(settings["seed"])
    call_kwargs = {
        "image": image_input,
        "generator": generator,
        "num_inference_steps": settings["num_inference_steps"],
        "guidance_scale": settings["guidance_scale"],
        "octree_resolution": settings["octree_resolution"],
        "num_chunks": settings["num_chunks"],
        "mc_algo": settings["mc_algo"],
        "output_type": "trimesh",
    }
    event("shape", 0.38, "Generating 3D geometry")
    started = time.time()
    result = supported_call(pipeline, **call_kwargs)
    mesh = result[0] if isinstance(result, (list, tuple)) else result
    print(f"Shape generation completed in {time.time() - started:.2f}s", flush=True)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return mesh


def simplify_mesh(mesh, settings: dict[str, Any]):
    if not settings.get("reduce_face"):
        return mesh
    event("simplify", 0.68, f"Reducing mesh to {settings['target_face_num']} faces")
    try:
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        return FaceReducer()(mesh, max_facenum=settings["target_face_num"])
    except Exception as exc:
        print(f"Mesh simplification skipped: {exc}", flush=True)
        return mesh


def texture_v2(mesh, image, settings: dict[str, Any]):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    subfolder = "hunyuan3d-paint-v2-0-turbo" if settings["texture_mode"] == "v2-turbo" else "hunyuan3d-paint-v2-0"
    event("texture", 0.72, "Loading Hunyuan3D Paint 2.0")
    try:
        paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder=subfolder)
    except TypeError:
        paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    if settings.get("cpu_offload") and hasattr(paint, "enable_model_cpu_offload"):
        paint.enable_model_cpu_offload()
    event("texture", 0.8, "Generating texture maps")
    try:
        textured = paint(mesh, image=image)
    except TypeError:
        textured = paint(mesh, image)
    del paint
    return textured


def texture_v21(mesh, source_image_path: str, settings: dict[str, Any], temp_dir: Path, output_path: Path) -> Path:
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    event("texture", 0.7, "Loading Hunyuan3D Paint 2.1 PBR")
    initial_mesh = temp_dir / "initial.glb"
    mesh.export(initial_mesh)
    resolution = settings["texture_resolution"]
    config = Hunyuan3DPaintConfig(6, resolution)
    config.realesrgan_ckpt_path = str(V21_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
    config.multiview_cfg_path = str(V21_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.custom_pipeline = str(V21_ROOT / "hy3dpaint" / "hunyuanpaintpbr")
    paint = Hunyuan3DPaintPipeline(config)
    obj_path = temp_dir / "pbr_textured.obj"
    event("texture", 0.8, "Generating PBR material maps")
    textured_obj = paint(
        mesh_path=str(initial_mesh),
        image_path=source_image_path,
        output_mesh_path=str(obj_path),
        save_glb=False,
    )
    textured_obj = Path(textured_obj or obj_path)
    glb_path = output_path if output_path.suffix.lower() == ".glb" else temp_dir / "pbr_textured.glb"
    textures = {
        "albedo": str(textured_obj).replace(".obj", ".jpg"),
        "metallic": str(textured_obj).replace(".obj", "_metallic.jpg"),
        "roughness": str(textured_obj).replace(".obj", "_roughness.jpg"),
    }
    create_glb_with_pbr_materials(str(textured_obj), textures, str(glb_path))
    del paint
    if output_path.suffix.lower() != ".glb":
        import trimesh
        converted = trimesh.load(glb_path, force="scene")
        converted.export(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = request["model"]
    settings = request["settings"]

    with tempfile.TemporaryDirectory(prefix="maestro_hy3d_") as temp_name:
        temp_dir = Path(temp_name)
        event("preparing", 0.04, "Preparing Hunyuan3D inputs")
        images, source_image_path = prepare_images(request, model["engine"], temp_dir)
        mesh = generate_mesh(request, images)
        mesh = simplify_mesh(mesh, settings)

        texture_mode = settings["texture_mode"]
        if texture_mode == "pbr":
            texture_v21(mesh, source_image_path, settings, temp_dir, output_path)
        else:
            if texture_mode in {"v2", "v2-turbo"}:
                front = images.get("front", next(iter(images.values())))
                mesh = texture_v2(mesh, front, settings)
            event("export", 0.94, f"Exporting {output_path.suffix.upper().lstrip('.')}")
            mesh.export(output_path)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Hunyuan3D did not produce an output file")
    event("completed", 1.0, "3D asset saved; releasing VRAM")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event("failed", 0.0, str(exc))
        raise
