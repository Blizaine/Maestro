# -*- coding: utf-8 -*-
"""Video Depth Anything preprocessor for temporally consistent depth maps.

Wraps the Video Depth Anything model to match the same interface as
DepthV2VideoAnnotator: forward(frames) -> list[np.ndarray uint8 HxWx3].

Video Depth Anything processes all frames together with temporal attention,
producing smooth, flicker-free depth sequences vs per-frame DAv2.
"""
import gc
import sys
import os
import time
import numpy as np
import torch


class VideoDepthAnnotator:
    """Temporally consistent video depth estimation using Video Depth Anything."""

    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(self, cfg, device=None):
        pretrained_model = cfg['PRETRAINED_MODEL']
        model_variant = cfg.get('MODEL_VARIANT', 'vitl')
        self.input_size = cfg.get('INPUT_SIZE', 518)

        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid variant '{model_variant}'. Must be one of: {list(self.MODEL_CONFIGS.keys())}")

        self.gpu_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Add the Video Depth Anything repo to sys.path
        vda_root = os.path.join(os.path.dirname(__file__), "video_depth_anything")
        if vda_root not in sys.path:
            sys.path.insert(0, vda_root)

        from video_depth_anything.video_depth import VideoDepthAnything

        config = self.MODEL_CONFIGS[model_variant]
        self.model = VideoDepthAnything(**config)
        state_dict = torch.load(pretrained_model, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        # Keep model on CPU until inference — the LTX transformer occupies most VRAM
        self.model = self.model.eval()
        print(f"[VideoDepth] Loaded Video Depth Anything ({model_variant}) on CPU, will use {self.gpu_device} for inference")

    @torch.inference_mode()
    def forward(self, frames):
        """Process video frames with temporal consistency.

        Args:
            frames: list of PIL Images or numpy arrays [H, W, 3] uint8

        Returns:
            list of numpy arrays [H, W, 3] uint8 (grayscale depth, 3-channel)
        """
        # Convert frames to numpy array [N, H, W, 3] uint8
        np_frames = []
        for frame in frames:
            if hasattr(frame, 'convert'):  # PIL Image
                np_frames.append(np.array(frame.convert('RGB')))
            elif isinstance(frame, torch.Tensor):
                np_frames.append(frame.detach().cpu().numpy().astype(np.uint8))
            else:
                np_frames.append(np.array(frame))

        if len(np_frames) == 0:
            return []

        stacked = np.stack(np_frames, axis=0)  # [N, H, W, 3]

        # Move model to GPU for inference, then back to CPU to free VRAM
        use_device = self.gpu_device
        try:
            self.model = self.model.to(use_device)
            print(f"[VideoDepth] Processing {len(np_frames)} frames on {use_device}")
        except RuntimeError as e:
            # CUDA OOM — fall back to CPU
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[VideoDepth] GPU OOM, falling back to CPU")
                torch.cuda.empty_cache()
                use_device = "cpu"
                self.model = self.model.to("cpu")
            else:
                raise

        try:
            t0 = time.time()
            depths, _ = self.model.infer_video_depth(
                stacked,
                target_fps=24.0,
                input_size=self.input_size,
                device=use_device,
            )
            elapsed = time.time() - t0
            fps = len(np_frames) / elapsed if elapsed > 0 else 0
            print(f"[VideoDepth] {len(np_frames)} frames in {elapsed:.1f}s ({fps:.1f} fps) on {use_device}")
        finally:
            # Always move model back to CPU to free VRAM for generation
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[VideoDepth] Inference done, model moved back to CPU")

        # depths: [N, H, W] float32
        # Normalize to 0-255 uint8 (global normalization for consistency)
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min > 1e-6:
            depth_norm = ((depths - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depths, dtype=np.uint8)

        # Convert to 3-channel [H, W, 3] per frame
        result = []
        for i in range(depth_norm.shape[0]):
            frame_depth = depth_norm[i]  # [H, W]
            frame_3ch = np.stack([frame_depth, frame_depth, frame_depth], axis=-1)
            result.append(frame_3ch)

        return result

    def unload(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            self.model = self.model.to("cpu")
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
