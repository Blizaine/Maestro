"""Concise, public guidance for using ComfyUI through Maestro and Agent Access."""
from __future__ import annotations


def get_comfyui_guide() -> dict[str, object]:
    """Return the MCP-facing ComfyUI workflow and safety guide."""
    return {
        "title": "ComfyUI workflows in Maestro",
        "overview": (
            "ComfyUI is optional and disabled by default. Maestro connects to an "
            "existing ComfyUI server. Maestro serializes its own generation jobs "
            "and unloads native models before a ComfyUI job. The optional dedicated "
            "memory setting asks ComfyUI to free its idle models; it does not create "
            "a separate process or reserve GPU memory. ComfyUI may share GPU memory "
            "with other applications."
        ),
        "workflow_import": {
            "endpoint": "POST /api/v1/comfyui/workflows",
            "instructions": (
                "Register a trusted ComfyUI API-format prompt graph (the graph "
                "accepted by ComfyUI's prompt endpoint), not a canvas/UI export. "
                "Provide its image, video, or audio kind and typed bindings. "
                "Registration stores the graph; Maestro does not execute it and "
                "does not install custom nodes automatically. It does not expose "
                "arbitrary raw server dispatch."
            ),
        },
        "bindings_and_preflight": {
            "endpoint": "POST /api/v1/comfyui/workflows/{id}/validate",
            "instructions": (
                "Bind Maestro inputs to explicit graph node IDs and input names. "
                "Validate with {\"inputs\": {...}} using the current selected "
                "values, including the checkpoint, before running. An empty body "
                "or {} validates defaults only. Preflight checks graph, node, "
                "model, and binding requirements; it does not test output quality."
            ),
        },
        "starter_workflows": (
            "The workflow library includes SDXL checkpoint text-to-image, a "
            "model-free connection check, and FLUX 3 Klein distilled text-to-image "
            "and single-reference editing templates. FLUX 3 needs current core "
            "node/model support and compatible diffusion model, text encoder and "
            "VAE filenames selected from that ComfyUI server. Its templates use "
            "16 Euler/simple steps, CFG 1, and independently selected output "
            "dimensions. Do not apply these distilled defaults to Klein Base "
            "or assume every Maestro quantized export is ComfyUI-compatible."
        ),
        "native_routes": (
            "Maestro keeps its native engine as the default. A ComfyUI route is "
            "opt-in per model type and runs through Maestro's shared generation "
            "boundary. For image and video routes, map prompt, seed, width, height, "
            "num_inference_steps, and guidance_scale; video routes also map "
            "video_length and fps. Other values such as references, audio, LoRAs, "
            "finishing, or multi-window controls are not inferred or translated. "
            "Bind every media reference explicitly. Native LoRA, finishing and "
            "multi-window requests are rejected by this first mapper; use Native "
            "or run a workflow whose own controls implement those operations."
        ),
        "model_reuse": (
            "Model discovery and scanning read configured model folders without "
            "loading tensor payloads or using the GPU. Linking records existing "
            "paths; it does not copy, move, rename, or delete model files. A "
            "filename match alone does not prove native compatibility. Maestro "
            "does not download models or install custom nodes automatically."
        ),
        "submission_and_recovery": {
            "run_endpoint": "POST /api/v1/comfyui/workflows/{id}/run",
            "receipt_endpoint": "GET /api/v1/comfyui/jobs/by-submission/{uuid}",
            "resume_endpoint": "POST /api/v1/comfyui/jobs/by-submission/{uuid}/resume",
            "abandon_endpoint": "POST /api/v1/comfyui/jobs/by-submission/{uuid}/abandon",
            "instructions": (
                "Run only a registered workflow ID, with authorized Maestro asset "
                "references for media and a fresh client_submission_id UUID for "
                "each logical run. If the POST outcome is unclear, GET the receipt "
                "for that exact UUID before taking action. The resume endpoint "
                "reconciles that submission's monitor state; it does not replay a "
                "prepared prompt. A prepared intent whose job was lost can be "
                "explicitly dismissed through abandon; dispatched work cannot. "
                "Never blindly resend an ambiguous run request or "
                "send arbitrary raw prompts to the ComfyUI server."
            ),
        },
        "cancellation": (
            "Cancellation deletes only an owned pending ComfyUI prompt. For an "
            "already running prompt it waits for terminal cleanup and discards "
            "cancelled outputs. It does not interrupt running work or clear "
            "a shared server queue. Never call ComfyUI's global interrupt or queue "
            "clear routes for per-job cancellation."
        ),
        "access": (
            "Reads and owner-checked receipt lookup use observe; connection tests "
            "and model discovery/scan/link use manage; registered workflow runs "
            "and receipt reconciliation use generate. Workflow imports, route or "
            "settings changes, and workflow deletion require admin."
        ),
    }


__all__ = ["get_comfyui_guide"]
