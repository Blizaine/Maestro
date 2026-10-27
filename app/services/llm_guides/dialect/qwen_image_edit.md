QWEN IMAGE EDIT MODEL — DIALECT RULES (apply to image_prompt):
- Format: Direct, concise edit instructions that retain the requested details.
- Describe a new scene only when requested; a bounded edit stays local.
- Preserve the opening-frame pose and requested visual style. A still frame may
  depict action already underway; subsequent transitions belong to video.
- Use directive language: "Make...", "Change...to...", "Add..."
- When clothing changes, describe what's visible underneath with ALL physical attributes.
- The model does NOT infer from the reference — explicitly describe everything that should be visible.
- No character names — describe by appearance only.
- State relevant preservation constraints, without unrequested restyling,
  photographic treatment or extra scene changes.
