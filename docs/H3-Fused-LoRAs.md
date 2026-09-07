# H3 Fused 4-Step LoRAs (experimental)

Both **H3 Fused 4-Step — Frames** and **References** can load compatible H3
character, style and concept LoRAs from the existing H3 library. Open Advanced
and select an adapter under LoRAs; Director uses its Video LoRAs selector.
Saved output settings preserve the selected adapters and their strengths.

Start with one adapter at a modest strength (for example, 0.3–0.5) and a short
clip. Compare with a no-LoRA render using the same seed, prompt, references,
duration and four-step settings. Increase strength only if the effect needs
it. These are starting points for testing, not validated settings for every
adapter. Quality, runtime and VRAM depend on the adapter and scene.

The checkpoint keeps its four-step default, RES sampler, no-CFG recipe and
optional 4–8 Total Steps. Adding a LoRA does not automatically increase steps.
Mystic is already fused at 0.7; the LoRA strength controls cannot remove it.

Extra Turbo/PDD acceleration adapters and VDN adapters are excluded from the
fused model's library. Requests that include them through old settings or the
API fail with an explanation rather than silently dropping adapters or
changing their strengths. VDN requires its own model. DoRA is also excluded
because the current INT8 ConvRot loader does not support its magnitude update.
Renamed files are checked using available metadata and tensor descriptors;
unmarked acceleration files cannot always be identified automatically.

Ordinary H3 adapters use Maestro's existing format conversion, Full/Pruned
AdaLN conversion and ConvRot-aware LoRA loading. Unsupported tensor layouts
or shapes still fail loader validation. Successful loading does not guarantee
good visual results at four steps.

The [checkpoint publisher's workflow](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot)
demonstrates an additional motion adapter in a specialized second refinement
pass. Maestro's ordinary LoRA selector does not implement that second-pass
workflow; it applies the selected adapters during the normal generation pass.
