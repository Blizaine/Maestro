# MiniMax H3 LightX2V accelerators

Maestro includes four pinned LightX2V standard-LoRA recipes in the H3 Turbo
preset selector. They are available only on the matching H3 workflow:

| Workflow | Preset | Evaluations | CFG | Flow shift |
| --- | --- | ---: | ---: | ---: |
| First / Last (FL2VA) | v1.3, 768p | 4 | 1 | 6 |
| First / Last (FL2VA) | v1.1, 768p | 8 | 1 | 6 |
| References (Ref2VA) | v1.0, 768p | 4 | 1 | 12 |
| References (Ref2VA) | v1.1, 768p | 8 | 1 | 12 |

The presets set their step count, CFG and video flow shift when selected. The
weights are downloaded on first use from the pinned `DeepBeepMeep/MiniMax-H3`
revision and checked against the manifest SHA-256 before use. Their adapter
format follows the normal H3 LoRA path. Fused Turbo, FastH3, VDN and Alibaba
PAI PDD recipes keep their existing dedicated compatibility rules and runtime.

The values match the upstream profiles in `profiles/minimax_h3_fl2va` and
`profiles/minimax_h3_ref2va`. Upstream recipe settings are not a local visual
quality benchmark; the presets are marked experimental.
