# Studio controls

Choose a media type, workflow and model at the top of Studio. The compact controls show the current resolution, aspect ratio and duration. Open a button to change it; the options come from the selected model, including Auto and model-specific tiers.

Duration contains Time, Window and Auto planning, the native-duration slider, presets and window settings. Time stops at five minutes; existing presets can still reach an hour where supported. Automatic window sizing continues while the panel is collapsed. Window overrides and saved GPU/model preferences work as before. H3 Reference sequences keep their **Carry motion and sound between windows** option here.

## Characters and media

**Characters** opens the saved library in Reference, supported Image, Viggle and Speech workflows. A Reference character remains one card containing its appearance and voice. Speech uses its saved voice; Image uses chosen original or recovered views. Viggle can prepare a character replacement frame or accept a manually edited frame.

Reference mode shows each input as a card and keeps one **Add reference** drop zone while the selected model has room. Drop files or tap to choose them. Open a card to preview, replace or describe a reference, choose an image/audio role, attach a video soundtrack, or adjust background isolation. The arrow controls reorder inputs on touch screens; reference labels update with their order. Saved character appearance and voice move together.

Frames and Extend keep their specific input roles: source video, start/end/timed frames, control video, soundtrack and supported references. Tiles wrap instead of scrolling sideways. Frame positions and per-input settings stay with their inputs. Image and Viggle keep their model-specific source/mask and preparation controls. Files still count against each model's real limits.

## Advanced

**Advanced** expands a full-width section directly below the Characters / Advanced row. It scrolls with the upper settings area; the prompt and Generate / Queue bar stay in place. Press Advanced again or the section's close button to collapse it. Its selected group and settings are retained.

- **Performance:** H3 optimizations, applicable text encoders/decoders and cache tuning. Existing compatible settings and defaults are retained.
- **Finishing:** face refinement, H3 audio refinement and supported post-processing, including scaling, temporal upsampling, grain and voice replacement.
- **LoRAs & presets:** creative adapters, strengths and saved setups.
- **Generation:** seed, guidance, inference steps, output count and other applicable model controls.

The closed Advanced button shows its active count. The line above Generate summarizes active settings. **Face refinement & character mapping** in the Reference character library opens the same Finishing settings. Gallery face refinement remains available for previous videos.

## Prompt and actions

The upper settings area scrolls. The prompt and action bar stay at the bottom, above hardware status. A long prompt scrolls inside its bounded editor; **Expand prompt editor** opens a large writing view using the same source text and state. H3's reviewed window prompts remain editable there. AI Faithful, AI Creative and Manual retain their existing behavior, and Enhance is next to the prompt controls.

**Recipes** and **Browse** open the existing libraries. Generate keeps its two-part action: the left side generates now, and the right third adds the current settings to the held queue when supported. Transform and Blend retain their existing queue limitations. Specialized audio and finishing tools retain their dedicated composers or Run actions.

All three theme families, their light/dark variants and Auto appearance remain available in Settings. On mobile, the sidebar follows the available viewport; hardware status temporarily tucks away when the on-screen keyboard reduces the available height, without changing its saved collapse preference.

## Local validation

After building `ui`, run `node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>`. The suite reads only the running app's model catalogue and model options. All browser writes, uploads, generations and downloads are intercepted at an isolated test origin. Screenshots are saved under `.codex-tmp/sidebar-validation/`.

Additional existing checks cover Studio duration convergence, saved H3 LoRA settings, character recovery/picking, and automatic/manual Viggle submissions. These UI checks do not run a GPU generation.
