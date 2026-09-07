# Studio controls

Choose Video, Image or Audio and a workflow at the top of Studio. The complete workflow list opens over the editor. References sit at the top of the large composition area, and the prompt fills the remaining space. Choose the model beside Generate at the bottom.

The fixed settings strip contains Characters, Resolution, Aspect, Duration and Advanced. Indicators show the current choices; Auto duration is labeled Auto. Settings open upward on desktop and as sheets on mobile. The options still come from the selected model, including Auto and model-specific tiers. Opening a setting does not resize the prompt or move Generate. Click another indicator to switch panels, click outside to dismiss, or use the close button or Escape.

Duration contains Time, Window and Auto planning, the native-duration slider, presets and window settings. Time stops at five minutes; existing presets can still reach an hour where supported. Automatic window sizing continues while the panel is collapsed. Window overrides and saved GPU/model preferences work as before. H3 Reference sequences keep their **Carry motion and sound between windows** option here.

## Characters and media

**Characters** opens the saved library beside the sidebar on desktop and in a sheet on mobile, in Reference, supported Image, Viggle and Speech workflows. A Reference character remains one card containing its appearance and voice. Speech uses its saved voice; Image uses chosen original or recovered views. Viggle can prepare a character replacement frame or accept a manually edited frame.

Reference mode shows compact input cards and keeps one **Add reference** drop zone while the selected model has room. Drop files or tap to choose them. Click a card to open its detailed settings: preview, replace or describe a reference, choose an image/audio role, attach a video soundtrack, or adjust background isolation. These controls open over the editor. The arrow controls reorder inputs on touch screens; reference labels update with their order. Saved character appearance and voice move together. Larger input collections scroll within their area while the prompt remains available.

Frames and Extend keep their specific input roles: source video, start/end/timed frames, control video, soundtrack and supported references. Tiles wrap instead of scrolling sideways. Frame positions and per-input settings stay with their inputs. Image and Viggle keep their model-specific source/mask and preparation controls. Files still count against each model's real limits.

## Advanced

**Advanced** opens a settings overlay from the bottom strip. Its selected group, settings and unfinished preset drafts are retained when closed.

- **Performance:** H3 optimizations, reference preparation detail, applicable text encoders/decoders and cache tuning. Existing compatible settings and defaults are retained.
- **Finishing:** face refinement, H3 audio refinement and supported post-processing, including scaling, temporal upsampling, grain and voice replacement.
- **LoRAs & presets:** creative adapters, strengths and saved setups.
- **Generation:** seed, guidance, inference steps, output count and other applicable model controls.

The closed Advanced button shows its active count; its tooltip lists those settings. **Face refinement & character mapping** in the Reference character library opens the same Finishing settings. Gallery face refinement remains available for previous videos.

## Prompt and actions

The prompt fills the available composition area. Long scripts scroll inside the editor; **Expand prompt editor** opens a full writing view using the same source text and state. H3's reviewed window prompts remain editable there. AI Faithful, AI Creative and Manual retain their existing behavior, and Enhance is next to the prompt controls.

The **…** menu beside the model selector opens **Recipes** and **Model Browser**. Browse is also available inside the model picker. Generate keeps its two-part action: the left side generates now, and the right third adds the current settings to the held queue when supported. Transform and Blend retain their existing queue limitations. Specialized audio and finishing tools retain their dedicated composers or Run actions.

All three theme families, their light/dark variants and Auto appearance remain available in Settings. On mobile, the sidebar and overlays follow the available viewport. The workflow header, reference area and hardware status temporarily tuck away while the keyboard reduces the available height, leaving room to write without changing saved preferences.

## Local validation

After building `ui`, run `node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>`. The suite reads only the running app's model catalogue and model options. All browser writes, uploads, generations and downloads are intercepted at an isolated test origin. Screenshots are saved under `.codex-tmp/sidebar-validation/`.

Additional existing checks cover Studio duration convergence, saved H3 LoRA settings, character recovery/picking, and automatic/manual Viggle submissions. These UI checks do not run a GPU generation.
