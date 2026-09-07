# Studio controls

Choose Video, Image or Audio and a workflow at the top of Studio. The complete workflow list opens over the editor. References sit at the top of the large composition area, and the prompt fills the remaining space. Choose the model beside Generate at the bottom.

The fixed settings strip keeps Characters on the left and groups Resolution, Aspect, Duration and Advanced together on the right. Advanced becomes an icon with its active count when the sidebar is narrow. In the smallest layouts, Characters also uses its icon so the controls remain separate and tappable. Indicators show the current choices; Auto duration is labeled Auto. Resolution and Aspect open compact lists above their buttons on both desktop and mobile, without redundant headings. Selecting a value closes its list; keyboard users can use the arrow keys, Home/End and Escape. The options still come from the selected model, including Auto and model-specific tiers. Duration and Advanced retain their detailed overlays. Opening a setting does not resize the prompt or move Generate. Click another indicator to switch panels or click outside to dismiss.

Duration contains Time, Window and Auto planning, the native-duration slider, presets and window settings. Time stops at five minutes; existing presets can still reach an hour where supported. Automatic window sizing continues while the panel is collapsed. Window overrides and saved GPU/model preferences work as before. H3 Reference sequences keep their **Carry motion and sound between windows** option here.

On mobile, settings overlays fit within the sidebar. The video Duration popup keeps a steady height while values change, with additional controls scrolling inside it. Sequence details sit below the sliders, so switching between one window and a longer sequence does not move the slider you are adjusting. The panel still adapts when the keyboard or available screen size changes.

## Characters and media

**Characters** opens the saved library beside the sidebar on desktop and in a sheet on mobile, in Reference, supported Image, Viggle and Speech workflows. A Reference character remains one card containing its appearance and voice. Speech uses its saved voice; Image uses chosen original or recovered views. Viggle can prepare a character replacement frame or accept a manually edited frame.

Reference mode shows compact input cards and keeps one **Add reference** drop zone while the selected model has room. Drop files or tap to choose them. Click a card to open its detailed settings: preview, replace or describe a reference, choose an image/audio role, attach a video soundtrack, or adjust background isolation. These controls open over the editor. The arrow controls reorder inputs on touch screens; reference labels update with their order. Saved character appearance and voice move together. Larger input collections scroll within their area while the prompt remains available.

Frames and Extend use the same compact, three-column tile layout as Reference mode, including on mobile. They keep their specific input roles: source video, start/end/timed frames, control video, soundtrack and supported references. Tiles wrap instead of scrolling sideways. Frame positions and per-input settings stay with their inputs. Image and Viggle keep their model-specific source/mask and preparation controls. Files still count against each model's real limits.

## Advanced

**Advanced** opens a settings overlay with collapsible section headings. Expand any combination of sections; their open state, settings and unfinished preset drafts are retained when the overlay is closed. Sections with no applicable controls are hidden. Available options still appear when switched off. When a model does not accept additional LoRAs, **Presets** remains available to save and restore its settings.

- **Performance:** H3 optimizations, reference preparation detail, applicable text encoders/decoders and cache tuning. Existing compatible settings and defaults are retained.
- **Finishing:** face refinement, H3 audio refinement and supported post-processing, including scaling, temporal upsampling, grain and voice replacement.
- **LoRAs & presets:** creative adapters, strengths and saved setups.
- **Generation:** seed, guidance, inference steps, output count and other applicable model controls.

The closed Advanced button shows its active count; its tooltip lists those settings. **Face refinement & character mapping** in the Reference character library opens the same Finishing settings. Gallery face refinement remains available for previous videos.

## Prompt and actions

The prompt fills the available composition area without a label or expand button. Its size and text width stay steady as you type or background status updates arrive, including when a scrollbar becomes necessary. Long scripts scroll inside the editor.

The magic button at the bottom-right explicitly enhances the prompt using **AI Faithful**, preserving the supplied events and dialogue. Its small arrow opens two choices: **AI Faithful** or **AI Creative**, which can add story beats and dialogue. The main button always uses Faithful, including after a Creative enhancement. Review or edit the result before submitting. Generate and Add to Queue use the visible prompt without running an unseen enhancement pass; old saved Manual/Auto/Creative choices no longer change that behavior. Speech retains its separate speech/dialogue enhancement menu in the same place.

In Image mode, Faithful clarifies the supplied description; Creative can add complementary visual details. Both keep explicit facts, character/reference constraints, requested text and edit boundaries. Restart a running backend after updating to activate the image enhancement instructions, then refresh the browser for the rebuilt controls.

For a long H3 sequence, Enhance prepares the individual window prompts. **Exact H3 prompts** opens their review screen, where each prompt remains editable. For LTX, enhancement writes one line per window directly in the prompt field. You can also choose a duration/window count and write those lines yourself. A missing window prompt produces an actionable message instead of triggering AI at submission. Auto duration estimates from the story or timed media; an old prompt-mode setting does not reinterpret paragraphs as separate windows. Previously queued jobs retain their original behavior.

The **…** menu beside the model selector opens **Recipes** and **Model Browser**. Browse is also available inside the model picker. Generate keeps its two-part action: the left side generates now, and the right third adds the current settings to the held queue when supported. Transform and Blend retain their existing queue limitations. Specialized audio and finishing tools retain their dedicated composers or Run actions.

All three theme families, their light/dark variants and Auto appearance remain available in Settings. On mobile, the sidebar follows both the height and vertical offset of the visible viewport, and the document behind the open drawer is held in place. The workflow header, reference area and hardware status temporarily tuck away while the keyboard reduces the available height, leaving room to write without changing saved preferences. Compact menus also follow their buttons when the keyboard moves the viewport.

## Local validation

After building `ui`, run `node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>`. The suite reads only the running app's model catalogue and model options. All browser writes, uploads, generations and enhancements are intercepted at an isolated test origin. It checks layouts, menus, simulated keyboard height/offset changes, and explicit enhancement/submission behavior. Screenshots are saved under `.codex-tmp/sidebar-validation/`. Set `MAESTRO_UI_ENHANCE_ONLY=1` to run just the enhancement checks during development. Keyboard geometry is simulated in Chromium; real iOS keyboard animation still needs device validation.

Additional existing checks cover Studio duration convergence, saved H3 LoRA settings, character recovery/picking, and automatic/manual Viggle submissions. These UI checks do not run a GPU generation.

Set `MAESTRO_UI_DURATION_ONLY=1` to run the duration popup checks alone. They drag the sliders across single/multiple-window boundaries with manual and automatic window sizing, verify sidebar bounds at desktop and mobile widths, and exercise scrolling and a simulated keyboard viewport.
