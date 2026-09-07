# Blaine — Maestro tutorial presenter

A 14-window tutorial with the same 457 spoken words, regrouped from the original 28 sections. Set the native window length to **14.4 seconds**. With the default continuation overlap, the expected finished video is **3 minutes 11.5 seconds**. Dialogue targets **2.8 words per second** while speaking, followed by a brief silent hold for changing the screen demonstration.

## Paste-ready prompt

Open [blaine-maestro-reference-manual.txt](blaine-maestro-reference-manual.txt), select all, and paste it into Maestro's prompt field. The file has exactly **14 non-empty physical lines**, one complete native H3 Reference prompt per window. Each line contains all six Context-IR fields and one explicit `<d>[English] ...</d>` dialogue block. Maestro restores the field separators at the model boundary. Visual word wrapping is fine; keep the actual line breaks. The headings, timestamps, and screen-recording cues in this guide are not part of the generation prompt.

## Maestro setup

1. In Studio Video, choose Reference and an H3 Reference model.
2. Add only the saved character Blaine. The prompts bind his image to `<Picture 1>`, visible identity to `<Subject 1>`, and voice to `<Audio 1>` for speaker `(S1)`. Keep the audio assigned as Voice reference.
3. Choose landscape 16:9. Blaine stays centered; resize and position the presenter video in the corner during editing.
4. Choose **Manual - one per line** for Window prompts.
5. In Advanced, set **Sequence Window Length to 14.4 seconds** (345 frames). This is the per-run setting for your GPU; the safe automatic default remains 10.1 seconds.
6. In Duration, choose **Window** and set **14 windows**. Confirm the prompt counter reads **14/14**.
7. Keep **Carry motion and sound between windows** enabled with the default 18-frame overlap, and choose **Continuous take per clip** for Sequence Camera Coverage.

The displayed 14.4-second window is exactly 14.375 seconds at 24 fps. The first window contributes 14.375 seconds; each later window contributes 13.625 seconds after overlap. Total: 191.500 seconds. The prompts contain 31–34 words each, spoken in approximately 11.07–12.14 seconds at 2.8 words per second. Their opening pause is 0.25 seconds; the remainder is an explicit silent, closed-mouth hold.

For a framing and voice check, use [the first-window prompt](blaine-maestro-reference-first-window.txt) with **one 14.4-second window**. It now includes both the welcome and workspace introduction. For the complete tutorial, use all 14 lines and 14 windows.

## Presenter direction

Blaine sits in a fixed chest-up shot against a neutral warm-gray background, wearing a plain charcoal crewneck. His head and shoulders remain visible with room for cropping. Lighting, framing, voice identity, and wardrobe stay consistent. Natural blinking and small facial movements make the delivery conversational; hands remain below frame. Record Maestro's real interface separately, then place that recording below the presenter layer in your edit.

## H3 format and pacing

These prompts follow Maestro's [Reference enhancement guide](../../app/services/llm_guides/enhance/minimax_h3_ref2va_video.md), [dialogue and timed-silence guide](../../app/services/llm_guides/enhance/minimax_h3_video.md), and [Reference sequence guide](../../app/services/llm_guides/enhance/minimax_h3_reference_sequence.md): six fields, explicit Subject/Speaker/Audio bindings, literal dialogue tags, a silent closing interval, nonverbal ambience, and music set to N/A. Speech is planned at 2.8 words per second, within the new 3 words per second ceiling.

The original spoken wording is unchanged. The script's advice to put dialogue in quotation marks refers to ordinary Maestro prompts, which the enhancer/compiler converts to native H3 tags. This Manual file supplies those tags directly.

The timing is a generation instruction, not forced audio alignment. Listen to the first-window test for repetitions or improvised speech before committing to the complete tutorial.

## Dialogue and screen-recording cues

These are approximate global editing ranges. Align your screen recording to the actual generated speech. The local intervals describe only each window's new footage; later windows already account for overlap. Use prepared outputs and Director projects so demonstrations do not require waiting through generation on camera.

### 01. Welcome / Workspaces — 0:00.000–0:14.375

> Hey, I'm Blaine. Let me show you how I use Maestro to create, organize, and edit media. First, choose a workspace. Separate workspaces help me keep experiments and finished projects organized.

31 spoken words. Local speech interval: 0.25–11.32s; silent hold through 14.375s.

Show on screen: Show the Maestro home interface with your presenter overlay. Then Open the workspace switcher and select the tutorial workspace.

### 02. Studio / Reference model — 0:14.375–0:28.000

> In Studio, choose video, image, or audio. For this walkthrough, let's start by creating a video. Choose a model for your workflow. I'm using MiniMax H3 Reference with my saved character.

31 spoken words. Local speech interval: 0.25–11.32s; silent hold through 13.625s.

Show on screen: Show Studio and its media choices, then select Video. Then Select the Reference workflow and your installed H3 Reference model.

### 03. Saved character / Writing the shot — 0:28.000–0:41.625

> Open the character library and add Blaine. That attaches my saved appearance and voice to the generation. Describe the framing, background, and performance. Put the words you want spoken in quotation marks.

32 spoken words. Local speech interval: 0.25–11.68s; silent hold through 13.625s.

Show on screen: Open the saved character library, add Blaine, and show his visual and voice references. Then Show a short example prompt with shot directions and one quoted spoken line.

### 04. Reference roles / Framing and duration — 0:41.625–0:55.250

> The reference establishes who appears. The prompt describes what happens, including the camera, setting, and exact dialogue. Set the aspect ratio and duration. I'm leaving space around my shoulders so this shot crops comfortably.

34 spoken words. Local speech interval: 0.25–12.39s; silent hold through 13.625s.

Show on screen: Point out the character reference, then highlight the corresponding scene directions. Then Show aspect ratio and duration controls, plus a presenter preview if available.

### 05. Manual windows / AI prompt modes — 0:55.250–1:08.875

> Longer videos use multiple windows. In Manual mode, each nonempty prompt line controls its matching window. AI Faithful follows your supplied events and dialogue. Creative mode can develop additional story beats and dialogue.

33 spoken words. Local speech interval: 0.25–12.04s; silent hold through 13.625s.

Show on screen: Show Window duration mode and Manual - one per line. Highlight two separate prompt lines. Then Show the AI - Faithful and AI - Creative story + dialogue options without generating.

### 06. Test generation / Gallery — 1:08.875–1:22.500

> Make a short test first. Check the face, voice, framing, and delivery before generating the full sequence. Finished takes appear in the gallery. Preview your results, favorite useful versions, and open their generation details.

34 spoken words. Local speech interval: 0.25–12.39s; silent hold through 13.625s.

Show on screen: Show a prepared one-window test and briefly preview its result. Then Open a gallery result, preview it, and show its favorite and details controls.

### 07. Load Settings / Image tools — 1:22.500–1:36.125

> Load Settings restores a previous setup. Change one thing at a time, then compare the new result. Image tools create artwork and edit existing images. Those results can become starting frames for your videos.

34 spoken words. Local speech interval: 0.25–12.39s; silent hold through 13.625s.

Show on screen: Use Load Settings on a prepared result and point out the restored Studio controls. Then Switch to Image and show Create and Edit with prepared examples.

### 08. Video tools / LTX controls — 1:36.125–1:49.750

> Video tools include Extend, Repaint, and Recast. Each workflow provides its own compatible models and inputs. Supported LTX workflows accept control videos, giving you another way to guide motion and structure.

31 spoken words. Local speech interval: 0.25–11.32s; silent hold through 13.625s.

Show on screen: Show the video workflow choices and their source-media inputs. Then Show an LTX control-video workflow with a prepared source and result.

### 09. Audio tools / LoRAs — 1:49.750–2:03.375

> Audio tools cover speech, music, and sound effects. Choose a model for the sound you're creating. For compatible models, browse LoRAs inside Maestro. Check their suggested weights and guides before applying them.

32 spoken words. Local speech interval: 0.25–11.68s; silent hold through 13.625s.

Show on screen: Show the audio workflows and a few prepared outputs. Then Show the LoRA browser and a guide for a compatible model.

### 10. Director / Music videos — 2:03.375–2:17.000

> Next, open Director. This mode helps turn your story or soundtrack into a sequence of planned shots. For music videos, add your track. Director analyzes its structure and energy to help plan the shots.

34 spoken words. Local speech interval: 0.25–12.39s; silent hold through 13.625s.

Show on screen: Open Director and show its production setup. Then Show a prepared Music Video analysis and shot plan.

### 11. Short films / Review or Auto — 2:17.000–2:30.625

> For short films, describe the story and characters. Director develops a screenplay and breaks it into shots. Review the plan and adjust individual shots, or use Auto to run the complete production pipeline.

33 spoken words. Local speech interval: 0.25–12.04s; silent hold through 13.625s.

Show on screen: Show a prepared Short Film story, characters, and shot plan. Then Show reviewable shots and the Auto workflow option.

### 12. Into the Editor / Timeline editing — 2:30.625–2:44.250

> Open the Editor to assemble your results. Director productions can arrive as separate clips with their soundtrack. Arrange clips on the timeline, trim their edges, and split where needed. Add transitions where they help.

34 spoken words. Local speech interval: 0.25–12.39s; silent hold through 13.625s.

Show on screen: Open a prepared Director production in the Editor timeline. Then Demonstrate a trim, split, and transition using prepared clips.

### 13. Presenter overlay / Audio balance — 2:44.250–2:57.875

> Place your presenter above the screen recording, then resize and move it into a bottom corner. Keep the tutorial voice clear. Balance the track volumes so background audio never competes with the explanation.

33 spoken words. Local speech interval: 0.25–12.04s; silent hold through 13.625s.

Show on screen: Place the screen recording on the lower layer and the presenter on the upper layer; resize and position it. Then Show the presenter-audio volume and lower the optional background-audio track.

### 14. Export / Close — 2:57.875–3:11.500

> Preview the complete edit, choose your export resolution and format, then render the finished video. Create in Studio, plan in Director, and finish in the Editor. I'm Blaine. Thanks for watching.

31 spoken words. Local speech interval: 0.25–11.32s; silent hold through 13.625s.

Show on screen: Preview the assembled timeline, then show the export panel and settings. Then Hold on the finished project or a clean overview of Maestro.

## Editing notes

Place the screen recording on a lower video layer and the presenter on a layer above it. Crop the presenter as needed, scale him down, and choose the corner that covers the fewest important controls. The background is a clean rectangular camera background. Keep the presenter voice prominent and use the silent holds to switch screen demonstrations.

This script follows the features documented in the local Maestro v2.0.1 README and current UI. Only supported LTX workflows are described as using control video, and LoRAs are described as model-dependent.
