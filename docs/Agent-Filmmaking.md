# Agent-led filmmaking

Maestro v4.1 adds title cards, finishing and take review to Local and Remote MCP.
Existing scopes still apply: observe reads and inspects media, generate renders
and authorizes retakes, and manage saves creative reviews and Editor projects.
Read `help("filmmaking")` for the complete live contract and examples.

## Title cards

Ask the agent to include an opening title, a story intertitle or an end card.
For reliable lettering it can request `get_creation_context` with workflow
`title.card`, author a card and use the normal prepare/submit flow. Cards render
on CPU without loading a model. Text is preserved; text that cannot fit or uses
unavailable glyphs produces an actionable error rather than being shortened.

Cards support exact text, subtitle, minimal/classic/bold style, background and
text colors, a 0.5–15 second hold, canvas dimensions and frame rate. An authorized
image asset can supply artwork. For fully painted lettering or animated titles,
the agent can instead use the installed image/video model and its native guide.
AI lettering requires visual review.

Director plans may include detached `title_cards`. `role` is `opening`, `end` or
`intertitle`; an intertitle also needs a one-based `after_clip` anchor. Cards do
not become characters or video reference subjects. Reserve their time when
requesting a total film duration. Opening/end cards wrap the already-muxed
master, including its audio. Music-video, source-audio-locked and seamless
intertitles require an explicitly reserved Editor timeline rather than silently
moving the soundtrack. Planner suggestions are optional and malformed
suggestions are skipped with a warning; explicit invalid cards are reported.

## Finish a film

`finish_film` creates a **new** editable Editor project. Supply ordered media
assets in `clips`, optional editable graphic `overlays`, narration/music/SFX
tracks, explicit trims, gains, fades and canvas settings. Use uploaded asset IDs
or gallery basenames plus workspace. Existing clip audio is preserved unless
explicitly disabled.

Each overlay has independent start/duration and a native graphic composition
with design-space width/height and rect, text or image elements. Each overlay is
stored on its own editable text track above the video. Image elements use an
authorized `asset` reference; Maestro resolves it to a project asset ID. Do not
put local paths or project asset IDs into image elements. Overlay timing must
fit within the assembled film. The graphic normalizer validates exact fields
and bounds, then scales design coordinates to the project canvas.

Audio rows accept optional `track` and `name`. Non-overlapping rows in the same
category with the same track name share an audio track. Overlapping rows use
deterministic parallel lanes named from the requested base (for example,
`Voice · 2`) so their authored starts stay unchanged. Different speaker tracks
remain independently editable; `name` labels each item. Music defaults to −18 dB,
effects to −9 dB and narration to 0 dB. Optional music ducking uses placed
narration or supplied speech ranges; Maestro does not infer speech timing from
a screenplay.

Example with video, a lower third, two named VO tracks and music:

```json
{
  "workspace": "default",
  "clips": [{"asset":{"filename":"interview.mp4","workspace":"default"},"duration":30,"include_audio":false}],
  "overlays": [{
    "name":"Alex lower third","start":0.5,"duration":7,
    "graphic":{"version":1,"width":1344,"height":768,"elements":[
      {"id":"bar","type":"rect","x":72,"y":570,"width":640,"height":120,"fill":"#101820","opacity":0.9,"border_color":"#FFFFFF","border_width":0,"radius":12},
      {"id":"speaker","type":"text","x":104,"y":592,"width":560,"height":64,"text":"ALEX RIVERA","font_family":"Arial","font_size":40,"color":"#FFFFFF","opacity":1,"align":"left","bold":true}
    ]}
  }],
  "narration": [
    {"asset":{"filename":"alex.wav","workspace":"default"},"track":"Voice — Alex","name":"Opening answer","start":0,"duration":8},
    {"asset":{"filename":"jamie.wav","workspace":"default"},"track":"Voice — Jamie","name":"Response","start":12,"duration":9}
  ],
  "music": [{"asset":{"filename":"piano.wav","workspace":"default"},"track":"Music bed","name":"Piano","start":0,"duration":30}],
  "export": true
}
```

`export: true` additionally starts native software export; otherwise it saves
the project only. The response includes `editable: true`, project ID/revision/
workspace and a track summary. Check the returned job before assuming export
succeeded. Existing projects are not overwritten.
## Review and polish

`inspect_take` probes and fully software-decodes the selected output, samples
five spread frames and sends an actual contact sheet to the agent. Set
`include_audio: true` for a short audio preview. The technical report and sampled
media do not certify continuous motion, dialogue accuracy or overall quality.
The agent should describe concrete observations and use `record_take_review`.
An unreviewed take stays unassessed; creative concerns do not block generation.

Only request extra renders within the user's explicit authorization. Create a
polish budget with maximum total renders and per-clip retakes, then use
`polish_take` for a selected owned Director clip and inspected source. The
budget is consumed durably before dispatch. A source change requires a fresh
inspection. Existing work must be idle. After a timeout, query the original
submission/request ID; never create a new ID to repeat uncertain work.

The first version polishes native Director takes. Arbitrary Studio files can be
inspected and assembled but are not automatically regenerated by this helper.
Model, hardware, fonts and MCP-client media support affect the result.
