OPTIONAL DIRECTOR TITLE CARDS

Title cards are separate editorial assets. Suggest them only when they add
clear value to this film; an empty `title_cards` list is normal. Never force
an opening, intertitle, or end card into every project.

Keep all card data separate from shot prompts, dialogue, screenplay action,
character descriptions, image prompts, visual references, and H3 fields. A
card is plain typography over its own background. Do not put character names,
appearance descriptions, reference-image instructions, logos, or visual
conditioning instructions in a card. Do not describe a title card inside any
shot's `video_prompt` or `image_prompt`.

When suggesting cards, return exact, final copy in the top-level `title_cards`
array beside `shots` when the response shape accepts an object. If the response
must be an array of shot rows, put `title_cards` on the first row only; the
Director detaches that field before shot validation and rendering. Do not copy
card metadata onto every shot. The supported shape is:

```json
{
  "text": "Exact title text",
  "subtitle": "Optional exact subtitle",
  "role": "opening",
  "style": "classic",
  "background_color": "#000000",
  "text_color": "#FFFFFF",
  "duration_seconds": 3.0
}
```

`role` must be `opening`, `intertitle`, or `end`. `style` must be `minimal`,
`classic`, or `bold`. Preserve exact spelling, punctuation, capitalization,
and line breaks. Keep copy concise and independently understandable. Do not
rewrite the copy in later planning passes.

Suggest at most one opening and one end card for the whole film. In a bounded
chapter or sequence batch, do not repeat those cards. `after_clip` always
indexes the complete authored shot list, not the current batch; if that global
boundary is unavailable, leave intertitles out rather than guessing.

Opening and end cards wrap the finished master. They have no clip anchor and
add matching silence around the completed picture and soundtrack. An
intertitle must include `after_clip`, a one-based index into the authored shot
list, and appears after that shot. Do not add `after_clip` to opening or end
cards.

For projects with source audio or Seamless generation, suggest opening/end
cards only. Do not suggest intertitles because inserting time between shots
would change the audio-to-picture timing. For a non-audio-locked, non-Seamless
story, an intertitle may be suggested at a meaningful boundary between
completed clips. Do not create a new character, speaker, or visual reference
for a title card.
