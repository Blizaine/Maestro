# Speech writing and long scripts

In **Audio → Speech → Advanced → Generation → Speech writer**, save writing
instructions for tone, personality, vocabulary, storytelling and dialogue.
For example: “A warm, observant narrator with dry humor; give each speaker a
distinct point of view.” These notes supplement the selected speech model's
format rules. They go to the writing LLM, and are not prepended to the spoken
script or used as a Fish voice reference.
The preferences are remembered in this browser and retained in output settings.

**Write** uses the selected speech duration as its approximate draft target by
default. For example, selecting 20 minutes requests about 2,600 spoken words.
The writer estimates 130 words per minute; actual audio duration varies with
language, voice, pauses and delivery. For audio generation, that same duration
remains a maximum, so the output can finish earlier.

An explicit length in the brief takes precedence: a ten-minute story, a
twelve-minute dialogue, a word count, or an explicit short/brief request.
**Draft length override (minutes)** in Speech writer takes highest priority
when positive; leave it at 0 to use the brief or selected speech duration.
With **Auto** speech length, the writer follows the brief, and an unqualified
“long story” requests a bounded ten-minute draft. There is no fixed 600-word
or 3,000-character limit on the assembled draft.

Use **Write** to draft a monologue or dialogue. Long requests are planned and
written in sections with a bounded continuity context, then returned as one
editable script. The current limit is an approximate sixty-minute draft and
thirty model calls. Word estimates exclude performance cues and speaker labels.
Sections that reach the model's answer limit get a bounded continuation. A
section that stops far below its word target can also receive a bounded expansion;
a repeated section gets a focused repair instead of counting as new story.
Previously generated text is continuity context, not new verbatim source text.
Empty, short, repeated or still-truncated drafts carry review warnings when the
bounded recovery cannot resolve them; raw drafts remain in enhancement history.
Substantially overlong drafts also show a warning: a fixed synthesis maximum
could stop before the whole script is spoken. Maestro retains the text for review.
Review the complete script and
speaker assignments before generating audio. Writing does not start synthesis.

Speech cleanup preserves repeated phrases and refrains instead of applying the
generic video-prompt repetition cutoff. A normal rhetorical repetition must not
silently remove the remaining story or later dialogue turns. The complete raw
writer response remains available in enhancement history.

For **Fish Audio Drama 3**, **Auto** speech length generates long scripts in
bounded parts, carrying speaker/delivery instructions and recent audio context,
then assembles one audio file. You do not have to supply one prompt per part.
**Maximum length** is an optional whole-file cap, up to one hour. Generation
shows part progress. If a part or the whole output reaches its limit, the
available audio is retained with an incomplete warning and part coverage in
output details. A model end token does not independently verify that every
word was spoken; listen to the result before treating it as a finished take.

These controls do not raise the native limits of other speech generators.
Writing a long script and synthesizing it are separate capabilities.

## API

Draft with `POST /api/v1/llm/enhance-prompt`:

```json
{
  "mode": "audio",
  "model_type": "fish_drama3_int4",
  "prompt": "Write a dialogue about two lighthouse keepers discovering a letter.",
  "tts_enhance_mode": "dialogue",
  "tts_voice_count": 2,
  "duration_seconds": 1200,
  "tts_writer_instructions": "Warm, witty characters and a satisfying ending.",
  "tts_target_minutes": 12
}
```

For drafting, `duration_seconds` supplies the default target when the brief has
no explicit size and `tts_target_minutes` is 0; a positive `tts_target_minutes`
overrides both. Auto duration (0) follows the brief. The example uses the
explicit twelve-minute writer override rather than its twenty-minute clip.

The response includes `enhanced`, `warnings` and the saved draft history ID.
Submit the reviewed script through `POST /api/v1/generate` with
`generation_mode: "audio"`, the chosen `model_type`, and `duration_seconds: 0`
for Fish Auto, or a positive whole-output maximum. Clone voices through the
ordinary audio reference/transcript fields. The file's model details retain
the synthesis part report and any incomplete warnings.
