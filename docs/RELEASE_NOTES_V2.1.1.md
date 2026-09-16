# Maestro v2.1.1

Released 8 September 2026.

This patch improves AI Creative dialogue, H3 character binding and planning
feedback, and fixes the log instructions used for bug reports.

## More complete Creative dialogue

- Conversation and tutorial drafts use their speech allocation more fully.
  The default speaking rate remains 2.8 words per second, with a maximum of 3
  across all speakers combined. Brief tactical dialogue, explicit silence and
  requests to use only supplied lines retain their existing exceptions.
- H3 checks explicit feature and topic lists against the actual spoken script.
  Mentioning a requested talking point only in a visual description does not
  satisfy that check.
- Inadequate H3 windows receive independent writing repairs and a second
  attempt with specific validation feedback, including the number of missing
  or excess words. A failed window no longer discards successful repairs in
  other windows.
- The focused dialogue writer has a dedicated guide and preserves the story
  schedule, allowed cast and exact supplied lines.

## Consistent characters and visible feedback

- Pronouns such as "They", "She" and "He" are excluded from inferred cast names.
- Unambiguous RefMod filenames resolve to the natural character names in the
  prompt throughout H3 planning and Subject/voice compilation. Ambiguous
  references, including separate versions of a character, remain distinct.
- Unresolved H3 planning warnings are visible in the normal Studio prompt area
  on desktop and mobile, with a **Review needed** indicator and instructions
  for reviewing or retrying the draft.
- Refresh retains the plan's AI Creative or AI Faithful setting.

## Finding logs — issue #118

The bug-report form and contributing guide now include the launcher script
folder in each log path. The README explains which file to use, how to open
the extensionless `latest` file, how to find older sessions, and what to provide
when no log file exists. See [Finding logs](../README.md#finding-logs).

## Updating and validation

Use **Update** in Pinokio, start Maestro and refresh the browser. Run AI Creative
again on an existing reviewed draft to apply the new planning rules.

See the [validation record](VALIDATION_V2.1.1.md) for completed checks and
the limits of the live prompt-planning tests.
