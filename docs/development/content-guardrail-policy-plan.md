# Local Content Guardrail Policy Plan

Status: product direction agreed; implementation deferred

Research date: 2026-08-14

## Purpose

Replace Maestro's broad keyword-based content checks with a transparent,
local-first policy that avoids obvious false positives while keeping one
narrow, non-optional boundary for explicit sexual depictions of minors.
Maestro runs locally, is open source, and already offers an adult-content
opt-in; its controls should therefore emphasize informed user choice instead
of silently rewriting or broadly rejecting prompts.

## Recommended user controls

Use one clearly named content-filtering selector with three levels:

1. **Standard** - contextual checks remain enabled and adult sexual content is
   restricted.
2. **Adult content allowed** - consensual adult content is permitted after the
   existing acknowledgement.
3. **Minimal safeguards** - general Maestro content filtering is disabled
   after an explicit local-only responsibility acknowledgement. A narrow hard
   block for explicit sexual depictions of minors remains active.

Keep the selected mode local to the installation. Do not require an account,
send telemetry, or transmit the acknowledgement.

## Context-aware evaluation

Do not classify a whole prompt by combining unrelated words. Evaluate
structured fields separately:

- character identity and stated age;
- visible actions;
- dialogue;
- signs and other background text;
- lyrics;
- props and setting.

For example, adult-oriented signage plus the dialogue phrase "Call me, baby"
must not be converted into a `sex + baby` match. Terms of address, song lyrics,
titles, and quoted text need their actual context.

When adult status is genuinely ambiguous in a sexual context, ask the user to
confirm that every depicted participant is at least 18 instead of issuing an
automatic refusal. Do not infer that an adult is a minor solely from words such
as `girl`, `boy`, `baby`, a school-like setting, or a fictional role.

## Transparency requirements

- Tell the user whether Maestro blocked the request or an underlying model or
  prompt-enhancement model refused it.
- Show the specific policy category and the relevant prompt field without
  exposing internal chain-of-thought.
- Never silently remove, soften, or replace prompt content.
- Keep prompt enhancement optional and preserve quoted dialogue verbatim.
- Make it clear that disabling Maestro's filter does not guarantee that every
  model will accept or accurately generate the request.

## Implementation outline

1. Inventory every safety check in Studio, Director planning, prompt
   enhancement, image generation, and repair/regeneration paths.
2. Replace keyword concatenation with a structured moderation input carrying
   field type, speaker/character, age assertions, and intended visual action.
3. Centralize the three-level policy so Studio and Director cannot diverge.
4. Add the responsibility acknowledgement and local persistence for Minimal
   safeguards.
5. Return typed decisions (`allow`, `confirm_adults`, `block`) with policy
   source (`maestro` or `upstream_model`) for clear UI messages.
6. Add a regression corpus covering benign ambiguous language, quoted
   dialogue, adult signage, lyrics, false age associations, valid adult
   content, and the retained hard-block category.

## Acceptance criteria

- The reported cyberpunk-signage prompt with adults saying "Call me, baby" is
  not blocked by Maestro merely because `sex` and `baby` occur in different
  contexts.
- Minimal safeguards disables Maestro's general filtering only after informed
  consent.
- Explicit sexual depictions of minors remain blocked in every mode.
- Director planning and downstream generation use the same selected policy.
- Users can distinguish a Maestro policy decision from an upstream model
  refusal.
- No prompt is altered without the user seeing and approving the change.
