IMAGE PROMPT (image_prompt) — edit instructions for the image model:

FORMAT: "create new scene, [environment]. [who] is now [where/doing what]."

CRITICAL — THE IMAGE PROMPT IS THE OPENING FRAME OF THE SCENE:
The image prompt creates the FIRST FRAME that the video animates FROM.
It must show the STARTING STATE — before any action happens in the video prompt.
- If the scene involves a woman removing her top → the image shows her WEARING the top.
- If a man walks to a door → the image shows him AWAY from the door, where he starts.
- If characters sit down → the image shows them STANDING, about to sit.
- The VIDEO PROMPT handles the action. The IMAGE PROMPT sets up the moment BEFORE.
- BAD: showing the end result of the scene's action (woman already topless, man already at door)
- GOOD: showing the starting position that the video will animate from

RULES:
- Always start with "create new scene".
- Anchor characters to the reference image(s): "the woman from the reference image"
  or "the man from the reference image" — this tells the model to preserve their identity.
- If multiple reference images, specify which: "the man from image one", "the woman from image two".
- The MAIN PERFORMER / protagonist IS the character shown in the reference image —
  ALWAYS anchor them ("the singer from the reference image"). Describing them
  loosely invents a NEW character design.
- Keep character references BRIEF — the model sees the photos and preserves identity when anchored.
- Under 80 words per prompt.

CLOTHING — NEVER NAME GARMENTS, EVER. THIS IS THE #1 RULE:
The single most common and damaging mistake is naming garment types. The
reference image already shows what each character wears — if you name a
garment, the image model REPLACES whatever the reference shows with whatever
you named. A man wearing a blue shirt and khaki pants becomes a man in a
khaki shirt and khaki slacks if you write "the man in khaki shirt and slacks".

ABSOLUTE BANS — never use ANY of these words in an image prompt:
  shirt, t-shirt, polo, blouse, sweater, sweatshirt, hoodie, jacket, coat,
  blazer, vest, cardigan, dress, gown, skirt, pants, slacks, trousers,
  jeans, shorts, leggings, tights, stockings, lab coat, scrubs, uniform,
  robe, bodice, corset, bikini, bra, lingerie, suit, tie, tuxedo.

HOW TO DISAMBIGUATE CHARACTERS WHEN MULTIPLE ARE IN FRAME:
Use COLOR ALONE, with NO garment word attached. The color points at what the
reference shows; the absence of a garment word lets the model preserve it.

- BAD:  "The man in the khaki shirt and slacks and the woman in the forest
         green sweater."
- GOOD: "The man in khaki and the woman in forest green."

- BAD:  "The doctor in the white lab coat stands next to the man in the
         dark blue polo."
- GOOD: "The doctor in white stands next to the man in dark blue."

- BAD:  "The woman in the red dress leans over the man in the grey suit."
- GOOD: "The woman in red leans over the man in grey."

- BAD:  "Wearing her delicate lace-trimmed bodice."
- GOOD: (omit — the reference already shows her outfit)

- BAD:  "In his dark leather jacket."
- GOOD: (omit — the reference already shows his jacket)

The ONLY time it is acceptable to name a garment is when clothing explicitly
CHANGES in the scene: "now in workout clothes after changing", "wearing a new
red dress she just put on". Changes require the garment word so the model
knows what is different. For every other case, use color alone or omit entirely.

BODY POSITION — describe PHYSICAL MECHANICS, not emotions or narrative:
- The image model cannot interpret "lost in the moment" or "locked together emotionally" — these
  produce random poses. Describe exact physical arrangement instead.
  BAD: "they greet each other warmly"
  GOOD: "she steps forward with arms raised, his hands extending to meet hers, faces inches apart"
- Specify: POSTURE (standing, seated, kneeling), LIMB POSITIONS, FACIAL STATE (mouth open,
  eyes closed), and SPATIAL RELATIONSHIP to other characters.

ANCHOR CONTACT POINTS TO ANATOMY, NOT SCENERY:
When one character leans toward, reaches for, or comes close to another, name the
specific target BODY PART — not the furniture or general area. A large object
anchor like "bed", "chair", or "table" is a 2D area, and the model picks whatever
point is nearest, which frequently produces anatomically wrong results.
- BAD: "leaning over his bed"      GOOD: "leaning over his lap"
- BAD: "reaching toward the table" GOOD: "reaching toward the cup on the table"
- BAD: "her face close to him"     GOOD: "her face close to his lips"
Applies to ALL image prompts, regardless of content type.

STYLE CONSISTENCY:
- Match the visual style of the reference image. If the reference is photorealistic, all
  prompts must be photorealistic. Do NOT introduce cartoon, anime, or illustration styles.
- End with "Use lighting and color temp from reference image." to preserve the look.

CRITICAL — EVERY IMAGE PROMPT MUST BE VISUALLY UNIQUE:
- VARY composition: close-up, wide shot, low angle, overhead, profile, over-shoulder.
- VARY environment WITHIN the scene's established location(s): "same environment" for
  most shots; move to a NEW location only when the scene description itself calls for
  it. If the user's description pins the scene to one location, EVERY shot stays there —
  vary the angle, framing, and distance, never the place.
- VARY who is featured: some shots focus on one character, others show the group.
- NEVER write the same framing/pose/environment for consecutive shots.

EXAMPLES SHOW FORMAT ONLY — never copy their subjects, animals, objects, or settings
into your prompts. Everything in your prompts must come from THIS production's scene
description and reference images.

GOOD EXAMPLES:
- "create new scene, same environment. Close-up of the woman from the reference image, relaxed expression. Use lighting and color temp from reference image. Preserve character identity, attire, body attributes, and the art style of the reference image."
- "create new scene, new kitchen environment. The man from the reference image is now seated at the table. Use lighting and color temp from reference image. Preserve character identity, attire, body attributes, and the art style of the reference image."
- Scene where woman removes dress → IMAGE shows: "create new scene, bedroom. The woman from the reference image is now standing by the bed. Use lighting and color temp from reference image. Preserve character identity, attire, body attributes, and the art style of the reference image." (NOT already undressed)

BAD EXAMPLES:
- "create new scene, same environment. Blonde man playing guitar." — no reference anchoring.
- Describing the END state: "the woman is now topless on the bed" when the scene is ABOUT her undressing.
- "wearing her delicate lace bodice and updo" — describing clothing causes inconsistency.
- "cartoon style illustration of..." (for a PHOTOREALISTIC reference) — introducing a style not in the reference.
- "photorealistic, 8k detailed photo of..." (for a HAND-DRAWN reference) — same mistake in the other direction.

STYLE CONSISTENCY:
- Match the visual MEDIUM and ART STYLE of the reference image, whatever it is.
- Stylized reference (hand-drawn, sketch, watercolor, anime, cartoon, oil painting,
  pixel art, etc.) → NAME that medium explicitly in EVERY image prompt. Without
  naming the medium, the image model defaults to photorealism and destroys the style.

IMAGE PROMPTS DESCRIBE A FROZEN FRAME:
- Describe WHERE each person IS, not what they are DOING over time.
- POSITIONS not movements. The image is a single frozen moment.
- NO motion verbs: no walking, running, reaching, turning, dancing, gesturing.
- NO motion-photography effects: no motion blur, speed lines, long exposure,
  camera shake. The still frame is SHARP — motion belongs to the video.
- Describe EXPRESSIONS as physical states: "mouth open, brow furrowed" not "looking angry".
- NEVER use character names — describe by appearance only.

ALWAYS end every image_prompt with: "Preserve character identity, attire, body attributes, and the art style of the reference image."
