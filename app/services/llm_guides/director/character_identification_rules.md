CHARACTER RULES:
- NEVER use character names in image_prompt, video_prompt, keyframe_prompts,
  window_prompts, or subjects_on_screen visual_description.
  NOT "Ava looks annoyed" — YES "the woman from the reference image looks annoyed".
  NOT "Alex is visible" — YES "the boy from the reference image is visible".
  Names are ONLY allowed inside quoted dialogue: "Alex, wait!" she says.
- When there is ONE reference image: use "[descriptor] from the reference image"
  where [descriptor] preserves the character's age, gender, and role EXACTLY as
  described in the screenplay or user prompt:
    teen boy → "the teen boy from the reference image"
    elderly woman → "the elderly woman from the reference image"
    young girl → "the young girl from the reference image"
    man → "the man from the reference image"
    female doctor → "the female doctor from the reference image"
  Do NOT normalize "teen boy" to "man", "young girl" to "woman", "elderly man"
  to "man", etc. The image model uses the descriptor to match the correct person
  in the reference — changing "teen boy" to "man" generates an adult male.
- When there are MULTIPLE reference images: use visual feature + image number:
  "the teen boy in blue from the first image", "the woman in red from the second image".

CLOTHING — NEVER NAME GARMENTS. THIS IS THE #1 RULE:
The single most common and damaging mistake is naming garment types anywhere
in the output — subjects_on_screen, image_prompt, video_prompt, keyframes,
window_prompts. If you name a garment, the image model REPLACES whatever the
reference shows with whatever you named.

ABSOLUTE BAN — never use ANY of these words in ANY output field:
  shirt, t-shirt, polo, blouse, sweater, sweatshirt, hoodie, jacket, coat,
  blazer, vest, cardigan, dress, gown, skirt, pants, slacks, trousers,
  jeans, shorts, lab coat, scrubs, uniform, robe, bodice, suit, tie.

HOW TO DISAMBIGUATE CHARACTERS — use COLOR ONLY + preserve age/role:
- BAD:  "man in dark blue shirt"     GOOD: "man in dark blue"
- BAD:  "woman in bright green sweater"  GOOD: "woman in bright green"
- BAD:  "attractive female doctor in white coat"  GOOD: "female doctor in white"
- BAD:  "The man in khaki shirt and slacks"  GOOD: "The man in khaki"
- BAD:  "teen boy in blue hoodie"    GOOD: "teen boy in blue"
- BAD:  "elderly woman in red dress" GOOD: "elderly woman in red"
- BAD:  "young girl in pink jacket"  GOOD: "young girl in pink"

CRITICAL — NEVER change the age or role descriptor:
- "teen boy" must stay "teen boy" — do NOT change to "man" or "young man"
- "girl" must stay "girl" — do NOT change to "woman"
- "elderly man" must stay "elderly man" — do NOT change to "man"
- "child" must stay "child" — do NOT change to "boy" or "girl" unless
  the gender is known from context

The reference image already shows what each character wears. Color alone is
enough for disambiguation. The model preserves the actual garment unchanged.

ONLY mention clothing when it explicitly CHANGES: "now in workout clothes",
"wearing a new red dress she just put on", "jacket removed".

- Each prompt is generated INDEPENDENTLY — the model has NO memory of other scenes.
- Re-anchor characters with color-based visual description in EVERY prompt.
