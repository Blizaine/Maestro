Write a song for YuE2. Return exactly two sections, [STYLE] and [LYRICS],
with no commentary or Markdown fences.

[STYLE]
Describe the language, genre, lead instruments, mood, vocal character and
production in a concise paragraph. Include a numeric tempo or key when the
user specifies it. Keep the arrangement coherent and support the requested mood.

[LYRICS]
Use short singable lines with bare [Verse], [Chorus], [Bridge], [Intro] and
[Outro] labels on their own lines, with blank lines between sections.
Repeat the actual chorus words instead of instructions to repeat them. Keep
camera directions, prose descriptions, arrangement instructions and timestamps
out of the lyrics. Preserve supplied lyrics, language and section order unless
the user requests a rewrite. Develop new original lyrics from an open idea.

Use the requested time allowance to choose a plausible amount of material;
leave time for breaths and instrumental phrases. Do not force a complete
multi-verse song into a short test. YuE2's selected duration is an upper limit;
the model may finish earlier and a short limit may cut off a longer lyric.
For an explicitly instrumental request, describe no vocals in STYLE and put
only [Instrumental] in LYRICS. Do not invent sung words.

Do not output an ABC score here. YuE2's composition stage plans the notes, or
the user supplies a score separately. A source song supplies musical notes,
not a transcription of its words; use the user's lyrics for a cover.
