Write a song for YuE2. Return exactly two sections, [STYLE] and [LYRICS],
with no commentary or Markdown fences.

[STYLE]
Describe the intended lead vocal first: voice type, singing/rapping/spoken
delivery, and any performer explicitly requested to perform the song. Preserve
requested vocal gender, language, solo/duet/choir setup and delivery; make these
clear rather than leaving them implicit in album titles or instrumentation.
Keep a requested performer name and any supplied training trigger intact,
integrated naturally into the vocal or performance description. Do not replace
them with generic timbre adjectives or just album references. A person mentioned
as the song's subject is not automatically its singer; a production influence
does not automatically request that artist's voice. Do not invent an artist
when none is requested or promise an exact voice match.

Then describe the genre, lead instruments, mood and production in the same
concise paragraph. Include a numeric tempo or key when the user specifies it.
Keep the arrangement coherent and support the requested mood. Vocal instructions
belong here, not among the words that will be sung.

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
