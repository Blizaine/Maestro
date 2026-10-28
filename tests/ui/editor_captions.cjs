const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require(path.resolve(__dirname, '../../ui/node_modules/esbuild'));
const bundle = esbuild.buildSync({
  stdin: { contents: "export * from './src/editor/editorCaptions'", resolveDir: path.resolve(__dirname, '../../ui'), loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'cjs', logLevel: 'silent',
});
const built = { exports: {} };
new Function('module', 'exports', bundle.outputFiles[0].text)(built, built.exports);
const { parseSrt, parseWebVtt, parseCaptionFile, exportSrt, exportWebVtt, collectCaptionTrackCues, captionTracks } = built.exports;

const srt = '\ufeff1\r\n00:00:01,250 --> 00:00:03,500\r\nFirst line\r\nsecond line\r\n\r\n2\r\n00:00:04.000 --> 00:00:05.125\r\nAnother cue\r\n';
const srtCues = parseSrt(srt);
assert.deepEqual(srtCues, [
  { start: 1.25, end: 3.5, text: 'First line\nsecond line' },
  { start: 4, end: 5.125, text: 'Another cue' },
]);
assert.deepEqual(parseCaptionFile('WEBVTT\n\n00:01.000 --> 00:02.000\nVTT body', 'captions.srt'), [
  { start: 1, end: 2, text: 'VTT body' },
]);

const vtt = 'WEBVTT - generated\n\nNOTE generated locally\nignore this\n\nSTYLE\n::cue { color: white }\n\nword-1\n00:00:02.125 --> 00:00:04.250 align:start position:10%\nFirst VTT line\nsecond VTT line\n\n00:04.500 --> 00:05.000\nLast cue\n';
const vttCues = parseWebVtt(vtt);
assert.deepEqual(vttCues, [
  { start: 2.125, end: 4.25, text: 'First VTT line\nsecond VTT line' },
  { start: 4.5, end: 5, text: 'Last cue' },
]);
assert.deepEqual(parseCaptionFile('1\n00:00:00,000 --> 00:00:01,000\nAuto SRT', 'captions.srt')[0].text, 'Auto SRT');

assert.equal(exportSrt([{ start: 1.2346, end: 2.5, text: 'Rounded\nmultiline' }]), '1\n00:00:01,235 --> 00:00:02,500\nRounded\nmultiline\n');
assert.equal(exportWebVtt([{ start: 0, end: 3600.25, text: 'Long cue' }]), 'WEBVTT\n\n00:00.000 --> 01:00:00.250\nLong cue\n');
assert.deepEqual(parseSrt(exportSrt(srtCues)), srtCues);
assert.deepEqual(parseWebVtt(exportWebVtt(vttCues)), vttCues);

for (const invalid of [
  '1\n00:60:00,000 --> 00:00:01,000\nBad minute',
  '1\n00:00:02,000 --> 00:00:01,000\nReversed',
  '1\n00:00:00,000 --> 00:00:01,000\n  ',
  'WEBVTT\n\n00:60.000 --> 00:61.000\nBad seconds',
]) assert.throws(() => invalid.startsWith('WEBVTT') ? parseWebVtt(invalid) : parseSrt(invalid));
assert.throws(() => parseWebVtt('00:00.000 --> 00:01.000\nMissing header'), /header/);
assert.throws(() => exportSrt([]), /at least one/);

const project = {
  canvas: { width: 1920, height: 1080, fps: 24 },
  tracks: [
    { id: 'title', name: 'Titles', type: 'text', items: [
      { id: 'ordinary', start: 1, duration: 1, text: 'title' },
      { id: 'caption', start: 2, duration: 1, text: 'caption', caption: true },
    ] },
  ],
};
assert.deepEqual(collectCaptionTrackCues(project), [{ start: 2, end: 3, text: 'caption' }]);
assert.deepEqual(captionTracks(project), [{ id: 'title', name: 'Titles', count: 1 }]);
console.log('SRT/WebVTT parsing, multiline cues, validation, round-trip export, and caption-only timeline export passed.');
