"""Execute the Studio voice actions and request mapping against saved characters."""
from pathlib import Path
import shutil
import subprocess
import unittest


UI = Path(__file__).resolve().parents[1] / "ui"
SETUP = r'''
const fs = require('fs'), ts = require('typescript'), vm = require('vm'), assert = require('assert/strict');
const exports = {};
vm.runInNewContext(ts.transpile(fs.readFileSync('src/lib/ttsVoices.ts', 'utf8'), { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }), { exports });
const { ttsVoiceLimit, ttsAudioModeForCount, ttsCharacterEnhancePrompt, ttsSpeakingVoiceCount, ttsVoicePaths, applyTtsVoices } = exports;
const source = ts.createSourceFile('useStore.ts', fs.readFileSync('src/stores/useStore.ts', 'utf8'), ts.ScriptTarget.Latest, true);
const actions = {};
function visit(node) {
  if (ts.isPropertyAssignment(node) && ['setTtsVoiceCount', 'setTtsVoiceCharacter', 'setTtsVoiceFile', 'setTtsVoiceName', 'addTtsVoice', 'removeTtsVoice', '_autoParseSpkeakerNames'].includes(node.name.getText(source))) actions[node.name.getText(source)] = node.initializer;
  ts.forEachChild(node, visit);
}
visit(source);
const model = (architecture, selection, max) => ({ architecture, audio_only: true, any_audio_prompt: true, max_voice_count: max, audio_prompt_type_sources: selection ? { selection } : null });
const h3 = model('minimax_h3_voice_audio', ['', 'A', 'AB'], 2);
const kugel = model('kugelaudio_0_open', ['', 'A', 'AB']);
const qwen = model('qwen3_tts_base', ['A', 'AB']);
const index = model('index_tts2', ['A', 'AB', 'AB2']);
const scenema = model('scenema_audio', ['', 'A2', 'AB2'], 2);
const dramabox = model('dramabox_audio', ['', 'A', 'AB']);
const chatterbox = model('chatterbox');
const character = (id, name) => ({ id, name, visual: { type: 'image', path: '/portraits/' + id + '.png' }, voice: { filename: name + '.wav', path: '/characters/' + id + '/voice.wav' } });
const blaine = character('blaine-id', 'Blaine'), yoda = character('yoda-id', 'Yoda');
let state;
function reset(options = h3) {
  state = { params: { prompt: '', audio_prompt_type: '' }, audioSubMode: 'speech', modelOptions: options, ttsVoices: [], ttsVoiceCount: 0, ttsSpeakerNamesManual: false };
  const context = { ...exports, get: () => state, set: value => Object.assign(state, typeof value === 'function' ? value(state) : value) };
  for (const [name, node] of Object.entries(actions)) state[name] = vm.runInNewContext(ts.transpile('(' + node.getText(source) + ')', { target: ts.ScriptTarget.ES2022 }), context);
}
reset();
'''


@unittest.skipUnless(shutil.which("node") and (UI / "node_modules/typescript").exists(), "UI dependencies required")
class TtsCharacterTests(unittest.TestCase):
    def run_case(self, script):
        result = subprocess.run([shutil.which("node"), "-e", SETUP + script], cwd=UI, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_cloning_models_accept_saved_character_paths_with_native_modes(self):
        self.run_case(r'''
for (const [options, first, second, limit] of [[h3,'A','AB',2],[kugel,'A','AB',6],[qwen,'A','AB',2],[index,'A','AB2',2],[scenema,'A2','AB2',2],[dramabox,'A','AB',2],[chatterbox,'A',null,1]]) {
  reset(options);
  state.setTtsVoiceCharacter(0, blaine);
  assert.equal(state.params.audio_guide, blaine.voice.path);
  assert.equal(state.params.audio_prompt_type, first);
  assert.equal(state.ttsVoices[0].characterId, blaine.id);
  assert.equal(ttsVoiceLimit(options), limit);
  if (second) {
    state.setTtsVoiceCharacter(1, yoda);
    assert.equal(state.params.audio_guide2, yoda.voice.path);
    assert.equal(state.params.audio_prompt_type, second);
  }
}
assert.equal(ttsVoiceLimit({ ...qwen, architecture: 'qwen3_tts_customvoice', any_audio_prompt: false }), 0);
assert.equal(ttsVoiceLimit({ ...qwen, architecture: 'qwen3_tts_voicedesign', any_audio_prompt: false }), 0);
''')

    def test_removing_clearing_and_replacing_voices_cannot_submit_stale_audio(self):
        self.run_case(r'''
state.setTtsVoiceCharacter(0, blaine); state.setTtsVoiceCharacter(1, yoda);
state.removeTtsVoice(0);
assert.equal(state.ttsVoiceCount, 1); assert.equal(state.params.audio_guide, yoda.voice.path);
assert.equal(state.params.audio_guide2, undefined); assert.equal(state.ttsSpeakerName1, 'Yoda');
state.setTtsVoiceFile(0, 'manual.wav', '/uploads/manual.wav');
assert.equal(state.ttsVoices[0].characterId, undefined);
state.setTtsVoiceFile(0, null, null);
const request = { prompt: 'Hello.', audio_guide: '/stale.wav', audio_guide6: '/stale6.wav' };
applyTtsVoices(request, state.ttsVoices, state.ttsVoiceCount, h3);
assert.equal(request.audio_guide, undefined); assert.equal(request.audio_guide6, undefined);
assert.equal(request._tts_character_id1, '');
state.setTtsVoiceCount(0); assert.equal(state.ttsVoices.length, 0); assert.equal(state.params.audio_prompt_type, '');
''')

    def test_saved_names_survive_auto_parse_and_map_complete_labels_once(self):
        self.run_case(r'''
state.params.prompt = 'Someone: Hello.\nAnother: Goodbye.';
state.setTtsVoiceCharacter(0, blaine); state.setTtsVoiceCharacter(1, yoda);
state._autoParseSpkeakerNames(state.params.prompt, true);
assert.equal(state.ttsVoices[0].name, 'Blaine'); assert.equal(state.ttsVoices[1].name, 'Yoda');
const request = { prompt: 'Blaine: Let me quote Yoda: try this.\nYoda: Welcome.' };
applyTtsVoices(request, state.ttsVoices, 2, h3);
assert.equal(request.prompt, 'Speaker 1: Let me quote Yoda: try this.\nSpeaker 2: Welcome.');
assert.equal(request._tts_original_prompt, 'Blaine: Let me quote Yoda: try this.\nYoda: Welcome.');
const names = [{ name: 'Speaker 2' }, { name: 'Blaine' }];
const noCascade = { prompt: 'Blaine: First.\nSpeaker 2: Second.' };
applyTtsVoices(noCascade, names, 2, h3);
assert.equal(noCascade.prompt, 'Speaker 2: First.\nSpeaker 1: Second.');
const single = { prompt: 'Blaine: Hello.\nSpeaker 1: Welcome.' };
applyTtsVoices(single, state.ttsVoices, 1, chatterbox);
assert.equal(single.prompt, ' Hello.\n Welcome.');
''')

    def test_limits_missing_voice_and_index_emotion_mode(self):
        self.run_case(r'''
assert.throws(() => state.setTtsVoiceCharacter(0, { ...blaine, voice: null }), /no saved voice/);
assert.equal(state.ttsVoiceCount, 0);
state.setTtsVoiceCharacter(0, blaine); state.setTtsVoiceCharacter(1, yoda);
assert.throws(() => state.setTtsVoiceCharacter(2, blaine), /no available voice slot/);
state.addTtsVoice(); assert.equal(state.ttsVoiceCount, 2);
state.setTtsVoiceCount(6); assert.equal(state.ttsVoiceCount, 2);
assert.equal(ttsAudioModeForCount(2, index, 'ABNV'), 'AB');
const request = { prompt: 'Hello.', audio_prompt_type: 'ABNV' };
applyTtsVoices(request, state.ttsVoices, 2, index);
assert.equal(request.audio_prompt_type, 'ABNV');
assert.equal(ttsSpeakingVoiceCount(2, index, 'ABNV'), 1);
assert.equal(ttsSpeakingVoiceCount(2, index, 'AB2NV'), 2);
assert.equal(request.audio_guide2, yoda.voice.path);
''')

    def test_character_identity_roundtrips_and_explicit_zero_does_not_restore_old_slots(self):
        self.run_case(r'''
state.setTtsVoiceCharacter(0, blaine); state.setTtsVoiceCharacter(1, yoda);
const request = { prompt: 'Blaine: Hello.\nYoda: Welcome.' };
applyTtsVoices(request, state.ttsVoices, 2, h3);
const text = source.text;
const start = text.indexOf('    const restoredSpeakerName1 =');
const end = text.indexOf('    set(s => ({', start);
assert.ok(start > 0 && end > start);
const restore = ts.transpile(text.slice(start, end) + '\n({restoredVoiceCount,restoredVoices})', { target: ts.ScriptTarget.ES2022 });
const context = { p: request, uploadFilenames: { audio_guide: 'voice.wav' }, _deriveBase: path => path?.split('/').pop() ?? null };
let restored = vm.runInNewContext(restore, context);
assert.equal(restored.restoredVoiceCount, 2);
assert.equal(restored.restoredVoices[0].characterId, blaine.id);
assert.equal(restored.restoredVoices[1].characterName, 'Yoda');
assert.equal(restored.restoredVoices[0].filename, 'Blaine.wav');
request._tts_voice_count = 0;
restored = vm.runInNewContext(restore, { ...context });
assert.equal(restored.restoredVoiceCount, 0); assert.equal(restored.restoredVoices.length, 0);
''')

    def test_model_switch_clamps_voices_and_clears_unused_reference_paths(self):
        self.run_case(r'''
reset(kugel);
for (let i = 0; i < 6; i++) state.setTtsVoiceCharacter(i, character('id' + i, 'Person ' + i));
let switchBlock;
function find(node) {
  if (ts.isIfStatement(node) && node.expression.getText(source) === "options.audio_only && get().audioSubMode === 'speech'") switchBlock = node;
  ts.forEachChild(node, find);
}
find(source); assert.ok(switchBlock);
for (const options of [h3, chatterbox, { ...qwen, any_audio_prompt: false }]) {
  const ttsDefaults = {}, paramUpdates = {};
  vm.runInNewContext(ts.transpile(switchBlock.getText(source), { target: ts.ScriptTarget.ES2022 }), {
    ...exports, get: () => state, options, newMaxVoiceCount: ttsVoiceLimit(options), currentVoiceCount: state.ttsVoiceCount, ttsDefaults, paramUpdates,
  });
  assert.equal(ttsDefaults.ttsVoiceCount, ttsVoiceLimit(options));
  assert.equal(paramUpdates.audio_guide6, undefined);
  if (!options.any_audio_prompt) assert.equal(paramUpdates.audio_guide, undefined);
}
''')

    def test_enhancement_receives_selected_names_without_stale_voice_slots(self):
        self.run_case(r'''
state.setTtsVoiceCharacter(0, blaine); state.setTtsVoiceCharacter(1, yoda);
let prompt = ttsCharacterEnhancePrompt('Explain Maestro.', state.ttsVoices, 2);
assert.ok(prompt.includes('Speaker 1: "Blaine"'));
assert.ok(prompt.includes('Speaker 2: "Yoda"'));
assert.ok(prompt.endsWith('Requested speech:\nExplain Maestro.'));
prompt = ttsCharacterEnhancePrompt('Hello.', state.ttsVoices, 1);
assert.ok(!prompt.includes('Yoda'));
assert.equal(ttsCharacterEnhancePrompt('Hello.', [], 0), 'Hello.');
''')

    def test_save_form_uses_the_shared_library_and_video_audio_extraction(self):
        self.run_case(r'''
const component = ts.createSourceFile('TtsCharacterLibrary.tsx', fs.readFileSync('src/components/Sidebar/TtsCharacterLibrary.tsx', 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let saveNode;
function find(node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(component) === 'save') saveNode = node.initializer;
  ts.forEachChild(node, find);
}
find(component); assert.ok(saveNode);
(async () => {
  for (const [visual, voice, valid] of [[{ name: 'portrait.png' }, { name: 'voice.mp4' }, true], [{ name: 'character.mp4' }, null, true], [{ name: 'portrait.png' }, null, false]]) {
    let request, selected, error = '';
    const context = {
      name: 'Blaine', visual, voice, characters: [], canAdd: true,
      setError: value => { error = value; }, setSaving: () => {}, setCharacters: () => {}, onCharactersChange: () => {},
      setName: () => {}, setVisual: () => {}, setVoice: () => {}, setFormOpen: () => {}, onSelect: value => { selected = value; },
      api: { uploadImage: async () => ({ path: '/uploads/visual' }), uploadAudio: async () => ({ path: '/uploads/extracted.wav' }), createCharacter: async body => { request = body; return blaine; } },
    };
    const save = vm.runInNewContext(ts.transpile('(' + saveNode.getText(component) + ')', { target: ts.ScriptTarget.ES2022 }), context);
    await save();
    if (!valid) { assert.equal(request, undefined); assert.ok(error.includes('voice reference')); continue; }
    assert.equal(request.visual_path, '/uploads/visual'); assert.equal(selected.id, blaine.id);
    assert.equal(request.use_video_voice, !voice); assert.equal(request.voice_path, voice ? '/uploads/extracted.wav' : undefined);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
''')


if __name__ == '__main__':
    unittest.main()
