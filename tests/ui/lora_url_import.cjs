// Run with: node tests/ui/lora_url_import.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const ts = require(path.join(root, 'ui/node_modules/typescript'));
function load(relative, globals = {}) {
  const filename = path.join(root, relative);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  vm.runInNewContext(code, { module, exports: module.exports, URL, ...globals }, { filename });
  return module.exports;
}
const { suggestLoraImportDirectory: suggest, resolveLoraImportDirectory: resolve } = load('ui/src/lib/loraImport.ts');
const cases = [
  ['https://huggingface.co/creator/MiniMax-Cinematic-LoRA', 'minimax_h3'],
  [' https://huggingface.co/creator/MINIMAX_H3-Film/tree/main ', 'minimax_h3'],
  ['https://huggingface.co/creator/MiniMax%20H3%20Film', 'minimax_h3'],
  ['https://civitai.com/models/123/minimax-h3-film', 'minimax_h3'],
  ['https://huggingface.co/creator/LTX-2.3-Motion', 'ltx2'],
  ['https://huggingface.co/creator/FLUX.2-Klein-9B-Film', 'flux2_klein_9b'],
  ['https://huggingface.co/creator/FLUX.2-Klein-4B-Film', 'flux2_klein_4b'],
  ['https://huggingface.co/creator/Klein-Film', ''],
  ['https://huggingface.co/creator/Wan2.1-I2V-Film', 'wan_i2v'],
  ['https://huggingface.co/creator/Wan2.2-I2V-Film', 'wan'],
  ['https://huggingface.co/creator/Minimax-Music-3', 'minimax_music3_music'],
  ['https://huggingface.co/creator/Film-LoRA?token=minimax', ''],
  ['https://huggingface.co/creator/Film-H3', ''],
  ['https://huggingface.co/MiniMaxCreator/LTX-2.3-Film', 'ltx2'],
  ['https://example.com/creator/minimax', ''],
  ['', ''],
  ['https://huggingface.co/creator/%ZZ', ''],
];
for (const [url, expected] of cases) assert.equal(suggest(url), expected, url);

(async () => {
  const requests = [];
  const client = load('ui/src/api/client.ts', { fetch: async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true, json: async () => ({ status: 'downloading' }) };
  } });
  const h3 = 'https://huggingface.co/creator/MiniMax-Film';
  await client.importHuggingFaceLora(h3, resolve(h3, ''));
  assert.equal(requests.at(-1).body.target_dir, 'minimax_h3');
  // A manual selection wins over both the original URL and later URL edits.
  for (const url of [h3, 'https://civitai.com/models/123/ltx-2.3-film']) {
    await client.importHuggingFaceLora(url, resolve(url, 'custom_adapters'));
    assert.equal(requests.at(-1).body.target_dir, 'custom_adapters');
    assert.equal(requests.at(-1).body.url, url);
  }
  const unknown = 'https://civitai.com/models/123';
  await client.importHuggingFaceLora(unknown, resolve(unknown, ''));
  assert.equal(requests.at(-1).body.target_dir, ''); // Preserve backend metadata detection.
  console.log(`${cases.length} URL suggestions and 4 import destination requests passed`);
})().catch(error => { console.error(error); process.exitCode = 1; });
