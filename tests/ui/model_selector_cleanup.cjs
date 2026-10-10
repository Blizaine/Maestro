// Focused checks for contextual model labels and visibility navigation.
// Run: node tests/ui/model_selector_cleanup.cjs
const assert = require('node:assert/strict');
const Module = require('node:module');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

async function loadNodeHelpers(sourcePath) {
  const filename = path.join(root, sourcePath);
  const transformed = await esbuild.transform(fs.readFileSync(filename, 'utf8'), {loader: 'ts', format: 'cjs'});
  const compiled = new Module(filename, module);
  compiled.filename = filename;
  compiled._compile(transformed.code, filename);
  return compiled.exports;
}

async function loadDisplayHelpers() {
  return loadNodeHelpers('ui/src/lib/modelDisplay.ts');
}

async function checkContextualNames() {
  const {modelDisplayName, modelNameForContext} = await loadDisplayHelpers();
  const h3 = (name, model_type = 'minimax_h3_ref2va') => ({
    model_type, name, architecture: model_type, family: 'minimax_h3',
  });

  assert.equal(
    modelNameForContext(h3('H3 Omni Pruned 20B — References'), 'video:references'),
    'H3 Pruned 20B',
    'References omits workflow words while preserving the Pruned checkpoint and size',
  );
  assert.equal(
    modelNameForContext(h3('H3 Omni — Pruned'), 'video:references'),
    'H3 Pruned',
    'The built-in References catalog name collapses to H3 plus its checkpoint distinction',
  );
  assert.equal(
    modelNameForContext(h3('H3 Omni Fused 4-Step — References (Experimental)'), 'video:references'),
    'H3 Fused 4-Step (Experimental)',
    'References preserves the fused recipe and experimental qualifier',
  );
  assert.equal(
    modelNameForContext(h3('MiniMax H3 Omni Full INT8 — References'), 'video:references'),
    'H3 Full INT8',
    'References preserves the Full checkpoint and quantization',
  );
  assert.equal(
    modelNameForContext(h3('H3 Eros Max · beta5 · INT8 ConvRot — References (CivitAI)'), 'video:references'),
    'H3 Eros Max · beta5 · INT8 ConvRot (CivitAI)',
    'References removes only the imported model workflow suffix and retains its source',
  );
  assert.equal(
    modelNameForContext(h3('MiniMax H3 — First / Last', 'minimax_h3'), 'video:frames'),
    'H3',
    'Frames removes a terminal First / Last descriptor',
  );
  assert.equal(
    modelNameForContext(h3('H3 First / Last — Pruned', 'minimax_h3'), 'video:frames'),
    'H3 Pruned',
    'Frames removes an embedded First / Last workflow label and keeps Pruned',
  );
  assert.equal(
    modelNameForContext(h3('H3 Fused 4-Step — Frames', 'minimax_h3_fused_turbo'), 'video:frames'),
    'H3 Fused 4-Step',
    'Frames keeps the fused recipe and removes the workflow suffix',
  );
  assert.equal(
    modelNameForContext(h3('H3 Fused 4-Step — First-Last — Frames (Experimental)', 'minimax_h3_fused_turbo'), 'video:frames'),
    'H3 Fused 4-Step (Experimental)',
    'Frames removes stacked workflow descriptors and keeps the qualifier',
  );

  const imported = {model_type: 'custom_model', name: 'Omni Edge — References', architecture: 'custom', family: 'custom'};
  assert.equal(modelNameForContext(imported, 'video:references'), 'Omni Edge', 'Non-H3 imported wording is preserved');
  assert.equal(modelDisplayName('minimax_h3_ref2va', [h3('H3 Omni — References')]), 'H3 Omni — References',
    'Global metadata labels continue to use the full registered name');
  assert.equal(
    modelNameForContext(h3('Studio First/Last Edition', 'minimax_h3'), 'video:frames'),
    'Studio First/Last Edition',
    'Embedded wording that is not a terminal workflow descriptor is preserved',
  );
}

async function checkVisibilityNavigation() {
  const source = [
    "import * as React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {useStore} from './src/stores/useStore';",
    "import {ModelSelector} from './src/components/Sidebar/ModelSelector';",
    'window.store = useStore;',
    'window.mountSelector = () => { window.selectorRoot = createRoot(document.getElementById("root")); window.selectorRoot.render(React.createElement(ModelSelector, {placement:"below"})); };',
  ].join('\n');
  const bundle = await esbuild.build({
    stdin: {contents: source, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent',
  });
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.setContent('<div id="root"></div>');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      const model = {
        model_type: 'minimax_h3_ref2va', name: 'MiniMax H3 Omni Pruned 20B — References',
        architecture: 'minimax_h3_ref2va', family: 'minimax_h3', is_i2v: true, is_t2v: true,
        guidance_max_phases: 1, fps: 24, supports_end_frame: true, supports_audio_input: true,
        supports_ref_images: true, omni_reference: true,
      };
      window.openedVisibilityScopes = [];
      window.resetScopes = [];
      window.modelFileDeletes = [];
      window.modelDownloads = [];
      window.store.setState({
        models: [model], families: [{id: 'minimax_h3', label: 'H3', order: 1}],
        enabledModels: new Set(), modelVisibilityOverrides: {}, generationMode: 'video',
        studioVideoWorkflow: 'references', studioImageWorkflow: 'generate', audioSubMode: 'speech',
        editSubMode: 'retake', params: {...window.store.getState().params, model_type: model.model_type, image_mode: 0},
        openModelVisibility: scope => window.openedVisibilityScopes.push(scope),
        resetEnabledModels: scope => window.resetScopes.push(scope),
        deleteModel: modelType => window.modelFileDeletes.push(modelType),
        downloadModel: modelType => window.modelDownloads.push(modelType),
      });
      window.mountSelector();
    });
    const trigger = page.getByRole('button', {name: 'Choose model'});
    await trigger.click();
    const enable = page.getByRole('button', {name: /Enable more models/});
    const disabled = {
      hasEnable: await enable.count() === 1,
      hasOneAvailable: (await page.locator('body').innerText()).includes('1 available'),
      hiddenCurrent: (await trigger.innerText()).trim() === 'No compatible model',
    };
    await enable.click();
    const enabledScope = await page.evaluate(() => window.openedVisibilityScopes.at(-1));
    const closesAfterSettings = await trigger.getAttribute('aria-expanded') === 'false';

    await page.evaluate(() => window.store.setState({enabledModels: new Set(['minimax_h3_ref2va'])}));
    const contextualTrigger = (await trigger.innerText()).trim();
    const fullTitle = await trigger.getAttribute('title');
    await trigger.click();
    const manage = page.getByRole('button', {name: 'Manage models', exact: true});
    const enabled = {hasManage: await manage.count() === 1,
      hasZeroCount: !(await page.locator('body').innerText()).includes('0 available')};
    const modelOption = page.getByRole('button', {name: 'H3 Pruned 20B'});
    const optionTitle = await modelOption.getAttribute('title');

    const resetAction = page.getByRole('button', {name: 'Reset model visibility for current workflow'});
    await resetAction.click();
    const resetDialog = page.getByRole('alertdialog');
    const resetMessage = await resetDialog.innerText();
    const resetMessageIsClear = /current workflow/i.test(resetMessage)
      && /No model files will be deleted/i.test(resetMessage)
      && /Hidden models can be re-enabled.*Enable more models/i.test(resetMessage);
    await resetDialog.getByRole('button', {name: 'Cancel'}).click();
    const afterCancel = await page.evaluate(() => ({
      resetScopes: [...window.resetScopes],
      enabledModels: [...window.store.getState().enabledModels],
    }));
    const dialogClearedAfterCancel = await page.getByRole('alertdialog').count() === 0;

    await resetAction.click();
    await page.getByRole('alertdialog').getByRole('button', {name: 'Confirm reset'}).click();
    const resetScope = await page.evaluate(() => window.resetScopes.at(-1));
    const modelFileEffects = await page.evaluate(() => ({
      deletes: [...window.modelFileDeletes], downloads: [...window.modelDownloads],
    }));

    await page.getByRole('button', {name: 'Reset model visibility for current workflow'}).click();
    await trigger.click();
    await page.waitForFunction(() => document.querySelector('[role="alertdialog"]') === null);
    const dialogClearedAfterPickerClose = await trigger.getAttribute('aria-expanded') === 'false';

    await trigger.click();
    await page.getByRole('button', {name: 'Reset model visibility for current workflow'}).click();
    await page.evaluate(() => window.store.setState({studioVideoWorkflow: 'frames'}));
    await page.waitForFunction(() => document.querySelector('[aria-label="Choose model"]')?.getAttribute('aria-expanded') === 'false');
    const closesAfterModeSwitch = await trigger.getAttribute('aria-expanded') === 'false';
    const dialogClearedAfterModeSwitch = await page.getByRole('alertdialog').count() === 0;
    await trigger.click();
    await page.getByRole('button', {name: 'Manage models', exact: true}).click();
    const framesScope = await page.evaluate(() => window.openedVisibilityScopes.at(-1));

    await page.evaluate(() => window.store.setState({
      studioVideoWorkflow: 'references', models: window.store.getState().models.map(model => ({...model, nsfw_only: true})),
      servicesConfig: {...window.store.getState().servicesConfig, nsfw_mode: false},
    }));
    await page.waitForFunction(() => document.querySelector('[aria-label="Choose model"]')?.innerText.trim() === 'No compatible model');
    const matureIsHidden = (await trigger.innerText()).trim() === 'No compatible model';
    assert.deepEqual(pageErrors, [], 'The selector renders without browser errors');
    assert.deepEqual(disabled, {hasEnable: true, hasOneAvailable: true, hiddenCurrent: true},
      'Disabled compatible models show an accurate Enable more models action');
    assert.equal(enabledScope, 'video:references', 'The action opens the References visibility scope');
    assert.equal(closesAfterSettings, true, 'Opening settings closes the model menu');
    assert.deepEqual(enabled, {hasManage: true, hasZeroCount: true},
      'Manage models stays available when zero compatible models are disabled');
    assert.equal(resetMessageIsClear, true,
      'Reset explains that only this workflow changes, files remain, and hidden models can be enabled again');
    assert.deepEqual(afterCancel, {resetScopes: [], enabledModels: ['minimax_h3_ref2va']},
      'Cancel leaves enabled models unchanged and does not invoke the reset action');
    assert.equal(dialogClearedAfterCancel, true, 'Cancel dismisses the inline reset confirmation');
    assert.equal(resetScope, 'video:references', 'Confirm resets only the current References workflow');
    assert.deepEqual(modelFileEffects, {deletes: [], downloads: []},
      'Reset visibility does not delete model files or start downloads');
    assert.equal(dialogClearedAfterPickerClose, true, 'Closing the picker clears the reset confirmation');
    assert.equal(contextualTrigger, 'H3 Pruned 20B', 'The selected model uses its concise contextual name');
    assert.match(fullTitle, /^MiniMax H3 Omni Pruned 20B — References/,
      'The trigger tooltip retains the full registered name');
    assert.equal(optionTitle, 'MiniMax H3 Omni Pruned 20B — References',
      'The model row title retains the full registered name');
    assert.equal(closesAfterModeSwitch, true, 'Switching workflows closes the open model list');
    assert.equal(dialogClearedAfterModeSwitch, true, 'Switching workflows clears the reset confirmation');
    assert.equal(framesScope, 'video:frames', 'The action follows the current workflow after a switch');
    assert.equal(matureIsHidden, true, 'Mature-only current models remain unavailable when Mature Mode is off');
    await page.close();
  } finally {
    await browser.close();
  }
}

async function checkVisibilityScopes() {
  const {getModelVisibilityScope, isModelEnabledInScope} = await loadNodeHelpers('ui/src/lib/modelVisibility.ts');
  const base = {generationMode: 'video', studioVideoWorkflow: 'references'};
  assert.equal(getModelVisibilityScope(base), 'video:references');
  assert.equal(getModelVisibilityScope({...base, studioVideoWorkflow: 'frames'}), 'video:frames');
  assert.equal(getModelVisibilityScope({generationMode: 'image', studioImageWorkflow: 'outpaint'}), 'image:outpaint');
  assert.equal(getModelVisibilityScope({generationMode: 'audio', audioSubMode: 'music'}), 'audio:music');
  assert.equal(isModelEnabledInScope('model-a', new Set(['model-a']), {}, 'video:references'), true,
    'A scoped setting inherits the global model choice until the user changes it');
  assert.equal(isModelEnabledInScope('model-a', new Set(['model-a']), { 'video:references': {'model-a': false} }, 'video:references'), false,
    'A workflow override can hide a globally enabled model');
  assert.equal(isModelEnabledInScope('model-b', new Set(), { 'video:references': {'model-b': true} }, 'video:references'), true,
    'A workflow override can enable a globally hidden model');
}

(async () => {
  await checkContextualNames();
  await checkVisibilityScopes();
  await checkVisibilityNavigation();
  console.log('Model selector cleanup checks passed: contextual labels, persistent management and reset actions, scoped reset confirmation, and dismissal');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
