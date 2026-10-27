// Run: node tests/ui/worlds_navigation.cjs
// Mounts real Studio/navigation components with an isolated Zustand store.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

async function main() {
  const appSource = fs.readFileSync(path.join(root, 'ui/src/App.tsx'), 'utf8');
  assert.match(appSource, /<Sidebar onOpenWorlds=\{openWorlds\} \/>/,
    'The app root provides the Worlds opener to Studio');
  const sidebarSource = fs.readFileSync(path.join(root, 'ui/src/components/Sidebar/Sidebar.tsx'), 'utf8');
  assert.match(sidebarSource, /<VideoWorkflowSelector onOpenWorlds=\{onOpenWorlds\} \/>/,
    'The Video workflow menu receives the dedicated Worlds opener');
  assert.match(appSource, /<WorldExplorer open=\{worldsOpen\} onClose=\{\(\) => setWorldsOpen\(false\)\} \/>/,
    'The app root wires the panel close callback');
  assert.match(appSource, /aria-label="Close World Explorer"/,
    'The backdrop exposes an accessible close action');
  assert.match(appSource, /if \(isMobile\) setSidebarOpen\(false\)/,
    'Opening Worlds from the mobile sidecar closes the sidecar first');

  const source = [
    "import * as React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {useStore, getModelsForFamily} from './src/stores/useStore';",
    "import {ModelSelector} from './src/components/Sidebar/ModelSelector';",
    "import {GenerateButton} from './src/components/Sidebar/GenerateButton';",
    "import {VideoWorkflowSelector} from './src/components/Sidebar/VideoWorkflowSelector';",
    'window.store = useStore;',
    'window.getModelsForFamily = getModelsForFamily;',
    'window.mount = () => {',
    '  const root = createRoot(document.getElementById("root"));',
    '  const Harness = () => {',
    '    return React.createElement("div", null,',
    '      React.createElement(VideoWorkflowSelector, {onOpenWorlds: () => window.store.getState().setWorldsOpen(true)}),',
    '      React.createElement(ModelSelector, {placement: "below"}),',
    '      React.createElement(GenerateButton, {stretch: true}),',
    '    );',
    '  };',
    '  root.render(React.createElement(Harness));',
    '};',
  ].join('\n');
  const bundle = await esbuild.build({
    stdin: {contents: source, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent('<div id="root"></div>');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      const world = {
        model_type: 'hy_nomas', name: 'Hy-Nomas World Runtime', family: 'wan',
        architecture: 'hy_nomas', is_i2v: true, is_t2v: true,
        guidance_max_phases: 1, fps: 24, world_model: true,
      };
      const studio = {
        model_type: 'studio_video', name: 'Studio Video', family: 'wan',
        architecture: 'wan', is_i2v: true, is_t2v: true,
        guidance_max_phases: 1, fps: 24,
      };
      window.worldCatalog = [world, studio];
      window.store.setState({
        servicesConfig: {worlds_enabled: false},
        modelsLoaded: true,
        modelsLoading: false,
        modelsLoadError: null,
        models: window.worldCatalog,
        families: [{id: 'wan', label: 'Wan', order: 1}],
        enabledModels: new Set(['hy_nomas', 'studio_video']),
        modelVisibilityOverrides: {},
        generationMode: 'video',
        sidebarMode: 'studio',
        studioVideoWorkflow: 'frames',
        studioImageWorkflow: 'generate',
        audioSubMode: 'speech',
        editSubMode: 'retake',
        params: {...window.store.getState().params, model_type: 'hy_nomas', image_mode: 0},
      });
      window.mount();
    });

    const workflowSelector = page.locator('button[title^="Text, first / last"]');
    await workflowSelector.click();
    const dialog = page.getByRole('dialog', {name: 'Choose a workflow'});
    const worlds = dialog.getByRole('button', {name: /^Worlds/});
    assert.equal(await worlds.count(), 0, 'Worlds stays out of Create while the experimental setting is disabled');
    await workflowSelector.click();
    await page.evaluate(() => window.store.setState({servicesConfig: {worlds_enabled: true}}));
    await workflowSelector.click();
    await worlds.waitFor();
    const placement = await dialog.evaluate(element => {
      const titles = [...element.querySelectorAll('div')].filter(item =>
        item.childElementCount === 0 && ['Create', 'Transform'].includes(item.textContent.trim()));
      const createTitle = titles.find(item => item.textContent.trim() === 'Create');
      const transformTitle = titles.find(item => item.textContent.trim() === 'Transform');
      const createGroup = createTitle.parentElement;
      const labels = [...createGroup.querySelectorAll(':scope > button')]
        .map(button => button.querySelector('span.min-w-0 > span')?.textContent.trim());
      const worldsOption = [...createGroup.querySelectorAll(':scope > button')]
        .find(button => button.innerText.includes('Create and explore saved worlds'));
      return {
        labels,
        inCreate: createGroup.contains(worldsOption),
        beforeTransform: Boolean(worldsOption.compareDocumentPosition(transformTitle) & Node.DOCUMENT_POSITION_FOLLOWING),
      };
    });
    assert.deepEqual(placement.labels.slice(-2), ['Blend', 'Worlds'], 'Worlds is immediately after Blend in Create');
    assert.equal(placement.inCreate, true);
    assert.equal(placement.beforeTransform, true, 'Create appears before Transform');
    const beforeOpen = await page.evaluate(() => {
      const state = window.store.getState();
      return {sidebarMode: state.sidebarMode, generationMode: state.generationMode,
        workflow: state.studioVideoWorkflow, modelType: state.params.model_type,
        imageMode: state.params.image_mode};
    });
    await worlds.click();
    assert.equal(await page.evaluate(() => window.store.getState().worldsOpen), true,
      'Selecting the menu item opens the dedicated Worlds panel');
    assert.equal(await page.evaluate(() => window.store.getState().sidebarMode), 'studio',
      'Opening Worlds preserves the current Studio/Director mode');
    assert.equal(await page.evaluate(() => window.store.getState().studioVideoWorkflow), 'frames',
      'Opening Worlds preserves the selected Studio workflow');
    assert.deepEqual(await page.evaluate(() => {
      const state = window.store.getState();
      return {sidebarMode: state.sidebarMode, generationMode: state.generationMode,
        workflow: state.studioVideoWorkflow, modelType: state.params.model_type,
        imageMode: state.params.image_mode};
    }), beforeOpen, 'Opening Worlds preserves the generation mode, selected model, and parameters');
    const selector = page.getByRole('button', {name: 'Choose model'});
    assert.equal((await selector.innerText()).trim(), 'No compatible model',
      'A saved world runtime is not presented as the current Studio choice');
    await selector.click();
    assert.equal(await page.getByRole('button', {name: 'Studio Video'}).count(), 1,
      'Ordinary video models remain selectable');
    assert.equal(await page.getByRole('button', {name: 'Hy-Nomas World Runtime'}).count(), 0,
      'World runtime metadata hides the model from the ordinary selector');
    assert.deepEqual(await page.evaluate(() => window.getModelsForFamily('wan', window.worldCatalog, 'video').map(model => model.model_type)),
      ['studio_video'], 'Shared mode/default model groups omit world-only models');
    assert.equal(await page.getByRole('button', {name: 'Need model'}).isDisabled(), true,
      'Studio generation stays blocked for a stale world-model selection');
    assert.deepEqual(errors, [], 'The mounted components have no runtime errors');
    await page.close();
  } finally {
    await browser.close();
  }
}

main().then(() => console.log('Worlds navigation and Studio model isolation checks passed.')).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
