// Offline integration test: real review component and store, mocked APIs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {assertEnhancementReview} = require('./enhancement_review.cjs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const output = path.join(root, '.codex-tmp/sidebar-validation');
fs.mkdirSync(output, {recursive: true});

(async () => {
  const bundle = await esbuild.build({stdin: {contents: [
    "import React from 'react'; import {createRoot} from 'react-dom/client';",
    "import {useStore} from './src/stores/useStore'; import {EnhancedJobReview} from './src/components/EnhancedJobReview';",
    "window.store = useStore; window.baseParams = {...useStore.getState().params};",
    "window.mountReview = job => {window.reviewRoot?.unmount(); const node = document.createElement('div'); document.body.append(node); window.reviewRoot = createRoot(node); window.reviewRoot.render(<EnhancedJobReview job={job} onClose={() => {window.reviewRoot.unmount(); node.remove();}}/>);};",
    "window.resetFixture = () => {const id = 'minimax_h3_ref2va_fused_turbo'; const model = {model_type:id, architecture:'minimax_h3_ref2va', omni_reference:true, fps:24, frames_minimum:124, frames_maximum:345, frames_steps:17}; useStore.setState({models:[model], modelOptions:model, generationMode:'video', sidebarMode:'studio', studioVideoWorkflow:'references', studioVideoEffectiveCreateRoute:'omni', durationSeconds:82.5, slidingWindowSeconds:14.375, slidingWindowOverlap:18, slidingWindowLocked:true, startImage:null, endImage:null, imageRefs:[], jobs:[], isGenerating:false, isEnhancing:false, h3WindowPlan:null, promptEnhanceError:null, params:{...window.baseParams, model_type:id, prompt:'A courier in a garden. No dialogue.', resolution:'864x480', image_mode:0, minimax_h3_references:[{type:'image',path:'/ref.png',name:'Courier'}]}});};",
  ].join('\n'), resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1360, height: 900}});
    const errors = [], llmRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const endpoint = new URL(route.request().url()).pathname;
      if (endpoint.includes('/llm/plan-h3-')) {
        const body = route.request().postDataJSON(); llmRequests.push(body);
        return route.fulfill({json: {source_prompt:body.prompt, signature:'new', planning_style:'adaptive',
          window_frames:345, effective_window_frames:345, window_count:6, windows:[], window_prompts:[], planning_warnings:[]}});
      }
      if (endpoint.startsWith('/api/')) return route.fulfill({json: {jobs:[], configured:true, loaded:true}});
      return route.fulfill({contentType:'text/html', body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
    });
    await page.goto('http://review.test');
    const assets = path.join(root, 'ui/dist/assets');
    await page.addStyleTag({content: fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(f => f.endsWith('.css'))), 'utf8')});
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => window.resetFixture());
    await assertEnhancementReview(page, output, llmRequests);
    assert.deepEqual(errors, []);
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exit(1);});
