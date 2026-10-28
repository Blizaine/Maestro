// Studio source-aspect regression for video models that support Auto ratio.
// The generation API request is intercepted locally; no model/GPU job starts.
// Run: node tests/ui/studio_explicit_aspect.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);

(async () => {
  const extensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.svg'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of extensions) {
      const withExtension = candidate + extension;
      if (fs.existsSync(withExtension) && fs.statSync(withExtension).isFile()) return fs.realpathSync(withExtension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'studio-aspect-test-workspace-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved
            ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`} ]};
        }
        return {path: fs.realpathSync(requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]})), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx' : extension === '.ts' ? 'ts'
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json'
            : extension === '.svg' ? 'file' : 'js';
        return {contents: fs.readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const bundle = await esbuild.build({
    absWorkingDir: ui,
    stdin: {
      contents: `
        import {createRoot} from 'react-dom/client';
        import {useStore} from './src/stores/useStore';
        import {submitGeneration} from './src/api/client';
        import {AspectRatioGrid} from './src/components/Sidebar/AspectRatioGrid';
        window.store = useStore;
        window.sendGeneration = submitGeneration;
        window.mount = () => createRoot(document.getElementById('root')).render(<AspectRatioGrid />);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    format: 'iife',
    target: 'es2022',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
    plugins: [resolver],
  });

  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage();
    const submissions = [];
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname === '/api/v1/generate' && route.request().method() === 'POST') {
        submissions.push(route.request().postDataJSON());
        return route.fulfill({json: {job_id: `synthetic-aspect-${submissions.length}`, status: 'queued'}});
      }
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {jobs: [], outputs: [], models: [], total: 0}});
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://192.168.56.23:38474');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => {
      window.store.setState(state => ({
        generationMode: 'video',
        modelOptions: {
          model_type: 'fixture-source-video',
          supports_auto_aspect: true,
          resolution_presets: {
            auto: {values: {auto: 'auto_720p', '16:9': 'auto_720p', '9:16': 'auto_720p'}},
            '720p': {values: {auto: 'auto_720p', '16:9': '1280x720', '9:16': '720x1280'}},
          },
        },
        resolutionPreset: '720p',
        aspectRatio: '16:9',
        params: {...state.params, model_type: 'fixture-source-video', prompt: 'Synthetic source-aware aspect fixture', resolution: '1280x720'},
      }));
      window.mount();
    });

    await page.evaluate(() => window.store.getState().setResolutionPreset('auto'));
    await page.getByRole('button', {name: 'Auto', exact: true}).waitFor();
    const autoState = await page.evaluate(() => ({
      preset: window.store.getState().resolutionPreset,
      aspect: window.store.getState().aspectRatio,
      resolution: window.store.getState().params.resolution,
    }));
    assert.deepEqual(autoState, {preset: 'auto', aspect: 'auto', resolution: 'auto_720p'},
      'A source-aware video model keeps the 720p Auto resolution and its Auto aspect selected.');
    await page.evaluate(() => window.sendGeneration(window.store.getState().params));
    await page.waitForFunction(() => window.store.getState().params.resolution === 'auto_720p');
    assert.equal(submissions.length, 1);
    assert.equal(submissions[0].resolution, 'auto_720p', 'Auto stays a source-aware resolution in the intercepted request.');

    await page.getByRole('button', {name: '9:16', exact: true}).click();
    const portraitState = await page.evaluate(() => ({
      preset: window.store.getState().resolutionPreset,
      aspect: window.store.getState().aspectRatio,
      resolution: window.store.getState().params.resolution,
    }));
    assert.deepEqual(portraitState, {preset: '720p', aspect: '9:16', resolution: '720x1280'},
      'Choosing portrait from Auto switches to a fixed 720p portrait preset.');
    await page.evaluate(() => window.sendGeneration(window.store.getState().params));
    await page.waitForFunction(() => window.store.getState().params.resolution === '720x1280');
    assert.equal(submissions.length, 2);
    assert.equal(submissions[1].resolution, '720x1280', 'The intercepted Studio request sends fixed portrait dimensions.');
    assert.deepEqual(pageErrors, []);
    console.log('Studio explicit aspect regression passed: source-aware Auto remains selected until a fixed 9:16 request is chosen and sent.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
