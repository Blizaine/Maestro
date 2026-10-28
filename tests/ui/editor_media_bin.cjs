// Responsive Editor media-bin regression. All HTTP is intercepted and all
// media, projects, uploads, and Director rows are synthetic fixtures.
// Run: node tests/ui/editor_media_bin.cjs
// Run with MAESTRO_MEDIA_BIN_BASELINE=1 to record the pre-redesign header size.
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

const BASELINE_HEADER_HEIGHT = {320: 293, 260: 293, 390: 293};
const WORKSPACE_A = 'Workspace A';
const WORKSPACE_B = 'Workspace B';

const makeAsset = (id, name, type, workspace, extra = {}) => ({
  id, name, display_name: name, type, origin: 'output', workspace,
  path: `/outputs/${name}`, url: `/media/${name}`, duration: type === 'video' ? 8 : type === 'audio' ? 12 : 0,
  width: type === 'audio' ? 0 : 1280, height: type === 'audio' ? 0 : 720,
  fps: type === 'video' ? 24 : 0, has_audio: type !== 'image', size: 100, ...extra,
});

const usedAsset = makeAsset('used-video', 'used-intro.mp4', 'video', WORKSPACE_A);
const favoriteAsset = makeAsset('favorite-image', 'needle-favorite.png', 'image', WORKSPACE_A, {favorite: true});
const workspaceVideo = makeAsset('workspace-video', 'workspace-b-roll.mov', 'video', WORKSPACE_B);
const workspaceAudio = makeAsset('workspace-audio', 'ambient-bed.wav', 'audio', WORKSPACE_B);
const library = [usedAsset, favoriteAsset, workspaceVideo, workspaceAudio];
const project = {
  schema_version: 6, id: 'media-bin-project', name: 'Media bin fixture', workspace: WORKSPACE_A,
  revision: 1, created_at: 1, updated_at: 1,
  canvas: {width: 1280, height: 720, fps: 24, background: '#000000'},
  assets: {[usedAsset.id]: usedAsset},
  tracks: [{
    id: 'fixture-video-track', name: 'Footage', type: 'video', z_index: 0,
    muted: false, locked: false, volume: 1,
    items: [{
      id: 'used-clip', asset_id: usedAsset.id, name: usedAsset.name, start: 0, duration: 8,
      source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain',
      transform: {x: 0, y: 0, scale: 1, rotation: 0},
    }],
  }],
  markers: [],
  export: {quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true, resolution: 'canvas', frame_rate: 'project', filename: ''},
  exports: [],
};
const directorRuns = [{
  id: 'director-fixture-run', workspace: WORKSPACE_A, pipeline_type: 'short_film_story',
  status: 'completed', clip_count: 3, scene_description: 'Synthetic Director package for the harbor scene',
  created_at: 2,
}];

function builtCss() {
  const builtHtml = fs.readFileSync(path.join(ui, 'dist/index.html'), 'utf8');
  const filename = /href="\/assets\/([^\"]+\.css)"/.exec(builtHtml)?.[1];
  assert.ok(filename && fs.existsSync(path.join(ui, 'dist/assets', filename)), 'Build the UI before measuring media-bin CSS.');
  return fs.readFileSync(path.join(ui, 'dist/assets', filename), 'utf8');
}

async function bundleMediaBin() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.svg'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      const withExtension = candidate + extension;
      if (fs.existsSync(withExtension) && fs.statSync(withExtension).isFile()) return fs.realpathSync(withExtension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'editor-media-bin-test-resolver',
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
  const result = await esbuild.build({
    absWorkingDir: ui,
    stdin: {
      contents: `
        import {createRoot} from 'react-dom/client';
        import {EditorMediaBin} from './src/editor/EditorMediaBin';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        const root = createRoot(document.getElementById('root'));
        window.mount = () => root.render(<div id="media-panel" style={{width:'320px',height:'100vh'}}><EditorMediaBin compact /></div>);
        window.setMediaPanelWidth = width => { document.getElementById('media-panel').style.width = width + 'px'; };
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
  return result.outputFiles[0].text;
}

function saveScreenshotDirectory() {
  const directory = path.join(root, '.codex-tmp', 'editor-media-compact-20261028');
  fs.mkdirSync(directory, {recursive: true});
  return directory;
}

async function measure(page, width, baseline = false) {
  await page.setViewportSize({width, height: 900});
  await page.evaluate(panelWidth => window.setMediaPanelWidth(panelWidth), width);
  await page.waitForTimeout(50);
  return page.evaluate(isBaseline => {
    const panel = document.getElementById('media-panel');
    const heading = panel.querySelector('h2');
    const header = isBaseline
      ? heading?.parentElement?.parentElement
      : panel.querySelector('[data-testid="editor-media-bin-header"]');
    const scroller = isBaseline
      ? header?.nextElementSibling
      : panel.querySelector('[data-testid="editor-media-bin-scroller"]');
    const headerRect = header?.getBoundingClientRect();
    const scrollerRect = scroller?.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    return {
      width: Math.round(panelRect.width),
      headerHeight: Math.round(headerRect?.height || 0),
      scrollerTop: Math.round(scrollerRect?.top || 0),
      headerBottom: Math.round(headerRect?.bottom || 0),
      panelScrollWidth: panel.scrollWidth,
      panelClientWidth: panel.clientWidth,
      filterRowHeight: Math.round(panel.querySelector('[data-testid="editor-media-bin-filters"]')?.getBoundingClientRect().height || 0),
    };
  }, baseline);
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 320, height: 900}});
    page.setDefaultTimeout(10_000);
    const pageErrors = [];
    const apiRequests = [];
    const uploads = [];
    const mediaProbes = [];
    const previewRequests = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      }
      if (url.pathname.startsWith('/api/')) {
        apiRequests.push({method: request.method(), path: url.pathname, url: request.url()});
        if (url.pathname === '/api/v1/editor/media/preview') {
          previewRequests.push(request.postDataJSON());
          return route.fulfill({json: {preview_id: 'synthetic-preview', waveform: []}});
        }
        if (url.pathname === '/api/v1/workspaces' && request.method() === 'GET') {
          return route.fulfill({json: {workspaces: [{name: WORKSPACE_A}, {name: WORKSPACE_B}], active: WORKSPACE_A}});
        }
        if (url.pathname === '/api/v1/outputs' && request.method() === 'GET') {
          const workspace = url.searchParams.get('workspace');
          const outputs = workspace === '__uploads__' ? [] : [{
            name: workspace === WORKSPACE_B ? 'refreshed-b.png' : 'refreshed-a.png',
            type: 'image', mode: null, size: 256, created_at: 4, url: '/synthetic/refreshed.png', workspace,
          }];
          return route.fulfill({json: {outputs, total: outputs.length}});
        }
        if (url.pathname === '/api/v1/director/pipelines' && request.method() === 'GET') {
          return route.fulfill({json: {pipelines: []}});
        }
        if (url.pathname.startsWith('/api/v1/file/')) {
          return route.fulfill({status: 204, body: ''});
        }
        if (url.pathname === '/api/v1/upload' && request.method() === 'POST') {
          uploads.push({headers: request.headers(), body: request.postDataBuffer()?.toString('latin1') || ''});
          const filename = /filename="([^"]+)"/.exec(uploads.at(-1).body)?.[1] || 'uploaded-fixture.png';
          return route.fulfill({json: {
            filename, path: `/uploads/${filename}`, url: `/uploads/${filename}`,
            duration_seconds: 0, has_audio: false,
          }});
        }
        if (url.pathname === '/api/v1/editor/media/probe') {
          const body = request.postDataJSON();
          mediaProbes.push(body);
          const isImage = body.asset.name.endsWith('.png');
          return route.fulfill({json: {
            name: body.asset.name, type: isImage ? 'image' : 'video', duration: isImage ? 0 : 6,
            width: isImage ? 640 : 1280, height: isImage ? 480 : 720,
            fps: isImage ? 0 : 24, has_audio: !isImage, audio_channels: 0,
            audio_sample_rate: 0, size: 1234, path: body.asset.path,
          }});
        }
        throw new Error(`Unexpected API request: ${request.method()} ${url.pathname}`);
      }
      return route.fulfill({status: 204, body: ''});
    });

    await page.goto('http://editor-media-bin.test');
    await page.addStyleTag({content: builtCss()});
    await page.addScriptTag({content: await bundleMediaBin()});
    await page.evaluate(({fixture, assets, runs}) => {
      document.documentElement.dataset.theme = 'graphite-studio';
      window.editorStore.setState({
        workspace: fixture.workspace,
        project: fixture,
        projects: [],
        library: assets,
        libraryWorkspaces: ['Workspace A', 'Workspace B'],
        directorRuns: runs,
        selectedItemId: null,
        selectedItemIds: [],
        selectedTrackId: 'fixture-video-track',
        playhead: 3,
        dirty: false,
        history: [],
        future: [],
        error: null,
      });
      window.mount();
    }, {fixture: project, assets: library, runs: directorRuns});
    const screenshotDirectory = saveScreenshotDirectory();

    if (process.env.MAESTRO_MEDIA_BIN_BASELINE === '1') {
      const baseline = [];
      for (const width of [320, 260, 390]) {
        baseline.push(await measure(page, width, true));
        await page.screenshot({path: path.join(screenshotDirectory, `media-bin-before-${width}.png`), fullPage: true});
      }
      console.log(`Pre-redesign media-bin header measurements: ${JSON.stringify(baseline)}`);
      return;
    }

    const layout = {};
    for (const width of [320, 260, 390]) {
      layout[width] = await measure(page, width);
      await page.screenshot({path: path.join(screenshotDirectory, `media-bin-${width}.png`), fullPage: true});
      assert.ok(layout[width].headerHeight <= 180,
        `The media header is compact at ${width}px: ${JSON.stringify(layout[width])}`);
      assert.ok(layout[width].panelScrollWidth <= layout[width].panelClientWidth + 1,
        `The media panel has no horizontal overflow at ${width}px: ${JSON.stringify(layout[width])}`);
      assert.ok(Math.abs(layout[width].headerBottom - layout[width].scrollerTop) <= 1,
        `Media filters sit immediately above the asset scroller at ${width}px: ${JSON.stringify(layout[width])}`);
      assert.ok(layout[width].headerHeight < BASELINE_HEADER_HEIGHT[width] - 50,
        `The media header is substantially shorter than its ${BASELINE_HEADER_HEIGHT[width]}px baseline at ${width}px.`);
    }
    assert.ok(layout[320].headerHeight <= 180, '320px desktop header meets the requested height target.');
    await measure(page, 260);
    const narrowHeaderLabels = await page.evaluate(() => {
      const panel = document.getElementById('media-panel');
      const heading = panel.querySelector('h2');
      const used = panel.querySelector('[aria-label="Used in project"]');
      return {
        media: {clientWidth: heading.clientWidth, scrollWidth: heading.scrollWidth},
        used: {clientWidth: used.clientWidth, scrollWidth: used.scrollWidth},
      };
    });
    assert.ok(narrowHeaderLabels.media.clientWidth >= 32
      && narrowHeaderLabels.media.scrollWidth <= narrowHeaderLabels.media.clientWidth + 1,
    `The Media heading remains readable at 260px: ${JSON.stringify(narrowHeaderLabels)}`);
    assert.ok(narrowHeaderLabels.used.clientWidth >= 30
      && narrowHeaderLabels.used.scrollWidth <= narrowHeaderLabels.used.clientWidth + 1,
    `The Used label remains readable at 260px: ${JSON.stringify(narrowHeaderLabels)}`);
    await measure(page, 320);

    const header = page.getByTestId('editor-media-bin-header');
    const scroller = page.getByTestId('editor-media-bin-scroller');
    const filters = page.getByTestId('editor-media-bin-filters');
    const toolbar = page.getByTestId('editor-media-bin-toolbar');
    for (const [name, value] of [
      ['All media', 'all'], ['Video', 'video'], ['Image', 'image'],
      ['Audio', 'audio'], ['Director', 'director'], ['Favorites', 'favorites'],
    ]) {
      const button = filters.getByRole('button', {name, exact: true});
      assert.equal(await button.getAttribute('aria-pressed'), String(value === 'all'));
    }
    assert.equal(await page.getByRole('button', {name: 'Grid view'}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByRole('button', {name: 'List view'}).getAttribute('aria-pressed'), 'false');
    assert.equal(await page.getByRole('button', {name: 'Refresh media'}).count(), 1);
    assert.equal(await page.getByRole('button', {name: 'Used in project'}).getAttribute('aria-pressed'), 'false');
    const initialSize = page.getByLabel('Media thumbnail size');
    assert.equal(await initialSize.count(), 1);
    assert.ok((await initialSize.boundingBox()).height >= 34, 'The thumbnail size slider has a comfortable touch target.');
    assert.equal(await page.getByRole('textbox', {name: 'Search media'}).count(), 1);
    assert.equal(await page.getByRole('combobox', {name: 'Filter media by workspace'}).count(), 1);
    assert.deepEqual(await header.locator(':scope > *').evaluateAll(elements => elements.map(element => element.dataset.testid || 'row')),
      ['row', 'row', 'editor-media-bin-toolbar', 'editor-media-bin-filters'],
      'The compact header has a search/workspace row, an action row, and filters as its final row.');

    for (const [name, expected] of [
      ['All media', ['used-intro.mp4', 'needle-favorite.png', 'workspace-b-roll.mov', 'ambient-bed.wav']],
      ['Video', ['used-intro.mp4', 'workspace-b-roll.mov']],
      ['Image', ['needle-favorite.png']],
      ['Audio', ['ambient-bed.wav']],
      ['Favorites', ['needle-favorite.png']],
      ['Director', []],
    ]) {
      const button = filters.getByRole('button', {name, exact: true});
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'), 'true', `${name} exposes its pressed state.`);
      const visibleNames = await scroller.locator('[data-testid="editor-media-asset"]').allTextContents();
      for (const expectedName of expected) assert.ok(visibleNames.some(value => value.includes(expectedName)), `${name} shows ${expectedName}.`);
      assert.equal(visibleNames.length, expected.length, `${name} shows only matching assets.`);
    }
    await filters.getByRole('button', {name: 'Director', exact: true}).click();
    assert.ok((await scroller.innerText()).includes('Synthetic Director package for the harbor scene'),
      'Director filter shows the synthetic Director package.');

    await filters.getByRole('button', {name: 'All media', exact: true}).click();
    await page.getByRole('combobox', {name: 'Filter media by workspace'}).selectOption(WORKSPACE_B);
    let workspaceNames = await scroller.locator('[data-testid="editor-media-asset"]').allTextContents();
    assert.equal(workspaceNames.length, 2, 'Workspace filter restricts the media library.');
    assert.ok(workspaceNames.every(text => text.includes('workspace-b-roll.mov') || text.includes('ambient-bed.wav')));
    await page.getByRole('combobox', {name: 'Filter media by workspace'}).selectOption('all');

    await page.getByRole('textbox', {name: 'Search media'}).fill('needle');
    const searchNames = await scroller.locator('[data-testid="editor-media-asset"]').allTextContents();
    assert.equal(searchNames.length, 1, 'Search narrows the library by media name.');
    assert.ok(searchNames[0].includes('needle-favorite.png'));
    await page.getByRole('textbox', {name: 'Search media'}).fill('');

    const usedFilter = page.getByRole('button', {name: 'Used in project'});
    await usedFilter.click();
    assert.equal(await usedFilter.getAttribute('aria-pressed'), 'true');
    const usedNames = await scroller.locator('[data-testid="editor-media-asset"]').allTextContents();
    assert.equal(usedNames.length, 1, 'Used-in-project filter matches current timeline assets.');
    assert.ok(usedNames[0].includes('used-intro.mp4'));
    await usedFilter.click();
    assert.equal(await usedFilter.getAttribute('aria-pressed'), 'false');

    await page.getByRole('button', {name: 'List view'}).click();
    assert.equal(await page.getByRole('button', {name: 'List view'}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByLabel('Media thumbnail size').count(), 0, 'Thumbnail sizing is grid-only.');
    assert.equal(await scroller.evaluate(element => getComputedStyle(element).display), 'block');
    await page.getByRole('button', {name: 'Grid view'}).click();
    assert.equal(await page.getByRole('button', {name: 'Grid view'}).getAttribute('aria-pressed'), 'true');
    const thumbnailSize = page.getByLabel('Media thumbnail size');
    await thumbnailSize.evaluate(element => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(element, '180');
      element.dispatchEvent(new Event('input', {bubbles: true}));
      element.dispatchEvent(new Event('change', {bubbles: true}));
    });
    assert.equal(await thumbnailSize.inputValue(), '180');
    assert.match(await scroller.evaluate(element => element.style.gridTemplateColumns), /180px/,
      'The thumbnail size control changes grid card sizing.');

    const toolbarButtons = [
      page.getByRole('button', {name: 'Upload media'}),
      page.getByRole('button', {name: 'Add text title'}),
      page.getByRole('button', {name: 'Create animated title'}),
    ];
    for (const button of toolbarButtons) {
      const bounds = await button.boundingBox();
      assert.ok(bounds && bounds.height >= 34, `Toolbar buttons remain touch-usable: ${JSON.stringify(bounds)}`);
    }
    const filterBounds = await filters.getByRole('button', {name: 'All media'}).boundingBox();
    assert.ok(filterBounds && filterBounds.height >= 32, `Filter buttons remain touch-usable: ${JSON.stringify(filterBounds)}`);
    assert.equal(await toolbar.getByRole('button', {name: 'Upload media'}).getAttribute('title'), 'Upload media');

    await page.getByRole('button', {name: 'Create animated title'}).click();
    const titleDialog = page.getByRole('dialog', {name: 'Create animated title'});
    await titleDialog.waitFor();
    await page.getByRole('button', {name: 'Close animated title'}).click();
    await titleDialog.waitFor({state: 'detached'});

    await page.getByRole('button', {name: 'Add text title'}).click();
    const titleText = await page.evaluate(() => {
      const project = window.editorStore.getState().project;
      return project.tracks.flatMap(track => track.items).find(item => item.text === 'Your title')?.text;
    });
    assert.equal(titleText, 'Your title', 'The compact Text action still adds an editable timeline title.');

    const file = {name: 'uploaded-fixture.png', mimeType: 'image/png', buffer: Buffer.from('synthetic media-bin upload')};
    await page.locator('input[type="file"]').setInputFiles(file);
    await page.waitForFunction(() => window.editorStore.getState().library.some(asset => asset.display_name === 'uploaded-fixture.png'));
    assert.equal(uploads.length, 1, 'Upload action sends one intercepted media upload.');
    assert.match(uploads[0].body, /filename="uploaded-fixture\.png"/);
    assert.equal(mediaProbes.length, 1, 'The uploaded file is probed before it is inserted.');
    assert.ok(await page.evaluate(() => window.editorStore.getState().project.tracks.flatMap(track => track.items)
      .some(item => item.asset_id && window.editorStore.getState().project.assets[item.asset_id]?.name === 'uploaded-fixture.png')),
    'Upload still adds the returned asset to the project.');

    await page.getByRole('button', {name: 'Refresh media'}).click();
    await page.waitForFunction(() => window.editorStore.getState().library.some(asset => asset.name === 'refreshed-a.png'));
    assert.ok(apiRequests.some(request => request.path === '/api/v1/workspaces'));
    assert.equal(apiRequests.filter(request => request.path === '/api/v1/outputs').length, 3,
      'Refresh reloads both workspace libraries and the virtual Uploads library.');
    assert.ok(await page.evaluate(() => window.editorStore.getState().library.some(asset => asset.name === 'refreshed-b.png')),
      'Refresh returns updated synthetic workspace media to the library.');

    const generationRequests = apiRequests.filter(request => /generate|generation|plan|download|restart/i.test(request.path));
    assert.deepEqual(generationRequests, [], 'Media-bin interactions never launch models, downloads, or server restarts.');
    assert.deepEqual(pageErrors, [], `No browser runtime errors occur: ${pageErrors.join('; ')}`);
    assert.ok(apiRequests.every(request => new URL(request.url).hostname === 'editor-media-bin.test'),
      'All HTTP traffic stays on the fully intercepted test origin.');
    console.log(`Editor media-bin regression passed; header heights ${JSON.stringify(Object.fromEntries(Object.entries(layout).map(([width, info]) => [width, info.headerHeight])))}.`);
    console.log(`Screenshots: ${screenshotDirectory}`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
