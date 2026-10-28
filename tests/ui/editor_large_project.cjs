// Exercise large-project timeline culling in the real Editor workspace.
// All HTTP is mocked; synthetic SVGs are data URIs and no media work is issued.
// Run `npm run build` from ui/ first, then `node tests/ui/editor_large_project.cjs`.
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

const CLIPS_PER_TRACK = 600;
const TRACK_COUNT = 4;
const ASSET_COUNT = 20;
const PROJECT_ID = 'editor-large-project';
const WORKSPACE = 'Large project fixtures';

function svgDataUri(index) {
  const hue = (index * 37) % 360;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64"><rect width="64" height="64" fill="hsl(${hue},70%,45%)"/><circle cx="32" cy="32" r="18" fill="#fff"/><text x="32" y="37" text-anchor="middle" font-family="sans-serif" font-size="12">${index}</text></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function makeProject() {
  const assets = Object.fromEntries(Array.from({length: ASSET_COUNT}, (_, index) => {
    const id = `svg-${String(index).padStart(2, '0')}`;
    return [id, {
      id,
      name: `${id}.svg`,
      type: 'image',
      origin: 'upload',
      path: `fixtures/${id}.svg`,
      url: svgDataUri(index),
      duration: 1,
      width: 64,
      height: 64,
      has_audio: false,
      // Offline metadata prevents the preview hook from asking the mocked API
      // to create a thumbnail. The timeline still receives the synthetic SVG URL.
      missing: true,
    }];
  }));

  const tracks = [
    {id: 'video-main', name: 'Video 1', type: 'video', z_index: 1, muted: false, locked: false},
    {id: 'video-overlay', name: 'Video 2', type: 'video', z_index: 2, muted: false, locked: false},
    {id: 'audio-main', name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false},
    {id: 'titles-main', name: 'Titles', type: 'text', z_index: 10, muted: false, locked: false},
  ].map((track, trackIndex) => ({
    ...track,
    items: Array.from({length: CLIPS_PER_TRACK}, (_, index) => {
      const start = index * 2;
      const common = {
        id: `track-${trackIndex}-clip-${String(index).padStart(4, '0')}`,
        name: `${track.name} clip ${String(index).padStart(4, '0')}`,
        start,
        duration: 1,
        source_in: 0,
        speed: 1,
        volume: 1,
        opacity: 1,
        fit: 'contain',
        transform: {x: 0, y: 0, scale: 1, rotation: 0},
        fade_in: 0,
        fade_out: 0,
      };
      if (track.type === 'text') {
        return {
          ...common,
          text: `Title ${String(index).padStart(4, '0')}`,
          style: {
            x: 0, y: 0, font_size: 48, font_family: 'Arial', color: '#ffffff',
            background_color: '#000000', background_opacity: 0.2, text_align: 'center',
          },
        };
      }
      return {
        ...common,
        ...(track.type === 'audio' ? {} : {
          asset_id: `svg-${String(index % ASSET_COUNT).padStart(2, '0')}`,
        }),
      };
    }),
  }));

  return {
    schema_version: 6,
    revision: 1,
    id: PROJECT_ID,
    name: 'Large timeline culling fixture',
    workspace: WORKSPACE,
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
    assets,
    tracks,
    markers: [],
    export: {
      quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
      resolution: 'canvas', frame_rate: 'project', filename: 'fixture.mp4',
      spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
    },
    exports: [],
  };
}

const project = makeProject();
const projectSummary = {
  id: project.id,
  name: project.name,
  workspace: project.workspace,
  created_at: project.created_at,
  updated_at: project.updated_at,
  duration: (CLIPS_PER_TRACK - 1) * 2 + 1,
  asset_count: ASSET_COUNT,
};

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'build the UI before running this large-project check');
  return fs.readFileSync(path.join(assets, cssFile), 'utf8');
}

async function bundleEditor() {
  const sourceExtensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of sourceExtensions) {
      if (fs.existsSync(candidate + extension)) return fs.realpathSync(candidate + extension);
    }
    for (const extension of sourceExtensions) {
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile)) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const fsResolver = {
    name: 'workspace-filesystem-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved
            ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`}]};
        }
        const resolved = requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]});
        return {path: fs.realpathSync(resolved), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx'
          : extension === '.ts' ? 'ts'
            : extension === '.jsx' ? 'jsx'
              : extension === '.json' ? 'json'
                : extension === '.css' ? 'css'
                  : extension === '.svg' ? 'file'
                    : 'js';
        return {
          contents: fs.readFileSync(args.path, 'utf8'),
          loader,
          resolveDir: path.dirname(args.path),
        };
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: `
        import {createRoot} from 'react-dom/client';
        import {EditorWorkspace} from './src/editor/EditorWorkspace';
        import {useStore} from './src/stores/useStore';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.store = useStore;
        window.editorStore = useEditorStore;
        window.mountEditor = () => createRoot(document.getElementById('root')).render(<EditorWorkspace/>);
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
    plugins: [fsResolver],
  });
  return result.outputFiles[0].text;
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? {
      executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    } : {}),
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 960}});
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      const url = new URL(request.url());
      requests.push({method: request.method(), path: url.pathname});
    });
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      const key = url.pathname;
      const json = body => route.fulfill({json: body});
      if (key === '/') {
        return route.fulfill({contentType: 'text/html', body:
          '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><div id="root" style="display:flex;height:100dvh;width:100vw"></div>'});
      }
      if (key === '/api/v1/editor/projects') return json({projects: [projectSummary]});
      if (key === `/api/v1/editor/projects/${project.id}`) return json(project);
      if (key === '/api/v1/editor/media/status') return json({assets: []});
      if (key === '/api/v1/workspaces') return json({workspaces: [{name: WORKSPACE, path: '/fixtures'}], active: WORKSPACE});
      if (key === '/api/v1/outputs') return json({outputs: [], total: 0});
      if (key === '/api/v1/director/pipelines') return json({pipelines: []});
      if (key.startsWith('/api/')) return json({items: [], outputs: [], jobs: [], pipelines: [], projects: [], status: 'available'});
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://editor-large-project.test');
    await page.addStyleTag({content: appCss()});
    await page.addScriptTag({content: await bundleEditor()});
    await page.evaluate(() => {
      window.store.setState({activeWorkspace: 'Large project fixtures', sidebarOpen: false});
      window.fixtureMountStart = performance.now();
      window.mountEditor();
    });

    await page.getByTestId('editor-layout').waitFor();
    await page.getByTestId('editor-inspector-pane').waitFor();
    await page.waitForFunction(id => window.editorStore.getState().project?.id === id, PROJECT_ID);
    const firstClip = page.locator('[data-editor-item-id="track-0-clip-0000"]');
    await firstClip.waitFor({state: 'attached'});
    const initialMetrics = await page.evaluate(() => ({
      mountToFirstClipMs: performance.now() - window.fixtureMountStart,
      mountedClipDomCount: document.querySelectorAll('[data-editor-item-id]').length,
      timelineScrollWidth: document.querySelector('[data-testid="editor-timeline-scroll"]').scrollWidth,
      timelineClientWidth: document.querySelector('[data-testid="editor-timeline-scroll"]').clientWidth,
      projectItems: window.editorStore.getState().project.tracks.reduce((count, track) => count + track.items.length, 0),
      projectAssets: Object.keys(window.editorStore.getState().project.assets).length,
      projectTracks: window.editorStore.getState().project.tracks.length,
    }));
    assert.equal(initialMetrics.projectItems, CLIPS_PER_TRACK * TRACK_COUNT);
    assert.equal(initialMetrics.projectAssets, ASSET_COUNT);
    assert.equal(initialMetrics.projectTracks, TRACK_COUNT);
    assert.ok(initialMetrics.mountedClipDomCount < 100,
      `initial offscreen culling should keep clip DOM below 100 (got ${initialMetrics.mountedClipDomCount})`);

    const distantItemId = `track-0-clip-${String(CLIPS_PER_TRACK - 1).padStart(4, '0')}`;
    await page.evaluate(() => {
      const scroll = document.querySelector('[data-testid="editor-timeline-scroll"]');
      window.fixtureFarScrollStart = performance.now();
      scroll.scrollLeft = scroll.scrollWidth;
      scroll.dispatchEvent(new Event('scroll', {bubbles: true}));
    });
    await page.locator(`[data-editor-item-id="${distantItemId}"]`).waitFor({state: 'attached'});
    await page.waitForFunction(id => {
      const node = document.querySelector(`[data-editor-item-id="${id}"]`);
      return node && node.getBoundingClientRect().width > 0;
    }, distantItemId);
    const farMetrics = await page.evaluate(id => {
      const scroll = document.querySelector('[data-testid="editor-timeline-scroll"]');
      return {
        farScrollRenderMs: performance.now() - window.fixtureFarScrollStart,
        mountedClipDomCount: document.querySelectorAll('[data-editor-item-id]').length,
        scrollLeft: scroll.scrollLeft,
        maxScrollLeft: scroll.scrollWidth - scroll.clientWidth,
        firstClipStillMounted: Boolean(document.querySelector('[data-editor-item-id="track-0-clip-0000"]')),
        distantClipMounted: Boolean(document.querySelector(`[data-editor-item-id="${id}"]`)),
      };
    }, distantItemId);
    assert.ok(farMetrics.scrollLeft > 0, 'timeline moved to its far end');
    assert.ok(farMetrics.distantClipMounted, 'the distant clip is rendered after scrolling');
    assert.ok(!farMetrics.firstClipStillMounted, 'the offscreen initial clip is unmounted after scrolling');
    assert.ok(farMetrics.mountedClipDomCount < 100,
      `far-scroll culling should keep clip DOM below 100 (got ${farMetrics.mountedClipDomCount})`);

    await page.locator(`[data-editor-item-id="${distantItemId}"]`).click();
    await page.waitForFunction(id => window.editorStore.getState().selectedItemId === id, distantItemId);
    assert.equal(await page.evaluate(() => window.editorStore.getState().selectedItemId), distantItemId,
      'a clip at the far end of the large project remains selectable');

    const mediaOrWriteRequests = requests.filter(({method, path: requestPath}) => (
      (method !== 'GET' && requestPath !== '/api/v1/editor/media/status')
      || /\/(?:upload|export|generate|generation|jobs?)(?:\/|$)/i.test(requestPath)
      || /\/editor\/media\/(?:preview|proxy|waveform|thumbnail)(?:\/|$)/i.test(requestPath)
      || /\/media\/(?:probe|preview|thumbnail|waveform)(?:\/|$)/i.test(requestPath)
    ));
    const mediaStatusRequests = requests.filter(({method, path: requestPath}) => (
      method === 'POST' && requestPath === '/api/v1/editor/media/status'
    ));
    assert.deepEqual(mediaOrWriteRequests, [],
      `fixture must not submit writes, media work, or generation: ${JSON.stringify(mediaOrWriteRequests)}`);
    assert.equal(mediaStatusRequests.length, 1,
      'the only media endpoint call is the mocked metadata availability check during workspace initialization');
    assert.deepEqual(errors, [], `Editor workspace raised browser errors: ${errors.join('; ')}`);

    console.log(JSON.stringify({
      fixture: {clips: initialMetrics.projectItems, tracks: initialMetrics.projectTracks, syntheticSvgAssets: initialMetrics.projectAssets},
      mount: initialMetrics,
      farScroll: farMetrics,
      distantSelection: distantItemId,
      requestCounts: {
        totalHttp: requests.length,
        apiGets: requests.filter(request => request.path.startsWith('/api/') && request.method === 'GET').length,
        mockedMediaStatus: mediaStatusRequests.length,
        forbidden: mediaOrWriteRequests.length,
      },
    }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
