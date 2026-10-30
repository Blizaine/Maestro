// Exercise the real Editor workspace against mocked HTTP so layout checks never
// create, save, upload, export, or generate media in a user's workspace.
// Run `npm run build` from ui/ first, then `node tests/ui/editor_layout.cjs`.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT
  || (() => {
    try {
      return require.resolve('playwright', {paths: [path.join(root, 'ui')]});
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
      return require.resolve('playwright');
    }
  })();
const {chromium} = require(playwrightPath);

const project = {
  schema_version: 1,
  id: 'editor-layout-project',
  name: 'Layout regression project',
  workspace: 'Layout fixtures',
  created_at: 1,
  updated_at: 1,
  canvas: {width: 1920, height: 1080, fps: 30, background: '#000000'},
  assets: {},
  tracks: [
    {id: 'video-1', name: 'Video 1', type: 'video', z_index: 1, muted: false, locked: false, items: []},
    {id: 'audio-1', name: 'Audio 1', type: 'audio', z_index: 0, muted: false, locked: false, items: []},
  ],
  markers: [],
  export: {
    quality: 'balanced', codec: 'h264', encoder: 'auto', include_audio: true,
    resolution: 'canvas', frame_rate: 'project', filename: 'edit.mp4',
    spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 1,
  },
  exports: [],
};
const projectSummary = {
  id: project.id,
  name: project.name,
  workspace: project.workspace,
  created_at: 1,
  updated_at: 1,
  duration: 0,
  asset_count: 0,
};

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'build the UI before running this layout check');
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
    page.on('pageerror', error => errors.push(error.message));
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
      if (key === '/api/v1/workspaces') return json({workspaces: [{name: project.workspace, path: '/fixtures'}], active: project.workspace});
      if (key === '/api/v1/outputs') return json({outputs: [], total: 0});
      if (key === '/api/v1/director/pipelines') return json({pipelines: []});
      if (key.startsWith('/api/')) return json({items: [], outputs: [], jobs: [], pipelines: [], projects: [], status: 'available'});
      return route.fulfill({status: 204, body: ''});
    });
    await page.goto('http://editor-layout.test');
    await page.addStyleTag({content: appCss()});
    await page.addScriptTag({content: await bundleEditor()});
    await page.evaluate(() => {
      window.store.setState({activeWorkspace: 'Layout fixtures', sidebarOpen: false});
      window.mountEditor();
    });

    await page.getByTestId('editor-layout').waitFor();
    await page.getByTestId('editor-inspector-pane').waitFor();
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-layout"]')?.dataset.layoutMode === 'desktop');

    const bounds = selector => page.locator(selector).boundingBox();
    const mediaBox = await bounds('[data-testid="editor-media-pane"]');
    const previewBox = await bounds('[data-testid="editor-preview-pane"]');
    const inspectorBox = await bounds('[data-testid="editor-inspector-pane"]');
    const timelineBox = await bounds('[data-testid="editor-timeline-pane"]');
    assert.ok(mediaBox && previewBox && inspectorBox && timelineBox, 'desktop renders all three panes and the timeline');
    assert.ok(mediaBox.x + mediaBox.width <= previewBox.x + 1, 'Media remains to the left of the viewer');
    assert.ok(previewBox.x + previewBox.width <= inspectorBox.x + 1, 'Inspector remains to the right of the viewer');
    assert.ok(Math.abs(timelineBox.x - mediaBox.x) < 1, 'timeline begins at the left edge of the workspace');
    assert.ok(Math.abs(timelineBox.width - (mediaBox.width + previewBox.width + inspectorBox.width + 16)) < 24,
      'timeline spans the full workspace width');

    const dragSeparator = async (testId, dx, dy) => {
      const box = await bounds(`[data-testid="${testId}"]`);
      assert.ok(box, `${testId} is available`);
      const x = box.x + box.width / 2;
      const y = box.y + box.height / 2;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x + dx, y + dy, {steps: 6});
      await page.mouse.up();
    };
    const leftHandle = page.getByTestId('editor-resize-left');
    await dragSeparator('editor-resize-left', 72, 0);
    await page.waitForFunction(() => Number(document.querySelector('[data-testid="editor-media-pane"]')?.getBoundingClientRect().width) > 350);
    await leftHandle.focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => Number(document.querySelector('[data-testid="editor-media-pane"]')?.getBoundingClientRect().width) > 370);

    await dragSeparator('editor-resize-right', -44, 0);
    await page.waitForFunction(() => Number(document.querySelector('[data-testid="editor-inspector-pane"]')?.getBoundingClientRect().width) > 350);
    await dragSeparator('editor-resize-timeline', 0, -48);
    await page.waitForFunction(() => Number(document.querySelector('[data-testid="editor-timeline-pane"]')?.getBoundingClientRect().height) > 330);
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('maestro-editor-layout-v1')));
    assert.ok(saved.leftWidth > 350 && saved.rightWidth > 350 && saved.timelineHeight > 330,
      'pointer and keyboard resizing persist panel sizes in browser-local storage');

    await page.setViewportSize({width: 1024, height: 900});
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-layout"]')?.dataset.layoutMode === 'intermediate');
    assert.equal(await page.getByTestId('editor-inspector-pane').count(), 0, 'intermediate widths collapse the persistent Inspector pane');
    await page.getByTestId('editor-inspector-trigger').click();
    const inspectorDrawer = page.getByTestId('editor-inspector-drawer');
    await inspectorDrawer.waitFor();
    const drawerBox = await inspectorDrawer.boundingBox();
    const previewAtIntermediate = await bounds('[data-testid="editor-preview-pane"]');
    assert.ok(drawerBox && previewAtIntermediate && drawerBox.width <= 360,
      'the contextual Inspector opens as a bounded drawer on intermediate screens');
    await page.keyboard.press('Escape');
    await inspectorDrawer.waitFor({state: 'detached'});

    await page.setViewportSize({width: 800, height: 720});
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-layout"]')?.dataset.layoutMode === 'intermediate');
    const narrowMedia = await bounds('[data-testid="editor-media-pane"]');
    const narrowPreview = await bounds('[data-testid="editor-preview-pane"]');
    assert.ok(narrowMedia && narrowPreview && narrowPreview.width > 0, 'narrow intermediate layouts keep the viewer usable');
    await page.getByTestId('editor-layout-reset').click();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('maestro-editor-layout-v1')).leftWidth === 292);

    await page.setViewportSize({width: 390, height: 844});
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-layout"]')?.dataset.layoutMode === 'mobile');
    assert.equal(await page.getByTestId('editor-resize-left').count(), 0, 'mobile uses touch controls instead of narrow resize handles');
    await page.getByTestId('editor-mobile-inspector').click();
    const mobileDrawer = page.getByTestId('editor-mobile-drawer');
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-mobile-drawer"]')?.getAttribute('aria-hidden') === 'false');
    await mobileDrawer.getByRole('button', {name: 'Media', exact: true}).click();
    await mobileDrawer.getByRole('button', {name: 'Adjust', exact: true}).click();
    await page.getByTestId('editor-mobile-timeline').click();
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-mobile-drawer"]')?.getAttribute('aria-hidden') === 'true');
    assert.ok(await page.getByTestId('editor-mobile-media').isVisible(), 'mobile exposes labeled Media, Timeline, and Adjust controls');

    const touch = await page.context().newCDPSession(page);
    await touch.send('Emulation.setTouchEmulationEnabled', {enabled: true, maxTouchPoints: 5});
    await page.setViewportSize({width: 844, height: 390});
    await page.waitForFunction(() => document.querySelector('[data-testid="editor-layout"]')?.dataset.layoutMode === 'mobile');
    const landscapeViewer = await bounds('[data-testid="editor-preview-pane"]');
    const landscapeTimeline = await bounds('[data-testid="editor-timeline-pane"]');
    assert.ok(landscapeViewer && landscapeTimeline && landscapeViewer.width > 250 && landscapeViewer.height > 100,
      'touch landscape retains a usable viewer');
    assert.ok(Math.abs(landscapeViewer.y - landscapeTimeline.y) < 2 && landscapeTimeline.x > landscapeViewer.x,
      'short landscape places viewer and timeline side by side');
    assert.ok(landscapeTimeline.y + landscapeTimeline.height <= 390, 'landscape timeline stays inside the viewport');

    assert.deepEqual(errors, [], `Editor workspace raised browser errors: ${errors.join('; ')}`);
    console.log('Editor workspace layout regression passed (desktop resizing and persistence, intermediate Inspector drawer, narrow and mobile controls).');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
