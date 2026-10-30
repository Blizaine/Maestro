// Exercise keying through the real Editor preview and Inspector. All HTTP is
// intercepted; source images/video are generated locally and never uploaded.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {createRequire} = require('node:module');
const zlib = require('node:zlib');
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

const samplePixels = [
  [0, 255, 0, 255],       // exact green key
  [255, 0, 0, 255],       // opaque red subject
  [80, 255, 0, 255],      // soft green edge
  [255, 0, 0, 128],       // source alpha survives keying
  [200, 220, 80, 255],    // green spill
  [200, 80, 220, 255],    // blue spill
  [0, 0, 255, 255],
  [32, 64, 96, 255],
];

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, payload) {
  const name = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(payload.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([name, payload])));
  return Buffer.concat([length, name, payload, checksum]);
}

function png(width, height, pixels) {
  const rows = [];
  for (let y = 0; y < height; y++) {
    rows.push(Buffer.from([0]));
    for (let x = 0; x < width; x++) rows.push(Buffer.from(pixels[x]));
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 6; // RGBA
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', zlib.deflateSync(Buffer.concat(rows))),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function createSyntheticVideo(directory) {
  const output = path.join(directory, 'editor-keying-source.mp4');
  const ffmpeg = process.env.MAESTRO_FFMPEG || 'ffmpeg';
  const result = spawnSync(ffmpeg, [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'lavfi', '-i', 'color=c=0x00ff00:s=64x64:r=24:d=0.5',
    '-f', 'lavfi', '-i', 'color=c=0xff0000:s=64x64:r=24:d=0.5',
    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=1',
    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
    '-map', '[v]', '-map', '2:a', '-t', '1',
    '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-shortest', '-movflags', '+faststart', output,
  ], {encoding: 'utf8', timeout: 30000, maxBuffer: 2 * 1024 * 1024});
  assert.equal(result.status, 0, result.stderr || result.error?.message || 'FFmpeg fixture generation failed');
  return fs.readFileSync(output);
}

function makeProject() {
  const asset = (id, name, type, url, width, height, duration, hasAudio) => ({
    id, name, type, origin: 'output', workspace: 'Keying test',
    path: `/outputs/${name}`, url, duration, width, height,
    fps: type === 'video' ? 24 : 0, has_audio: hasAudio, size: 100,
  });
  return {
    schema_version: 6,
    id: 'editor-keying-test',
    name: 'Keying UI test',
    workspace: 'Keying test',
    revision: 1,
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {
      still: asset('still', 'key-fixture.png', 'image', '/media/key-fixture.png', 8, 2, 0, false),
      video: asset('video', 'key-video.mp4', 'video', '/media/original.mp4', 64, 64, 1, true),
    },
    tracks: [
      {
        id: 'image-track', name: 'Image layer', type: 'video', z_index: 0,
        muted: false, locked: false, volume: 1,
        items: [{
          id: 'image-item', asset_id: 'still', name: 'Still', start: 1, duration: 2,
          source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain',
          transform: {x: 0, y: 0, scale: 1, rotation: 0},
          color: {saturation: 0}, muted: false, disabled: false,
        }],
      },
      {
        id: 'video-track', name: 'Video layer', type: 'video', z_index: 1,
        muted: false, locked: false, volume: 1,
        items: [{
          id: 'video-item', asset_id: 'video', name: 'Video', start: 10, duration: 1,
          source_in: 0, speed: 1, volume: 1, opacity: 1, fit: 'contain',
          transform: {x: 0, y: 0, scale: 1, rotation: 0},
          muted: false, disabled: false,
        }],
      },
      {id: 'audio-track', name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false, volume: 1, items: []},
    ],
    markers: [],
    export: {
      quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true,
      resolution: 'canvas', frame_rate: 'project', filename: '',
    },
    exports: [],
  };
}

function appCss() {
  const assets = path.join(ui, 'dist/assets');
  const cssFile = fs.readdirSync(assets).find(name => name.endsWith('.css'));
  assert.ok(cssFile, 'Build the UI before running this browser check');
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
        import {EditorPreview} from './src/editor/EditorPreview';
        import {EditorInspector} from './src/editor/EditorInspector';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
        function App() {
          return <main style={{display:'flex',gap:16,width:1180,height:780,padding:12}}>
            <div style={{width:800,height:740,minWidth:0}}><EditorPreview /></div>
            <aside style={{width:350,height:740,overflow:'auto'}}><EditorInspector compact /></aside>
          </main>;
        }
        const root = createRoot(document.getElementById('root'));
        window.mountEditorKeying = () => root.render(<App />);
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

async function readKeyCanvas(page, itemId = 'image-item') {
  return page.evaluate(itemId => {
    const canvas = document.querySelector(`[data-editor-canvas-item-id="${itemId}"] [data-editor-keyed-canvas]`);
    if (!canvas) return null;
    const width = canvas.width, height = canvas.height;
    if (canvas.dataset.keyRenderer === 'webgl') {
      const gl = canvas.getContext('webgl');
      if (!gl || gl.isContextLost()) return {width, height, renderer: 'webgl-lost', pixels: []};
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      return {width, height, renderer: 'webgl', pixels: [...pixels]};
    }
    const context = canvas.getContext('2d');
    return {width, height, renderer: 'cpu', pixels: [...context.getImageData(0, 0, width, height).data]};
  }, itemId);
}

(async () => {
  const fixtureDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'maestro-editor-keying-'));
  let browser;
  const failures = [];
  const check = (condition, message) => { if (!condition) failures.push(message); };
  try {
    const videoBytes = createSyntheticVideo(fixtureDirectory);
    const imageBytes = png(8, 2, samplePixels);
    const alternatePixels = samplePixels.map(([r, g, b, a]) => [r, g, b, a]);
    alternatePixels[1] = [0, 0, 255, 255];
    const alternateImageBytes = png(8, 2, alternatePixels);
    const project = makeProject();
    const bundle = await bundleEditor();
    const launchOptions = {
      headless: true,
      args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'],
    };
    if (process.env.MAESTRO_CHROME) launchOptions.executablePath = process.env.MAESTRO_CHROME;
    else if (process.platform === 'win32') launchOptions.channel = 'chrome';
    browser = await chromium.launch(launchOptions);

    async function openEditor({forceCpu = false} = {}) {
      const page = await browser.newPage({viewport: {width: 1440, height: 900}});
      page.setDefaultTimeout(8000);
      const previewRequests = [];
      const saves = [];
      const pageErrors = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.addInitScript(force => {
        try { localStorage.setItem('maestro-editor-preview-quality', 'mobile'); } catch {}
        if (!force) return;
        const original = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function (type, ...args) {
          if (type === 'webgl' && this.hasAttribute('data-editor-keyed-canvas')) return null;
          return original.call(this, type, ...args);
        };
      }, forceCpu);
      await page.route('**/*', async route => {
        const request = route.request();
        const url = new URL(request.url());
        const json = body => route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        });
        if (url.pathname === '/' || url.pathname === '/index.html') {
          return route.fulfill({
            contentType: 'text/html',
            body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>',
          });
        }
        if (url.pathname === '/media/key-fixture.png') return route.fulfill({status: 200, contentType: 'image/png', body: imageBytes});
        if (url.pathname === '/media/alternate.png') return route.fulfill({status: 200, contentType: 'image/png', body: alternateImageBytes});
        if (url.pathname.endsWith('.mp4')) {
          const range = request.headers().range?.match(/^bytes=(\d+)-(\d*)$/);
          if (range) {
            const start = Number(range[1]);
            const end = Math.min(videoBytes.length - 1, range[2] ? Number(range[2]) : videoBytes.length - 1);
            const body = videoBytes.subarray(start, end + 1);
            return route.fulfill({
              status: 206,
              headers: {
                'Content-Type': 'video/mp4',
                'Accept-Ranges': 'bytes',
                'Content-Range': `bytes ${start}-${end}/${videoBytes.length}`,
                'Content-Length': String(body.length),
              },
              body,
            });
          }
          return route.fulfill({
            status: 200,
            headers: {'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Content-Length': String(videoBytes.length)},
            body: videoBytes,
          });
        }
        if (url.pathname.startsWith('/api/')) {
          if (url.pathname === '/api/v1/editor/media/preview') {
            const body = request.postDataJSON();
            previewRequests.push(body);
            return json({
              thumbnail_url: '/media/key-fixture.png',
              proxy_url: body.include_proxy ? '/media/mobile-proxy.mp4' : null,
            });
          }
          if (url.pathname === '/api/v1/editor/projects/editor-keying-test' && request.method() === 'PUT') {
            const body = request.postDataJSON();
            saves.push(body);
            return json({...body.project, revision: body.project.revision + 1});
          }
          if (url.pathname === '/api/v1/editor/projects' && request.method() === 'GET') {
            return json({projects: [{
              id: project.id, name: project.name, workspace: project.workspace,
              created_at: project.created_at, updated_at: project.updated_at,
              duration: 11, asset_count: 2,
            }]});
          }
          return json({});
        }
        return route.fulfill({status: 204, body: ''});
      });
      await page.goto('http://editor-keying.test');
      await page.addStyleTag({content: appCss()});
      await page.addScriptTag({content: bundle});
      await page.evaluate(initialProject => {
        window.editorStore.setState({
          workspace: initialProject.workspace,
          project: initialProject,
          selectedItemId: 'image-item',
          selectedItemIds: ['image-item'],
          selectedTrackId: 'image-track',
          playhead: 1.5,
          playing: false,
          dirty: false,
          saving: false,
          history: [],
          future: [],
          error: null,
        });
        window.mountEditorKeying();
      }, project);
      return {page, previewRequests, saves, pageErrors};
    }

    const gpu = await openEditor();
    const page = gpu.page;
    await page.waitForFunction(() => {
      const image = document.querySelector('img[src="/media/key-fixture.png"]');
      return image?.complete && image.naturalWidth === 8;
    });
    await page.getByLabel('Enable chroma key').check();
    await page.waitForFunction(() => {
      const canvas = document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]');
      return canvas?.dataset.keyPreviewReady === 'true';
    });
    const gpuPixels = await readKeyCanvas(page);
    check(gpuPixels?.renderer === 'webgl', `Expected real WebGL keying, got ${gpuPixels?.renderer || 'no canvas'}`);
    if (gpuPixels?.renderer === 'webgl') {
      const at = index => gpuPixels.pixels.slice(index * 4, index * 4 + 4);
      const green = at(0), red = at(1), edge = at(2), sourceAlpha = at(3), greenSpill = at(4);
      check(green[3] <= 3, `Pure green should be transparent; got ${green}`);
      check(red[0] >= 252 && red[1] <= 3 && red[2] <= 3 && red[3] >= 252, `Opaque red should remain; got ${red}`);
      check(Math.abs(edge[3] - 80) <= 3, `Soft edge matte should be near 80/255; got ${edge[3]}`);
      check(Math.abs(sourceAlpha[3] - 128) <= 3, `Original source alpha should survive; got ${sourceAlpha[3]}`);
      check(Math.abs(greenSpill[1] - 180) <= 3, `Green spill should fall to about 180; got ${greenSpill[1]}`);
    }
    const cssFilter = await page.locator('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]').evaluate(canvas => canvas.style.filter);
    check(cssFilter.includes('url('), `Color adjustments should remain after keying; canvas filter was ${cssFilter}`);

    await page.getByRole('button', {name: 'Blue', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');
    const bluePixels = await readKeyCanvas(page);
    if (bluePixels) {
      const blueSpill = bluePixels.pixels.slice(5 * 4, 5 * 4 + 4);
      check(Math.abs(blueSpill[2] - 180) <= 3, `Blue spill should fall to about 180; got ${blueSpill[2]}`);
    }
    await page.getByRole('button', {name: 'Green', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');

    await page.locator('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]').evaluate(canvas => {
      window.__keyCanvasBeforeInactive = canvas;
      canvas.dataset.keyPreviewReady = 'false';
    });
    await page.evaluate(() => window.editorStore.getState().setPlayhead(0.5));
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"]')?.style.opacity === '0');
    await page.evaluate(() => window.editorStore.getState().setPlayhead(1.5));
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');
    check(await page.locator('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]').evaluate(canvas => canvas === window.__keyCanvasBeforeInactive),
      'A prepared image canvas should resume on the same element after active=false → true');

    const originalAfterLoad = await page.evaluate(async () => {
      const image = document.querySelector('img[src="/media/key-fixture.png"]');
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      context.drawImage(image, 0, 0);
      return [...context.getImageData(0, 0, canvas.width, canvas.height).data.slice(0, 4)];
    });
    check(originalAfterLoad[3] === 255, `The original source must stay opaque when read directly; got ${originalAfterLoad}`);

    await page.evaluate(() => {
      const state = window.editorStore.getState();
      state.selectItem('video-item', 'video-track');
      state.setPlayhead(10.25);
    });
    await page.waitForFunction(() => {
      const video = document.querySelector('video[data-editor-item-id="video-item"]');
      return video && video.currentSrc.endsWith('/mobile-proxy.mp4');
    });
    const videoBeforeKey = await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => {
      window.__videoNode = video;
      return {src: video.currentSrc, muted: video.muted};
    });
    check(videoBeforeKey.src.endsWith('/mobile-proxy.mp4'), `Mobile preview should initially use the mocked proxy; got ${videoBeforeKey.src}`);
    await page.getByLabel('Enable chroma key').check();
    await page.waitForFunction(() => document.querySelector('video[data-editor-item-id="video-item"]')?.currentSrc.endsWith('/original.mp4'));
    await page.waitForTimeout(250);
    const keyedVideoNode = await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => video === window.__videoNode);
    check(keyedVideoNode, 'Enabling keying should preserve the existing video element');
    check(gpu.previewRequests.filter(request => request.include_proxy).length === 1,
      'Keying should not schedule an additional proxy-preview request');
    check(await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => video.currentSrc.endsWith('/original.mp4')),
      'Keyed video should use its original source URL even in mobile preview quality');

    await page.waitForFunction(() => {
      const video = document.querySelector('video[data-editor-item-id="video-item"]');
      return video?.readyState >= 2;
    }, undefined, {timeout: 10000}).catch(() => {});
    const hasDecodedVideo = await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => video.readyState >= 2);
    if (hasDecodedVideo) {
      await page.evaluate(async () => {
        const video = document.querySelector('video[data-editor-item-id="video-item"]');
        video.pause();
        const seeked = new Promise(resolve => video.addEventListener('seeked', resolve, {once: true}));
        window.editorStore.getState().setPlayhead(10.75);
        await seeked;
      });
      await page.waitForFunction(() => {
        const video = document.querySelector('video[data-editor-item-id="video-item"]');
        const canvas = document.querySelector('[data-editor-canvas-item-id="video-item"] [data-editor-keyed-canvas]');
        return video?.currentTime >= 0.7 && canvas?.dataset.keyPreviewReady === 'true';
      });
      const redFrame = await readKeyCanvas(page, 'video-item');
      const redFrameAlpha = redFrame?.pixels[(32 * redFrame.width + 32) * 4 + 3];
      const redFrameDebug = await page.evaluate(() => {
        const video = document.querySelector('video[data-editor-item-id="video-item"]');
        const canvas = document.querySelector('[data-editor-canvas-item-id="video-item"] [data-editor-keyed-canvas]');
        const sample = document.createElement('canvas');
        sample.width = sample.height = 1;
        const context = sample.getContext('2d');
        context.drawImage(video, 0, 0, 1, 1);
        return {
          time: video.currentTime, ready: video.readyState, paused: video.paused,
          src: video.currentSrc, source: [...context.getImageData(0, 0, 1, 1).data],
          renderer: canvas?.dataset.keyRenderer, outputReady: canvas?.dataset.keyPreviewReady,
        };
      });
      check(redFrameAlpha >= 252, `Seeking to the red source frame should redraw an opaque matte; got alpha ${redFrameAlpha}; ${JSON.stringify(redFrameDebug)}`);
      const videoTime = await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => video.currentTime);
      await page.getByLabel('Preview quality').selectOption('original');
      await page.getByLabel('Enable chroma key').uncheck();
      const afterVideoToggle = await page.locator('video[data-editor-item-id="video-item"]').evaluate(video => ({
        same: video === window.__videoNode, time: video.currentTime, muted: video.muted,
      }));
      check(afterVideoToggle.same, 'Disabling keying should preserve the video element and its embedded audio source');
      check(Math.abs(afterVideoToggle.time - videoTime) < 0.08, `A key toggle at original quality should preserve the video clock; ${videoTime} → ${afterVideoToggle.time}`);
      check(afterVideoToggle.muted === videoBeforeKey.muted, 'Key toggles should not change the video audio mute state');
    } else {
      failures.push('Synthetic H264 fixture did not reach HAVE_CURRENT_DATA; seek/redraw checks could not run');
    }

    // Detaching audio from a keyed clip must preserve visual-only settings on
    // the source while carrying volume automation onto the linked audio item.
    await page.evaluate(() => {
      const store = window.editorStore;
      store.setState({
        selectedItemId: 'video-item', selectedItemIds: ['video-item'],
        selectedTrackId: 'video-track', playhead: 10.25,
      });
      store.getState().updateItem('video-item', {
        keying: {enabled: true, color: '#00ff00', similarity: 0.2, softness: 0.1, spill: 0.4},
        color: {saturation: 0.75},
        keyframes: {
          volume: [
            {time: 0, value: 0.25, easing: 'linear'},
            {time: 1, value: 0.75, easing: 'linear'},
          ],
          opacity: [
            {time: 0, value: 0.5, easing: 'linear'},
            {time: 1, value: 1, easing: 'linear'},
          ],
        },
      });
      store.getState().detachSelectedAudio();
      const state = store.getState();
      const source = state.project.tracks.find(track => track.id === 'video-track').items[0];
      const audio = state.project.tracks.find(track => track.type === 'audio').items[0];
      window.__detachedAudioSnapshot = {
        sourceKeying: source.keying,
        sourceColor: source.color,
        sourceMuted: source.muted,
        audioKeying: audio?.keying,
        audioColor: audio?.color,
        audioKeyframes: audio?.keyframes,
      };
    });
    const detached = await page.evaluate(() => window.__detachedAudioSnapshot);
    check(detached.sourceKeying?.enabled === true, 'Audio detach should keep chroma key settings on the original video');
    check(detached.sourceColor?.saturation === 0.75, 'Audio detach should keep color adjustments on the original video');
    check(detached.sourceMuted === true, 'Audio detach should mute the original embedded audio');
    check(detached.audioKeying === undefined && detached.audioColor === undefined,
      'Detached audio should not carry visual keying or color fields');
    check(detached.audioKeyframes?.volume?.length === 2 && !detached.audioKeyframes?.opacity,
      'Detached audio should retain volume automation without visual opacity automation');
    await page.evaluate(() => window.editorStore.getState().undo());
    const detachedUndo = await page.evaluate(() => {
      const state = window.editorStore.getState();
      const sourceTrack = state.project.tracks.find(track => track.id === 'video-track');
      const audioTrack = state.project.tracks.find(track => track.type === 'audio');
      const source = sourceTrack?.items.find(item => item.id === 'video-item');
      return {
        sourceKeying: source?.keying,
        sourceMuted: source?.muted,
        audioCount: audioTrack?.items.length,
        tracks: state.project.tracks.map(track => [track.id, track.type, track.items.length]),
      };
    });
    check(detachedUndo.sourceKeying?.enabled === true && detachedUndo.sourceMuted === false && detachedUndo.audioCount === 0,
      `Undo should remove detached audio and restore keyed video state: ${JSON.stringify(detachedUndo)}`);

    // Re-open an image selection and verify save, undo/redo, disabled identity,
    // and track-lock behavior through the real Inspector and Zustand actions.
    await page.evaluate(() => {
      const store = window.editorStore;
      const project = store.getState().project;
      const tracks = project.tracks.map(track => track.id === 'image-track'
        ? {...track, items: track.items.map(item => item.id === 'image-item'
          ? {...item, keying: {enabled: false, color: '#00ff00', similarity: 0.15, softness: 0.1, spill: 0.5}}
          : item)}
        : track);
      window.editorStore.setState({
        selectedItemId: 'image-item', selectedItemIds: ['image-item'],
        selectedTrackId: 'image-track', playhead: 1.5,
        project: {...project, tracks},
      });
    });
    await page.waitForFunction(() => !document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]'));
    await page.getByLabel('Enable chroma key').check();
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');
    const persisted = await page.evaluate(() => window.editorStore.getState().saveProject());
    await page.waitForFunction(() => window.editorStore.getState().saving === false);
    check(persisted?.tracks[0]?.items[0]?.keying?.enabled === true, 'Saving should retain the authored keying object');
    check(gpu.saves.at(-1)?.project?.tracks[0]?.items[0]?.keying?.enabled === true,
      'The mocked project-save request should persist keying settings');
    await page.evaluate(() => window.editorStore.getState().undo());
    await page.evaluate(() => window.editorStore.setState({
      selectedItemId: 'image-item', selectedItemIds: ['image-item'], selectedTrackId: 'image-track',
    }));
    await page.waitForFunction(() => !document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]'));
    await page.evaluate(() => window.editorStore.getState().redo());
    await page.evaluate(() => window.editorStore.setState({
      selectedItemId: 'image-item', selectedItemIds: ['image-item'], selectedTrackId: 'image-track',
    }));
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');
    check(await page.getByLabel('Enable chroma key').isChecked(), 'Redo should restore the keying settings');
    await page.getByLabel('Enable chroma key').uncheck();
    await page.waitForFunction(() => !document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]'));
    const originalAfterDisable = await page.evaluate(() => {
      const image = document.querySelector('img[src="/media/key-fixture.png"]');
      return {opacity: image.style.opacity, src: image.currentSrc};
    });
    check(originalAfterDisable.opacity === '1', 'Disabling keying should restore the original image layer');

    await page.evaluate(() => window.editorStore.getState().toggleTrackLock('image-track'));
    check(await page.getByLabel('Enable chroma key').isDisabled(), 'Locked tracks should disable the keying toggle');
    const lockedProject = await page.evaluate(() => window.editorStore.getState().project);
    await page.getByLabel('Enable chroma key').click({force: true}).catch(() => {});
    check(await page.evaluate(() => window.editorStore.getState().project.tracks[0].items[0].keying?.enabled === false),
      'A locked keying control must not change project settings');
    await page.evaluate(() => window.editorStore.getState().toggleTrackLock('image-track'));

    // Compare renderer outputs for every RGBA channel, then lose a real
    // WebGL context to exercise the same CPU fallback used by older devices.
    const cpu = await openEditor({forceCpu: true});
    await cpu.page.getByLabel('Enable chroma key').check();
    await cpu.page.waitForFunction(() => {
      const canvas = document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]');
      return canvas?.dataset.keyRenderer === 'cpu' && canvas.dataset.keyPreviewReady === 'true';
    });
    const cpuPixels = await readKeyCanvas(cpu.page);
    check(cpuPixels?.renderer === 'cpu', `Forced fallback should use the 2D renderer; got ${cpuPixels?.renderer}`);
    if (gpuPixels?.renderer === 'webgl' && cpuPixels?.renderer === 'cpu') {
      const mismatches = [];
      for (let pixel = 0; pixel < Math.min(gpuPixels.pixels.length, cpuPixels.pixels.length) / 4; pixel++) {
        const offset = pixel * 4;
        const visible = Math.max(gpuPixels.pixels[offset + 3], cpuPixels.pixels[offset + 3]) > 0;
        for (let channel = 0; channel < 4; channel++) {
          if (channel < 3 && !visible) continue;
          if (Math.abs(gpuPixels.pixels[offset + channel] - cpuPixels.pixels[offset + channel]) > 3) {
            mismatches.push([offset + channel, gpuPixels.pixels[offset + channel], cpuPixels.pixels[offset + channel]]);
          }
        }
      }
      check(gpuPixels.width === cpuPixels.width && gpuPixels.height === cpuPixels.height,
        `WebGL/CPU canvas dimensions should match: ${gpuPixels.width}×${gpuPixels.height} vs ${cpuPixels.width}×${cpuPixels.height}`);
      check(mismatches.length === 0, `WebGL and CPU pixels differ by more than 3 bytes: ${JSON.stringify(mismatches.slice(0, 5))}; GPU=${JSON.stringify(gpuPixels.pixels.slice(0, 32))}; CPU=${JSON.stringify(cpuPixels.pixels.slice(0, 32))}`);
    }
    await page.evaluate(() => {
      window.editorStore.setState({
        selectedItemId: 'image-item', selectedItemIds: ['image-item'], selectedTrackId: 'image-track',
      });
      window.editorStore.getState().setPlayhead(1.5);
    });
    await page.getByLabel('Enable chroma key').check();
    await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyPreviewReady === 'true');
    const contextLossWorked = await page.evaluate(() => {
      const canvas = document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]');
      const gl = canvas?.getContext('webgl');
      const extension = gl?.getExtension('WEBGL_lose_context');
      if (!extension) return false;
      extension.loseContext();
      return true;
    });
    if (contextLossWorked) {
      await page.waitForFunction(() => document.querySelector('[data-editor-canvas-item-id="image-item"] [data-editor-keyed-canvas]')?.dataset.keyRenderer === 'cpu');
      check(Boolean(await page.getByText('Reduced-resolution key preview').count()),
        'A lost WebGL context should switch the active preview to the CPU renderer');
    } else {
      failures.push('Browser did not expose WEBGL_lose_context; forced CPU fallback still ran');
    }

    check(gpu.pageErrors.length === 0, `WebGL page errors: ${gpu.pageErrors.join('; ')}`);
    check(cpu.pageErrors.length === 0, `CPU fallback page errors: ${cpu.pageErrors.join('; ')}`);
    await cpu.page.close();
    await page.close();
    assert.deepEqual(failures, []);
    console.log('Editor keying passed: GPU/CPU pixels, source alpha, despill, Inspector persistence/undo/lock, seek redraw and original-source playback.');
  } finally {
    await browser?.close();
    fs.rmSync(fixtureDirectory, {recursive: true, force: true});
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
