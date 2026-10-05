// Frames video input posters use the managed thumbnail endpoint after upload.
// The browser app, upload, and image responses are mocked; this test does not
// contact the running Maestro service or generate media.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {useGalleryInputs} from './src/lib/galleryInputs';
    import {InputsPanel} from './src/components/Sidebar/InputsPanel';
    import {MediaInputCard} from './src/components/Sidebar/MediaInputCard';
    import {FileUploadZone} from './src/components/shared/FileUploadZone';
    window.store=useStore; window.galleryInputs=useGalleryInputs;
    const root=createRoot(document.getElementById('root'));
    window.mount=()=>{
      const state=useStore.getState();
      useStore.setState({generationMode:'video',studioVideoWorkflow:'frames',sidebarMode:'studio',
        params:{...state.params,model_type:'minimax_h3_fused_turbo',image_mode:0,
          image_start:undefined,image_end:undefined,image_refs:undefined,frames_positions:undefined,
          video_prompt_type:'GV',audio_prompt_type:'K',video_guide:undefined},
        models:[],enabledModels:new Set(),modelOptions:{fps:24,supports_end_frame:false,
          audio_prompt_type_sources:{choices:[['Control video','K']]}},
        startImage:null,endImage:null,imageRefs:[],durationSeconds:4,
        slidingWindowSeconds:4,slidingWindowOverlap:0,continueVideo:null,continueVideoPath:'',continueVideoUrl:''});
      root.render(<><InputsPanel/>
        <section><FileUploadZone label="Control video" accept=".mp4" filename="advanced.mp4"
          videoUrl="/api/v1/uploads/advanced.mp4" onFile={()=>{}} onClear={()=>{}}/></section>
        <section><MediaInputCard title="Reference clip" subtitle="Video" kind="video"
          mediaUrl="/api/v1/file/reference.mp4?workspace=Origin%20A" expanded={false} editorId="ref-settings"
          onEdit={()=>{}} onRemove={()=>{}}/></section>
      </>);
    };
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});

  const browser = await playwright.chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 900, height: 850}});
    page.setDefaultTimeout(5000);
    const errors = [], posterRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body:
        '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>'});
      if (url.pathname === '/api/v1/upload') {
        const body = route.request().postDataBuffer()?.toString('latin1') || '';
        const name = /filename="([^"]+)"/.exec(body)?.[1] || 'control.mp4';
        return route.fulfill({json: {filename: name, path: `C:\\Maestro\\uploads\\${name}`, url: `/api/v1/uploads/${name}`}});
      }
      if (url.pathname.startsWith('/api/v1/thumbnail/')) {
        posterRequests.push(`${url.pathname}${url.search}`);
        return route.fulfill({contentType: 'image/jpeg', body: png});
      }
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {}});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto('http://frames-video-input.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(() => window.mount());

    const nativeFrameInput = page.getByLabel('Choose a frame image');
    assert.equal(await nativeFrameInput.count(), 1, 'Frames keeps its mounted native image input');
    const registered = await page.evaluate(async () => {
      const input = window.galleryInputs.getState().targets.find(target => target.kind === 'video' && target.label === 'control video');
      await input.receive(new File(['video fixture'], 'control.mp4', {type: 'video/mp4'}));
      return window.store.getState().params.video_guide;
    });
    assert.match(registered, /control\.mp4$/, 'upload is stored through the existing control-video handler');
    await page.waitForFunction(() => {
      const poster = document.querySelector('img[alt="Control video"]');
      return poster?.complete && poster.naturalWidth > 0;
    });
    assert.equal(await page.locator('img[alt="Control video"]').getAttribute('src'),
      '/api/v1/thumbnail/control.mp4?size=480', 'the Frames tile shows the server first-frame poster');
    assert.equal(await nativeFrameInput.evaluate(node => node.isConnected), true,
      'upload rerenders do not detach the dedicated frame image picker');

    // Advanced upload zones can consume the uploads URL, and H3 reference
    // cards keep the source workspace attached to their poster request.
    await page.waitForFunction(() => Array.from(document.querySelectorAll('img')).some(image =>
      image.src.includes('/api/v1/thumbnail/advanced.mp4?workspace=__uploads__&size=480') && image.complete && image.naturalWidth > 0));
    await page.waitForFunction(() => Array.from(document.querySelectorAll('img')).some(image =>
      image.src.includes('/api/v1/thumbnail/reference.mp4?workspace=Origin+A&size=480') && image.complete && image.naturalWidth > 0));
    assert.ok(posterRequests.includes('/api/v1/thumbnail/reference.mp4?workspace=Origin+A&size=480'),
      'poster resolution preserves a reference video’s workspace query');

    await page.getByRole('button', {name: 'Remove', exact: true}).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.video_guide), undefined,
      'removing the video preview keeps the existing clear action');
    assert.equal(await page.getByLabel('Choose a frame image').count(), 1,
      'the frame picker remains mounted after removing the video');
    assert.deepEqual(errors, [], 'video input previews render without browser errors');
    console.log('Frames, advanced upload, and reference video inputs request server first-frame posters; remove and frame-picker behavior stays intact');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
