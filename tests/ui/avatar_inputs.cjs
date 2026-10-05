// LongCat Avatar inputs run in a model-free browser fixture. Upload requests
// are intercepted here; this test never contacts a running Maestro instance.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'playwright');

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

(async () => {
  const contents = [
    "import React from 'react';",
    "import { createRoot } from 'react-dom/client';",
    "import { useStore } from './src/stores/useStore';",
    "import { useGalleryInputs } from './src/lib/galleryInputs';",
    "import * as helpers from './src/lib/avatarWorkflow';",
    "import { AvatarControls } from './src/components/Sidebar/AvatarControls';",
    "import { GenerateButton } from './src/components/Sidebar/GenerateButton';",
    'window.store = useStore; window.galleryInputs = useGalleryInputs; window.avatarHelpers = helpers;',
    "const root = createRoot(document.getElementById('root'));",
    'let mountId = 0;',
    'window.mount = (modelType, seed = {}) => {',
    '  const state = useStore.getState();',
    "  const params = { ...state.params, model_type: modelType, image_mode: 0, image_start: undefined, image_refs: undefined, video_prompt_type: '', audio_guide: undefined, audio_guide2: undefined, speakers_locations: undefined, ...seed.params };",
    '  const referenceFiles = seed.referenceFiles ? [new File([\"reference\"], seed.referenceFiles, { type: \"image/png\" })] : [];',
    '  useStore.setState({ generationMode: \"video\", studioVideoWorkflow: \"avatar\", studioVideoEffectiveCreateRoute: \"avatar\", sidebarMode: \"studio\", models: [{ model_type: \"longcat_avatar\", name: \"LongCat Avatar\", family: \"longcat\", architecture: \"longcat_avatar\", fps: 24, is_i2v: true, is_t2v: false, supports_audio_input: true, guidance_max_phases: 1 }, { model_type: \"longcat_avatar_multi\", name: \"LongCat Avatar Multi\", family: \"longcat\", architecture: \"longcat_avatar\", fps: 24, is_i2v: true, is_t2v: false, supports_audio_input: true, guidance_max_phases: 1 }], enabledModels: new Set([\"longcat_avatar\", \"longcat_avatar_multi\"]), params, startImage: null, imageRefs: seed.imageRefs || referenceFiles, imageRefType: seed.imageRefType || \"\", audioGuideFilename: null, audioGuide2Filename: null });',
    '  root.render(<><AvatarControls key={++mountId} /><section aria-label="Generation actions"><GenerateButton stretch /></section></>);',
    '};',
  ].join('\n');
  const bundle = await esbuild.build({
    stdin: { contents, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' },
    logLevel: 'silent',
  });

  const browser = await playwright.chromium.launch({
    headless: true,
    ...(process.platform === 'win32'
      ? { executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe' }
      : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 760, height: 900 } });
    page.setDefaultTimeout(5000);
    const pageErrors = [];
    const uploadedNames = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    const slowClearStarted = deferred();
    const slowClearRelease = deferred();
    const slowWorkflowStarted = deferred();
    const slowWorkflowRelease = deferred();
    const slowModelStarted = deferred();
    const slowModelRelease = deferred();
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');

    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') {
        return route.fulfill({
          contentType: 'text/html',
          body: '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>',
        });
      }
      if (url.pathname === '/api/v1/upload-audio') {
        const body = route.request().postDataBuffer()?.toString('latin1') || '';
        const name = /filename="([^"]+)"/.exec(body)?.[1] || 'voice.wav';
        uploadedNames.push(name);
        if (name === 'slow-clear.wav') {
          slowClearStarted.resolve();
          await slowClearRelease.promise;
        } else if (name === 'slow-workflow.wav') {
          slowWorkflowStarted.resolve();
          await slowWorkflowRelease.promise;
        } else if (name === 'slow-model.wav') {
          slowModelStarted.resolve();
          await slowModelRelease.promise;
        }
        return route.fulfill({
          json: {
            filename: name,
            path: 'C:\\Maestro\\uploads\\' + name,
            url: '/api/v1/uploads/' + name,
            duration_seconds: 2,
          },
        });
      }
      if (url.pathname.startsWith('/api/v1/file/')) {
        return route.fulfill({ contentType: 'image/png', body: png });
      }
      if (url.pathname.startsWith('/api/')) return route.fulfill({ json: {} });
      return route.fulfill({ status: 404, body: '' });
    });

    await page.goto('http://avatar-inputs.test');
    await page.addScriptTag({ content: bundle.outputFiles[0].text });
    await page.evaluate(() => window.mount('longcat_avatar', {
      params: {
        image_start: 'C:\\Maestro\\uploads\\restored-anchor.png',
        audio_guide: 'C:\\Maestro\\uploads\\restored-voice.wav',
      },
    }));

    await page.waitForFunction(() => window.galleryInputs.getState().targets.length >= 2);
    const generationActions = page.getByRole('region', { name: 'Generation actions' });
    assert.equal(await generationActions.locator('button').count(), 2,
      'fixture exposes Generate and Add to Queue actions');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), false,
      'restored anchor and Voice 1 make Single generation available');
    assert.equal(await generationActions.getByRole('button', { name: 'Add current Studio settings to the queue' }).isDisabled(), false,
      'restored Single inputs enable Add to Queue');
    assert.equal(await page.getByLabel('Choose Voice audio #2').count(), 0,
      'Single shows only the first voice input');
    assert.equal(await page.getByRole('region', { name: 'Speaker regions' }).count(), 0,
      'Single hides speaker regions');
    assert.equal(await page.locator('img[alt="Anchor image preview"]').getAttribute('src'),
      '/api/v1/file/restored-anchor.png', 'restored anchor paths use the managed file URL');
    assert.equal(await page.getByText('restored-voice.wav', { exact: true }).count(), 1,
      'restored voice labels are derived from their stored path');

    const singleTargets = await page.evaluate(() => window.galleryInputs.getState().targets.map(target => ({
      kind: target.kind, label: target.label,
    })));
    assert.ok(singleTargets.some(target => target.kind === 'image' && target.label === 'avatar anchor image'),
      'the anchor image registers as a Gallery image destination');
    assert.ok(singleTargets.some(target => target.kind === 'audio' && target.label === 'Voice audio #1'),
      'Voice audio #1 registers as a Gallery audio destination');
    assert.ok(!singleTargets.some(target => target.label === 'Voice audio #2'),
      'Single does not register a second voice destination');

    await page.getByRole('button', { name: 'Remove anchor image' }).click();
    await page.getByRole('button', { name: 'Remove Voice audio #1' }).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.image_start), undefined,
      'clearing a restored anchor removes its stored path');
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide), undefined,
      'clearing a restored voice removes its stored path');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'Generate stays disabled after clearing required inputs');
    assert.equal(await generationActions.getByRole('button', { name: 'Add to queue unavailable' }).isDisabled(), true,
      'Add to Queue stays disabled after clearing required inputs');

    const imageBytes = Buffer.from('anchor fixture');
    await page.getByLabel('Choose anchor image').setInputFiles({
      name: 'anchor.png', mimeType: 'image/png', buffer: imageBytes,
    });
    await page.waitForFunction(() => window.store.getState().startImage?.name === 'anchor.png');
    assert.equal(await page.locator('img[alt="Anchor image preview"]').count(), 1,
      'a selected anchor shows a preview');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'an anchor alone does not enable Single generation without Voice 1');
    const galleryImage = await page.evaluate(async () => {
      const target = window.galleryInputs.getState().targets.find(item => item.label === 'avatar anchor image');
      const accepted = await target.receive(new File(['gallery image'], 'gallery-anchor.png', { type: 'image/png' }));
      const image = target.getImages()[0];
      return { accepted, name: image instanceof File ? image.name : image.name };
    });
    assert.equal(galleryImage.accepted, true, 'Gallery image reuse uses the same anchor handler');
    assert.equal(galleryImage.name, 'gallery-anchor.png', 'the active anchor is exposed to Gallery source capture');

    await page.getByLabel('Choose Voice audio #1').setInputFiles({
      name: 'single-voice.wav', mimeType: 'audio/wav', buffer: Buffer.from('voice fixture'),
    });
    await page.waitForFunction(() => window.store.getState().params.audio_guide?.endsWith('single-voice.wav'));
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), false,
      'anchor plus Voice 1 enables Generate for Single');
    assert.equal(await generationActions.getByRole('button', { name: 'Add current Studio settings to the queue' }).isDisabled(), false,
      'anchor plus Voice 1 enables Add to Queue for Single');
    assert.ok(uploadedNames.includes('single-voice.wav'), 'voice upload is intercepted by the fixture');
    assert.equal(await page.getByText('single-voice.wav', { exact: true }).count(), 1,
      'successful upload stores a visible voice label');
    await page.getByRole('button', { name: 'Remove Voice audio #1' }).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide), undefined,
      'clearing an uploaded voice removes its path');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'clearing Voice 1 blocks Generate again');
    await page.getByRole('button', { name: 'Remove anchor image' }).click();
    assert.equal(await page.evaluate(() => window.store.getState().startImage), null,
      'clearing an uploaded anchor uses the existing start-image setter');

    await page.evaluate(() => window.mount('longcat_avatar_multi', {
      params: {
        image_refs: ['C:\\Maestro\\uploads\\reference-anchor.png'],
        video_prompt_type: 'KI',
      },
      imageRefType: 'KI',
    }));
    await page.waitForFunction(() =>
      window.store.getState().params.speakers_locations === '0:0:50:100 50:0:100:100');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'Multi with an anchor and valid regions still needs both voices');
    assert.equal(await page.getByLabel('Choose Voice audio #2').count(), 1,
      'Multi shows the second voice input');
    assert.equal(await page.getByRole('region', { name: 'Speaker regions' }).count(), 1,
      'Multi shows its editable speaker regions');
    assert.equal(await page.locator('img[alt="Anchor image preview"]').getAttribute('src'),
      '/api/v1/file/reference-anchor.png', 'active restored reference paths use managed file URLs');
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, video_prompt_type: 'KFI' },
    })));
    await page.waitForFunction(() => !document.querySelector('img[alt="Anchor image preview"]'));
    assert.equal(await page.locator('img[alt="Anchor image preview"]').count(), 0,
      'KFI references are not treated as Avatar anchor images');
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, video_prompt_type: 'KI', frames_positions: '0:00' },
    })));
    await page.waitForFunction(() => !document.querySelector('img[alt="Anchor image preview"]'));
    assert.equal(await page.locator('img[alt="Anchor image preview"]').count(), 0,
      'timed frame references are not treated as Avatar anchor images');
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, video_prompt_type: '', frames_positions: '' },
    })));
    await page.waitForFunction(() => !document.querySelector('img[alt="Anchor image preview"]'));
    assert.equal(await page.locator('img[alt="Anchor image preview"]').count(), 0,
      'saved reference paths without an active prompt type are not treated as Avatar anchors');
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, video_prompt_type: 'KI' },
    })));
    await page.waitForFunction(() => document.querySelector('img[alt="Anchor image preview"]'));
    assert.equal(await page.locator('div.border-sky-300').count(), 1,
      'the preview marks Voice 1 region');
    assert.equal(await page.locator('div.border-amber-300').count(), 1,
      'the preview marks Voice 2 region');
    const restoredInvalidRegions = '96:0:100:100 50:0:100:100';
    await page.evaluate(value => window.store.setState(state => ({
      params: { ...state.params, speakers_locations: value },
    })), restoredInvalidRegions);
    assert.equal(await page.getByRole('alert').count(), 1,
      'restored regions removed by the image-interior crop show a visible validation error');
    assert.equal(await page.evaluate(() => window.store.getState().params.speakers_locations), restoredInvalidRegions,
      'restored invalid regions remain intact until the user edits them');
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, speakers_locations: '0:0:50:100 50:0:100:100' },
    })));
    await page.waitForFunction(() => !document.querySelector('[role="alert"]'));

    await page.getByLabel('Voice 1 Left (%)').fill('10');
    await page.getByLabel('Voice 1 Top (%)').fill('10');
    await page.getByLabel('Voice 1 Right (%)').fill('60');
    await page.getByLabel('Voice 1 Bottom (%)').fill('80');
    await page.getByLabel('Voice 2 Left (%)').fill('40');
    await page.getByLabel('Voice 2 Top (%)').fill('15');
    await page.getByLabel('Voice 2 Right (%)').fill('95');
    await page.getByLabel('Voice 2 Bottom (%)').fill('90');
    const orderedRegions = '10:10:60:80 40:15:95:90';
    assert.equal(await page.evaluate(() => window.store.getState().params.speakers_locations),
      orderedRegions, 'Voice 1 and Voice 2 regions are written in order');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'valid speaker regions do not bypass missing audio inputs');

    await page.getByLabel('Voice 1 Left (%)').fill('101');
    assert.equal(await page.getByRole('alert').count(), 1, 'out-of-range percentages show a visible validation error');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'invalid speaker bounds block Generate');
    assert.equal(await generationActions.getByRole('button', { name: 'Add to queue unavailable' }).isDisabled(), true,
      'invalid speaker bounds block Add to Queue');
    assert.match(await page.evaluate(() => window.store.getState().params.speakers_locations), /^101:/,
      'invalid manual values remain visible to submit-time validation');
    await page.getByLabel('Voice 1 Left (%)').fill('10');
    assert.equal(await page.getByRole('alert').count(), 0, 'the validation error clears after fixing the bound');
    await page.getByRole('button', { name: 'Swap' }).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.speakers_locations),
      '40:15:95:90 10:10:60:80', 'swapping regions swaps the voice assignments');

    const helperResults = await page.evaluate(() => ({
      horizontal: window.avatarHelpers.parseAvatarSpeakerRegions('0:50 50:100'),
      clippedAway: window.avatarHelpers.parseAvatarSpeakerRegions('96:0:100:100 0:0:50:100'),
      reversed: window.avatarHelpers.parseAvatarSpeakerRegions('70:0:20:100 50:0:100:100'),
      unrelated: window.avatarHelpers.isLongCatAvatarModel({ model_type: 'longcat_video' }),
      multi: window.avatarHelpers.isMultiSpeakerAvatarModel({ model_type: 'longcat_avatar_multi' }),
    }));
    assert.deepEqual(helperResults.horizontal, [
      { left: 0, top: 0, right: 50, bottom: 100 },
      { left: 50, top: 0, right: 100, bottom: 100 },
    ], 'legacy horizontal pairs expand to full-height ordered boxes');
    assert.equal(helperResults.clippedAway, null, 'a box erased by LongCat interior clipping is rejected');
    assert.equal(helperResults.reversed, null, 'reversed box edges are rejected');
    assert.equal(helperResults.unrelated, false, 'only the two LongCat Avatar IDs are accepted');
    assert.equal(helperResults.multi, true, 'the multi-speaker ID is recognized');

    await page.getByLabel('Choose Voice audio #1').setInputFiles({
      name: 'multi-voice-1.wav', mimeType: 'audio/wav', buffer: Buffer.from('voice one'),
    });
    await page.waitForFunction(() => window.store.getState().params.audio_guide?.endsWith('multi-voice-1.wav'));
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'Multi remains blocked until Voice 2 is uploaded');
    await page.getByLabel('Choose Voice audio #2').setInputFiles({
      name: 'multi-voice-2.wav', mimeType: 'audio/wav', buffer: Buffer.from('voice two'),
    });
    await page.waitForFunction(() => window.store.getState().params.audio_guide2?.endsWith('multi-voice-2.wav'));
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), false,
      'anchor, both voices, and valid ordered regions enable Generate for Multi');
    assert.equal(await generationActions.getByRole('button', { name: 'Add current Studio settings to the queue' }).isDisabled(), false,
      'valid Multi inputs enable Add to Queue');
    assert.ok(uploadedNames.includes('multi-voice-1.wav') && uploadedNames.includes('multi-voice-2.wav'),
      'Multi uploads each voice to the intercepted audio endpoint');
    await page.getByRole('button', { name: 'Remove Voice audio #2' }).click();
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide2), undefined,
      'Voice audio #2 can be cleared independently');
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'clearing Voice 2 blocks Multi generation');

    await page.getByLabel('Choose Voice audio #1').setInputFiles({
      name: 'slow-clear.wav', mimeType: 'audio/wav', buffer: Buffer.from('slow voice'),
    });
    await slowClearStarted.promise;
    assert.equal(await generationActions.locator('button').nth(0).isDisabled(), true,
      'replacing Voice 1 clears the previous path and blocks Generate while upload is pending');
    assert.equal(await generationActions.getByRole('button', { name: 'Add to queue unavailable' }).isDisabled(), true,
      'Add to Queue stays blocked while a replacement Voice 1 upload is pending');
    await page.getByRole('button', { name: 'Cancel Voice audio #1 upload' }).click();
    slowClearRelease.resolve();
    await page.waitForTimeout(50);
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide), undefined,
      'a cleared input rejects the response from an older upload');

    await page.getByLabel('Choose Voice audio #2').setInputFiles({
      name: 'slow-workflow.wav', mimeType: 'audio/wav', buffer: Buffer.from('slow voice'),
    });
    await slowWorkflowStarted.promise;
    await page.evaluate(() => window.store.setState({ studioVideoWorkflow: 'frames' }));
    slowWorkflowRelease.resolve();
    await page.waitForTimeout(50);
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide2), undefined,
      'switching workflows rejects an older upload response');

    await page.evaluate(() => window.store.setState({ studioVideoWorkflow: 'avatar' }));
    await page.getByLabel('Choose Voice audio #2').setInputFiles({
      name: 'slow-model.wav', mimeType: 'audio/wav', buffer: Buffer.from('slow voice'),
    });
    await slowModelStarted.promise;
    await page.evaluate(() => window.store.setState(state => ({
      params: { ...state.params, model_type: 'longcat_avatar' },
    })));
    slowModelRelease.resolve();
    await page.waitForTimeout(50);
    assert.equal(await page.evaluate(() => window.store.getState().params.audio_guide2), undefined,
      'switching from Multi to Single rejects the second voice response');
    assert.equal(await page.getByLabel('Choose Voice audio #2').count(), 0,
      'Single hides the second input after a model switch');
    assert.deepEqual(pageErrors, [], 'Avatar input renders without browser errors');
    console.log('LongCat Avatar inputs, restored paths, Gallery reuse, speaker bounds, generation readiness, and stale upload guards pass');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
