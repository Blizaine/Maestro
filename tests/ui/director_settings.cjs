const assert = require('node:assert/strict');
const path = require('node:path');

// Exercise real Director controls and persistence on sidebar_redesign's isolated
// origin. Model LoRA data is synthetic; no live settings or generations are used.
async function assertDirectorSettings(page, sidebar, output) {
  const h3 = 'minimax_h3_ref2va_fused_turbo', image = 'flux2_klein_9b', ltx = 'ltx2_22B_distilled_1_1';
  const videoFile = 'FilmStyle.safetensors', imageFile = 'PortraitStyle.safetensors';
  const phaseCounts = {[h3]: 0, [image]: 1, [ltx]: 2};
  const routeLoras = async route => {
    const endpoint = new URL(route.request().url()).pathname;
    const model = decodeURIComponent(endpoint.split('/')[4]);
    const filename = model === image ? imageFile : videoFile;
    const body = endpoint.endsWith('/details') ? {loras: [{filename, recommended_weights: {default: 0.8, min: 0.6, max: 1.0, source: 'default'}}]}
      : endpoint.includes('/guide/') ? {guide: ''}
        : {loras: [filename], guidance_max_phases: phaseCounts[model] ?? 1};
    return route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
  };
  await page.route('**/api/v1/loras/**', routeLoras);
  const queued = [];
  const routeQueue = async route => {
    if (route.request().method() === 'POST') queued.push(route.request().postDataJSON().params);
    return route.fulfill({contentType: 'application/json', body: JSON.stringify({version: 1, paused: true, entries: [], running: false})});
  };
  await page.route('**/api/v1/director/queue', routeQueue);
  const settle = () => page.waitForTimeout(200);
  const director = sidebar.getByTestId('director-chat');
  const readLora = mode => page.evaluate(mode => window.store.getState().savedLoraPerMode[mode], mode);
  const setup = async (model, videoState) => {
    await page.evaluate(model => window.resetFixture(model), model);
    await settle();
    await page.evaluate(({model, image, videoState}) => {
      window.store.setState({sidebarMode: 'director', directorSkill: 'short_film', shortFilmPath: 'story',
        directorStep: 'style', directorLoading: false, directorError: null, directorAnalysis: null,
        directorPlannedClips: [], directorClipPlans: [], directorClipImages: [], directorLlmLog: [],
        directorSpeakers: [], directorSpeakerMappings: [], pipelineStatus: null,
        directorSceneDescription: '', directorH3References: [], directorShotImageGuidance: 'generate',
        selectedModelPerMode: {video: model, image}, savedLoraPerMode: videoState ? {video: videoState} : {},
        directorReferenceImage: null, directorReferenceImagePath: '', directorCharacterRefs: [], directorLocationRefs: [],
        directorAudioFile: null, directorAudioPath: '', directorVoiceRef: null, directorVoiceRefPath: '', directorQueueEditingEntryId: null,
        directorCharacterRefPaths: [], directorLocationRefPaths: [], shortFilmCharacters: [],
        directorResolution: '480p', directorAspectRatio: '16:9', directorSeamless: false});
    }, {model, image, videoState});
    await settle();
  };
  try {
    for (const width of [1360, 390, 320]) {
      await page.setViewportSize({width, height: 900});
      await setup(h3);
      const auto = director.getByRole('switch', {name: 'Automatic duration'});
      const time = director.getByRole('button', {name: 'Time', exact: true});
      const windowTab = director.getByRole('button', {name: 'Window', exact: true});
      const slider = director.getByRole('slider', {name: 'Target duration model duration'});
      const timecode = director.getByRole('textbox', {name: 'Target duration timecode'});
      assert.equal(await auto.getAttribute('aria-checked'), 'true');
      assert.equal(await director.getByRole('button', {name: /^Auto\b/}).count(), 0, 'The old Auto tab is absent');
      assert.equal(await slider.isEnabled(), true, 'Auto slider remains interactive');
      assert.ok(await slider.evaluate(node => Number(getComputedStyle(node.parentElement).opacity) < 1));
      for (const label of ['10m', '15m', '30m', '60m', 'Custom']) {
        assert.equal(await director.getByRole('button', {name: label, exact: true}).count(), 1);
      }
      for (const label of ['30s', '1m', '2m', '3m', '4m', '5m']) {
        assert.equal(await director.getByRole('button', {name: label, exact: true}).count(), 0);
      }
      await slider.scrollIntoViewIfNeeded();
      const thumb = await slider.boundingBox();
      const value = Number(await slider.inputValue()), max = Number(await slider.getAttribute('max'));
      await page.mouse.move(thumb.x + 8 + value / max * (thumb.width - 16), thumb.y + thumb.height / 2);
      await page.mouse.down();
      try {
        await settle();
        assert.equal(await auto.getAttribute('aria-checked'), 'false', 'Grabbing the dimmed thumb disables Auto immediately');
        assert.equal(await slider.evaluate(node => getComputedStyle(node.parentElement).opacity), '1');
        const after = await slider.boundingBox();
        assert.ok(Math.abs(after.y - thumb.y) < 1, 'Auto takeover does not move the thumb');
      } finally { await page.mouse.up(); }
      await slider.focus(); await slider.press('Home'); await settle();
      const minimum = await page.evaluate(() => window.store.getState().shortFilmTargetDuration);
      await slider.press('ArrowRight'); await settle();
      const next = await page.evaluate(() => window.store.getState().shortFilmTargetDuration);
      const step = await page.evaluate(model => window.options[model].frames_steps / window.options[model].fps, h3);
      assert.ok(minimum >= 10 && Math.abs(next - minimum - step) < 0.001, 'Director keeps its minimum and model-aligned duration steps');
      await slider.press('End'); await settle();
      assert.ok(await page.evaluate(() => window.store.getState().shortFilmTargetDuration <= 300));
      await windowTab.click(); await settle();
      assert.equal(await auto.isVisible(), true, 'Auto remains available from Window');
      const tabTop = (await time.boundingBox()).y;
      await director.getByRole('spinbutton', {name: 'Window count', exact: true}).fill('3');
      await settle();
      assert.equal(await director.getByRole('spinbutton', {name: 'Window count', exact: true}).inputValue(), '3');
      await auto.click(); await settle();
      assert.equal(await auto.getAttribute('aria-checked'), 'true');
      assert.ok(Math.abs((await time.boundingBox()).y - tabTop) < 1, 'Auto/current duration header keeps the tabs aligned');
      await director.getByRole('button', {name: '10m', exact: true}).click(); await settle();
      assert.equal(await auto.getAttribute('aria-checked'), 'false', 'A long preset takes over from Auto');
      assert.ok(await page.evaluate(() => window.store.getState().shortFilmTargetDuration > 590));
      await auto.click(); await timecode.fill('00:01:25'); await timecode.press('Enter'); await settle();
      assert.equal(await page.evaluate(() => window.store.getState().shortFilmTargetDuration), 85);

      const videoToggle = director.getByRole('button', {name: 'Video LoRAs', exact: true});
      await videoToggle.click(); await settle();
      await director.getByRole('button', {name: 'FilmStyle', exact: true}).click(); await settle();
      const strength = director.getByRole('slider', {name: 'FilmStyle LoRA strength', exact: true});
      assert.equal(await strength.count(), 1, 'Zero guidance phases still provides an H3 weight slider');
      await strength.focus(); await strength.press('Home'); await settle();
      assert.deepEqual((await readLora('video')).loraWeights[videoFile], [0]);
      assert.equal((await readLora('video')).loras_multipliers, '0.00', 'Explicit zero is serialized');
      const number = director.getByRole('spinbutton', {name: 'FilmStyle LoRA strength value', exact: true});
      await number.fill('1'); await settle();
      await videoToggle.click(); await videoToggle.click(); await settle();
      assert.equal(await strength.inputValue(), '1', 'Saved 1.0 is not overwritten by metadata recommendations');
      await number.fill('0.45'); await settle();
      assert.equal(await strength.inputValue(), '0.45');
      assert.equal((await readLora('video')).loras_multipliers, '0.45');
      await videoToggle.click(); await videoToggle.click(); await settle();
      assert.equal(await strength.inputValue(), '0.45', 'Strength survives closing and reopening');
      const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem('maestro_mode_settings')).savedLoraPerMode.video);
      assert.equal(persisted.loras_multipliers, '0.45', 'Weight is saved in browser settings');

      await director.getByRole('button', {name: 'Image LoRAs', exact: true}).click(); await settle();
      await director.getByRole('button', {name: 'PortraitStyle', exact: true}).click(); await settle();
      const imageStrength = director.getByRole('slider', {name: 'PortraitStyle LoRA strength', exact: true});
      await imageStrength.focus(); await imageStrength.press('End'); await settle();
      assert.deepEqual((await readLora('image')).loraWeights[imageFile], [2]);
      assert.deepEqual((await readLora('video')).loraWeights[videoFile], [0.45], 'Image edits do not change Video weights');
      for (const range of [strength, imageStrength]) {
        await range.scrollIntoViewIfNeeded();
        const box = await range.boundingBox(), bounds = await sidebar.boundingBox();
        assert.ok(box.width > 30 && box.x >= bounds.x && box.x + box.width <= bounds.x + bounds.width, 'Weight slider fits the panel at ' + width);
      }
      assert.ok(await director.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Director cannot overflow horizontally');
      if (width === 390) {
        await page.screenshot({path: path.join(output, 'director-lora-390.png')});
        await page.evaluate(() => window.store.getState().startDirectorPipeline('queue'));
        assert.equal(queued.length, 1, await page.evaluate(() => window.store.getState().directorError));
        assert.equal(queued[0].target_duration, 85);
        assert.equal(queued[0].video_model, h3);
        assert.equal(queued[0].video_loras.loras_multipliers, '0.45', 'Director queue receives the reviewed Video weight');
        assert.equal(queued[0].image_loras.loras_multipliers, '2.00', 'Director queue receives the separate Image weight');
      }
    }
    await setup(h3, {activated_loras: [videoFile], loras_multipliers: '0.35', loraWeights: {[videoFile]: []}, availableLoras: [videoFile]});
    await director.getByRole('button', {name: 'Video LoRAs', exact: true}).click(); await settle();
    assert.equal(await director.getByRole('slider', {name: 'FilmStyle LoRA strength', exact: true}).inputValue(), '0.35', 'Legacy empty H3 arrays recover the saved multiplier');
    assert.deepEqual((await readLora('video')).loraWeights[videoFile], [0.35]);
    await setup(ltx, {activated_loras: [videoFile], loras_multipliers: '0.30;0.70', loraWeights: {}, availableLoras: [videoFile]});
    await director.getByRole('button', {name: 'Video LoRAs', exact: true}).click(); await settle();
    const first = director.getByRole('slider', {name: 'FilmStyle phase 1 LoRA strength', exact: true});
    const second = director.getByRole('slider', {name: 'FilmStyle phase 2 LoRA strength', exact: true});
    assert.equal(await first.inputValue(), '0.3'); assert.equal(await second.inputValue(), '0.7');
    await second.focus(); await second.press('Home'); await settle();
    assert.deepEqual((await readLora('video')).loraWeights[videoFile], [0.3, 0]);
    assert.equal((await readLora('video')).loras_multipliers, '0.30;0.00', 'Multi-phase model weights remain independent');
    console.log('Director settings: compact duration Auto/tabs/native steps/presets, H3 zero-phase weights, Image isolation, legacy recovery, multi-phase weights, persistence and held queue payload passed');
  } finally {
    await page.unroute('**/api/v1/loras/**', routeLoras);
    await page.unroute('**/api/v1/director/queue', routeQueue);
    await page.setViewportSize({width: 1360, height: 900});
    await page.evaluate(() => { window.resetFixture(); window.store.setState({savedLoraPerMode: {}}); });
    await settle();
  }
}

module.exports = {assertDirectorSettings};
