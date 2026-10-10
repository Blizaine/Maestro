const assert = require('node:assert/strict');
const path = require('node:path');

// Real Director UI with the isolated API fixtures from sidebar_redesign.cjs.
// Does not plan a project, write user settings, or submit a generation.
async function assertDirectorLayout(page, sidebar, output) {
  const settle = () => page.waitForTimeout(200);
  const director = sidebar.getByTestId('director-chat');
  const messages = director.getByTestId('director-messages');
  const composer = director.getByTestId('director-composer');
  const setup = async model => {
    await page.evaluate(model => {
      window.resetFixture(model);
      window.store.setState({sidebarMode: 'director', directorSkill: 'short_film', shortFilmPath: 'story',
        directorStep: 'style', directorLoading: false, directorLoadingMessage: null, directorError: null,
        directorAnalysis: null, directorPlannedClips: [], directorClipPlans: [], directorClipImages: [],
        directorLlmLog: [], directorSpeakers: [], directorSpeakerMappings: [], pipelineStatus: null,
        directorSceneDescription: '', directorH3References: [], directorShotImageGuidance: 'prompt_only',
        selectedModelPerMode: {video: model, image: 'flux2_klein_9b'},
        directorReferenceImage: null, directorReferenceImagePath: '', directorCharacterRefs: [], directorLocationRefs: [],
        shortFilmCharacters: [{name: 'Blaine', description: 'A detective wearing a brown coat.'}],
        directorResolution: '480p', directorAspectRatio: '16:9', directorSeamless: false});
    }, model);
    await settle();
  };
  const fits = async label => {
    const measurements = await director.evaluate(root => {
      const nodes = [root, ...root.querySelectorAll('*')];
      return nodes.filter(node => node.clientWidth && node.scrollWidth > node.clientWidth + 1
        && !['TEXTAREA', 'INPUT', 'SELECT'].includes(node.tagName)).map(node => ({
          tag: node.tagName, classes: String(node.className).slice(0, 100),
          width: node.clientWidth, scrollWidth: node.scrollWidth,
          text: node.textContent.slice(0, 80),
        }));
    });
    assert.deepEqual(measurements, [], label + ' has no overflowing content');
    assert.equal(await messages.evaluate(node => {node.scrollLeft = 100; return node.scrollLeft}), 0, 'Director cannot pan sideways');
    assert.ok(await sidebar.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Sidebar fits');
  };
  const assertAnalysisScroll = async width => {
    const analysis = {duration: 236.5, sample_rate: 44100, bpm: 117.5, beats: [], downbeats: [],
      sections: [{start: 0, end: 30, label: 'verse', energy: 0.6}], onset_envelope: [], lyrics: null, vocals_path: null};
    await page.evaluate(analysis => {
      window.resetFixture('minimax_h3_ref2va_fused_turbo');
      window.store.setState({sidebarMode: 'director', directorSkill: 'music_video', directorMusicSource: 'upload',
        directorStep: 'analyze', directorLoading: true, directorLoadingMessage: 'Loading transcription model…',
        directorError: null, directorAudioFile: new File(['fixture'], 'fixture-track.wav', {type: 'audio/wav'}),
        directorAnalysis: null, directorPlannedClips: [], directorClipPlans: [], directorClipImages: [], directorLlmLog: [],
        directorSpeakers: [], directorSpeakerMappings: [], pipelineStatus: null,
        selectedModelPerMode: {video: 'minimax_h3_ref2va_fused_turbo', image: 'flux2_klein_9b'},
        directorH3References: [], directorReferenceImage: null, directorReferenceImagePath: '',
        directorCharacterRefs: [], directorCharacterRefPaths: [], directorLocationRefs: [], directorLocationRefPaths: [],
        shortFilmCharacters: [], shortFilmPath: null, directorSceneDescription: ''});
    }, analysis);
    await settle();

    const scrollDuringAnalysis = await messages.evaluate(node => {
      const max = node.scrollHeight - node.clientHeight;
      node.scrollTop = Math.floor(max * 0.35);
      return {max, top: node.scrollTop};
    });
    assert.ok(scrollDuringAnalysis.max > 20, `Analysis panel is scrollable at width ${width}`);
    await page.evaluate(() => window.store.setState({directorLoadingMessage: 'Detecting beats…'}));
    await settle();
    assert.equal(await messages.evaluate(node => node.scrollTop), scrollDuringAnalysis.top,
      'Analysis progress preserves the user’s manual Director scroll at width ' + width);

    await messages.evaluate(node => {node.scrollTop = node.scrollHeight});
    const outerScrollBefore = await page.evaluate(() => {
      const messageList = document.querySelector('[data-testid="director-messages"]');
      const ancestors = [];
      for (let node = messageList?.parentElement; node; node = node.parentElement) {
        ancestors.push([node.tagName, String(node.className), node.scrollTop, node.scrollLeft]);
      }
      return {page: [window.scrollX, window.scrollY], ancestors};
    });
    const clips = [{start: 0, end: 10, section_label: 'verse', energy: 0.6,
      suggested_prompt_hint: '', beat_count: 4, duration_frames: 240}];
    await page.evaluate(({analysis, clips}) => window.store.setState({directorStep: 'structure', directorLoading: false,
      directorLoadingMessage: null, directorAnalysis: analysis, directorPlannedClips: clips}), {analysis, clips});
    await page.waitForFunction(() => {
      const messages = document.querySelector('[data-testid="director-messages"]');
      const referenceSetup = document.querySelector('[data-testid="director-reference-setup"]');
      return messages && referenceSetup
        && Math.abs(referenceSetup.getBoundingClientRect().top - messages.getBoundingClientRect().top - messages.clientTop - 8) < 3;
    }, null, {timeout: 1500});

    const referenceSetup = director.getByTestId('director-reference-setup');
    const addReference = referenceSetup.getByRole('button', {name: 'Add reference', exact: true});
    const characters = referenceSetup.getByRole('button', {name: 'Characters', exact: true});
    const assertInsideMessageBounds = async (control, label) => {
      const [controlBox, messageBox] = await Promise.all([control.boundingBox(), messages.boundingBox()]);
      assert.ok(controlBox && messageBox, label + ' has a visible layout box at width ' + width);
      assert.ok(controlBox.y >= messageBox.y - 1 && controlBox.y + controlBox.height <= messageBox.y + messageBox.height + 1,
        label + ' is inside the visible Director message area at width ' + width);
    };
    await assertInsideMessageBounds(addReference, 'Optional Add reference control');
    await assertInsideMessageBounds(characters, 'Optional Characters control');
    const outerScrollAfter = await page.evaluate(() => {
      const messageList = document.querySelector('[data-testid="director-messages"]');
      const ancestors = [];
      for (let node = messageList?.parentElement; node; node = node.parentElement) {
        ancestors.push([node.tagName, String(node.className), node.scrollTop, node.scrollLeft]);
      }
      return {page: [window.scrollX, window.scrollY], ancestors};
    });
    assert.deepEqual(outerScrollAfter, outerScrollBefore,
      'Analysis completion moves only the inner Director scroll at width ' + width);

    const manualTop = await messages.evaluate(node => {
      const max = node.scrollHeight - node.clientHeight;
      node.scrollTop = Math.max(1, Math.min(max - 1, Math.round(max * 0.45)));
      return node.scrollTop;
    });
    await page.evaluate(() => window.store.setState({directorLoading: true, directorLoadingMessage: 'Recalculating the clip layout…'}));
    await settle();
    assert.equal(await messages.evaluate(node => node.scrollTop), manualTop,
      'Structure progress preserves the user’s manual scroll at width ' + width);
    await page.evaluate(() => window.store.setState({directorLoadingMessage: 'Updating shot durations…'}));
    await settle();
    assert.equal(await messages.evaluate(node => node.scrollTop), manualTop,
      'Repeated setup progress does not steal scroll at width ' + width);
    await page.evaluate(() => window.store.setState({directorLoading: false, directorLoadingMessage: null, directorStep: 'style'}));
    await settle();
    await page.evaluate(() => window.store.setState({directorSceneDescription: 'A quiet performance under blue-hour street lights.'}));
    await settle();
    assert.equal(await messages.evaluate(node => node.scrollTop), manualTop,
      'Style setup edits preserve the user’s manual scroll at width ' + width);

    const planningScrollRange = await messages.evaluate(node => node.scrollHeight - node.clientHeight);
    assert.ok(planningScrollRange > 20, 'Planning panel has room to tail at width ' + width);
    await messages.evaluate(node => {node.scrollTop = 0});
    await page.evaluate(() => window.store.setState({directorStep: 'plan', directorLoading: true,
      directorLoadingMessage: 'Planning video prompts…', pipelineStatus: {status: 'running', phase: 'planning',
        progress: {message: 'Planning video prompts…'}}}));
    await page.waitForTimeout(450);
    const atPlanningTail = await messages.evaluate(node => node.scrollHeight - node.clientHeight - node.scrollTop <= 2);
    assert.ok(atPlanningTail, 'Planning resumes useful progress tailing at width ' + width);
    await messages.evaluate(node => {node.scrollTop = 0});
    await page.evaluate(() => window.store.setState({directorLoadingMessage: 'Writing the final prompt pass…'}));
    await page.waitForTimeout(450);
    assert.ok(await messages.evaluate(node => node.scrollHeight - node.clientHeight - node.scrollTop <= 2),
      'Later planning progress continues to tail at width ' + width);

    // Music projects can also land directly on Style after analysis. Keep the
    // reference and character controls visible on this path too.
    await page.evaluate(() => window.store.setState({directorStep: 'analyze', directorLoading: true,
      directorLoadingMessage: 'Transcribing audio…', directorAnalysis: null, directorPlannedClips: [],
      directorClipPlans: [], directorClipImages: [], directorLlmLog: [], directorError: null,
      directorAudioFile: new File(['fixture'], 'direct-style-track.wav', {type: 'audio/wav'}),
      directorH3References: [], directorReferenceImage: null, directorReferenceImagePath: '',
      directorCharacterRefs: [], directorCharacterRefPaths: [], directorLocationRefs: [], directorLocationRefPaths: [],
      pipelineStatus: null}));
    await settle();
    await messages.evaluate(node => {node.scrollTop = node.scrollHeight});
    await page.evaluate(({analysis, clips}) => window.store.setState({directorStep: 'style', directorLoading: false,
      directorLoadingMessage: null, directorAnalysis: analysis, directorPlannedClips: clips}), {analysis, clips});
    await page.waitForFunction(() => {
      const messages = document.querySelector('[data-testid="director-messages"]');
      const referenceSetup = document.querySelector('[data-testid="director-reference-setup"]');
      return messages && referenceSetup
        && Math.abs(referenceSetup.getBoundingClientRect().top - messages.getBoundingClientRect().top - messages.clientTop - 8) < 3;
    }, null, {timeout: 1500});
    await assertInsideMessageBounds(addReference, 'Direct-to-style Add reference control');
    await assertInsideMessageBounds(characters, 'Direct-to-style Characters control');
    console.log(`Director analysis scroll at ${width}px: reference anchor, stable setup scroll and planning tail passed`);
  };
  for (const width of [1360, 390, 320]) {
    await page.setViewportSize({width, height: 844});
    await setup('minimax_h3_ref2va_fused_turbo');
    await fits('H3 story setup at ' + width);
    const input = composer.locator('textarea');
    await input.fill('A detective searches the city. '.repeat(80));
    await fits('Long Director composer at ' + width);
    assert.ok(await input.evaluate(node => node.scrollHeight > node.clientHeight), 'Long composer scrolls internally');
    if (width >= 768) {
      const scrollBeforeEdit = await sidebar.evaluate(node => node.scrollTop);
      await input.evaluate(node => {
        node.scrollTop = node.scrollHeight;
        node.focus();
        node.setSelectionRange(node.value.length, node.value.length);
        node.dispatchEvent(new Event('input', {bubbles: true}));
        node.dispatchEvent(new KeyboardEvent('keyup', {bubbles: true, key: 'End'}));
        node.dispatchEvent(new Event('select', {bubbles: true}));
      });
      await settle();
      assert.equal(await sidebar.evaluate(node => node.scrollTop), scrollBeforeEdit,
        'Clicking or typing in visible long text does not reposition the sidebar');
    }
    const bounds = await sidebar.boundingBox();
    for (const name of ['Start Director project now', 'Add Director project to queue']) {
      const button = await composer.getByRole('button', {name, exact: true}).boundingBox();
      assert.ok(button.x >= bounds.x && button.x + button.width <= bounds.x + bounds.width, name + ' stays inside the sidebar');
    }
    await messages.evaluate(node => {node.scrollTop = 100});
    assert.ok(await messages.evaluate(node => node.scrollTop > 0), 'Director still scrolls vertically');
    const pagePosition = await page.evaluate(() => [window.scrollX, window.scrollY]);
    await page.evaluate(() => window.store.setState({directorError: 'Could not read /models/' + 'a_long_model_filename_'.repeat(35) + '.safetensors'}));
    await settle();
    await fits('Long error path at ' + width);
    assert.deepEqual(await page.evaluate(() => [window.scrollX, window.scrollY]), pagePosition, 'Progress scrolling does not move the surrounding page');
    if (width < 768) {
      await input.focus();
      await page.evaluate(() => {
        Object.defineProperty(window.visualViewport, 'height', {configurable: true, value: 350});
        Object.defineProperty(window.visualViewport, 'offsetTop', {configurable: true, value: 100});
        window.visualViewport.dispatchEvent(new Event('resize'));
        window.visualViewport.dispatchEvent(new Event('scroll'));
      });
      await settle();
      await fits('Director keyboard at ' + width);
      const field = await input.boundingBox();
      assert.ok(field.y >= 100 && field.y + field.height <= 450, 'Director composer stays above the keyboard');
      await input.press('End'); await input.pressSequentially(' Keep the ending quiet.');
      assert.match(await page.evaluate(() => window.store.getState().directorSceneDescription), /Keep the ending quiet\./);
      await page.screenshot({path: path.join(output, `director-keyboard-${width}.png`)});
      await page.evaluate(() => {
        delete window.visualViewport.height; delete window.visualViewport.offsetTop;
        window.visualViewport.dispatchEvent(new Event('resize'));
        window.visualViewport.dispatchEvent(new Event('scroll'));
      });
      await settle();
    }

    await setup('ltx2_22B_distilled_1_1');
    await page.evaluate(() => {
      const image = new File(['<svg xmlns="http://www.w3.org/2000/svg"/>'], 'reference.svg', {type: 'image/svg+xml'});
      window.store.setState({directorReferenceImage: image});
    });
    await settle();
    await fits('Character naming at ' + width);
    await sidebar.getByPlaceholder('Character 1 name', {exact: true}).fill('Blaine');
    assert.equal(await page.evaluate(() => window.store.getState().shortFilmCharacters[0].name), 'Blaine');
    await page.screenshot({path: path.join(output, `director-${width}.png`)});
    await page.evaluate(() => window.store.setState({directorSkill: 'music_video', shortFilmPath: null, directorMusicSource: 'upload',
      directorSpeakers: ['SPEAKER_00'], directorSpeakerMappings: [{speakerId: 'SPEAKER_00', name: 'Blaine', role: 'singing'}]}));
    await settle();
    await fits('Music video speaker controls at ' + width);
    await sidebar.getByPlaceholder('e.g. man in green hoodie').fill('Blaine in a coat');
    assert.equal(await page.evaluate(() => window.store.getState().directorSpeakerMappings[0].name), 'Blaine in a coat');
    await page.evaluate(() => window.store.setState({directorStep: 'review_video',
      directorLlmLog: [{stage: 'plan', text: '<think>' + 'long_identifier_'.repeat(80) + '</think>\n' + 'prompt_token_'.repeat(100)}]}));
    await sidebar.getByRole('button', {name: 'Video planning (done)', exact: true}).click();
    await fits('Expanded planning log at ' + width);
    await assertAnalysisScroll(width);
  }
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  console.log('Director layout: horizontal bounds, vertical scrolling, long text/logs, character/speaker fields and simulated keyboard passed');
}

module.exports = {assertDirectorLayout};
