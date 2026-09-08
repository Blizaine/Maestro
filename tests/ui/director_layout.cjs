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
  for (const width of [1360, 390, 320]) {
    await page.setViewportSize({width, height: 844});
    await setup('minimax_h3_ref2va_fused_turbo');
    await fits('H3 story setup at ' + width);
    const input = composer.locator('textarea');
    await input.fill('A detective searches the city. '.repeat(80));
    await fits('Long Director composer at ' + width);
    assert.ok(await input.evaluate(node => node.scrollHeight > node.clientHeight), 'Long composer scrolls internally');
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
  }
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  console.log('Director layout: horizontal bounds, vertical scrolling, long text/logs, character/speaker fields and simulated keyboard passed');
}

module.exports = {assertDirectorLayout};
