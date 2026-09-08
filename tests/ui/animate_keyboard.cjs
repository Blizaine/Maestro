const assert = require('node:assert/strict');
const path = require('node:path');

// Uses the real Animate -> Image transition; all media/API requests are mocked
// by sidebar_redesign.cjs. No model is loaded or generation submitted.
async function assertAnimateKeyboard(page, sidebar, output) {
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt', exact: true});
  const visibleInput = async (input, name, minimumHeight = 100) => {
    const box = await input.boundingBox(), footer = await page.getByTestId('studio-settings-strip').boundingBox();
    const visible = await page.evaluate(() => ({top: window.visualViewport.offsetTop, bottom: window.visualViewport.offsetTop + window.visualViewport.height}));
    assert.ok(box.height >= minimumHeight, `${name} has room to write (${box.height}px)`);
    assert.ok(box.y >= visible.top && box.y + box.height <= Math.min(footer.y, visible.bottom) + 1, `${name} is visible above the footer and keyboard`);
    assert.equal(await input.evaluate(node => {
      const box = node.getBoundingClientRect();
      return document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2) === node;
    }), true, `${name} can be tapped, rather than clipped behind another panel`);
  };
  const keyboard = async (height, top, resizeWindow) => {
    await page.evaluate(({height, top, resizeWindow}) => {
      window.testOriginalInnerHeight ??= Object.getOwnPropertyDescriptor(window, 'innerHeight');
      if (resizeWindow) Object.defineProperty(window, 'innerHeight', {configurable: true, value: height});
      Object.defineProperty(window.visualViewport, 'height', {configurable: true, value: height});
      Object.defineProperty(window.visualViewport, 'offsetTop', {configurable: true, value: top});
      window.dispatchEvent(new Event('resize'));
      window.visualViewport.dispatchEvent(new Event('resize'));
      window.visualViewport.dispatchEvent(new Event('scroll'));
    }, {height, top, resizeWindow});
    await settle();
  };
  const dismissKeyboard = async () => {
    await page.evaluate(() => {
      if (window.testOriginalInnerHeight) Object.defineProperty(window, 'innerHeight', window.testOriginalInnerHeight);
      delete window.testOriginalInnerHeight;
      delete window.visualViewport.height; delete window.visualViewport.offsetTop;
      window.dispatchEvent(new Event('resize'));
      window.visualViewport.dispatchEvent(new Event('resize'));
      window.visualViewport.dispatchEvent(new Event('scroll'));
    });
    await settle();
  };
  for (const width of [390, 320]) {
    await page.setViewportSize({width, height: 844});
    await page.evaluate(() => {
      window.resetFixture('viggle_animate', 'video', 'animate');
      const s = window.store.getState();
      window.store.setState({selectedModelPerMode: {...s.selectedModelPerMode, image: 'flux2_klein_9b'},
        params: {...s.params, video_guide: '/uploads/control.mp4', _viggle_source_seconds: 10, viggle_character: undefined}});
    });
    await sidebar.getByRole('button', {name: 'Edit current video frame in Maestro'}).click();
    await sidebar.getByText('Editing Viggle reference frame', {exact: true}).waitFor();
    assert.equal(await page.evaluate(() => window.store.getState().params.model_type), 'flux2_klein_9b');
    assert.equal(await sidebar.getByRole('button', {name: 'Apply & return'}).isDisabled(), true);
    await prompt.fill(Array.from({length: 15}, () => 'Keep the source pose, props, camera framing and lighting.').join('\n'));
    await prompt.evaluate(node => {window.testAnimatePrompt = node; node.setSelectionRange(15, 20);});
    // Some mobile browsers shrink innerHeight along with visualViewport.height.
    // Comparing those two heights alone misses the keyboard in this case.
    await keyboard(350, 100, true);
    await visibleInput(prompt, `Frame-edit prompt at ${width}px`);
    assert.equal(await prompt.evaluate(node => node === window.testAnimatePrompt && node.selectionStart === 15 && node.selectionEnd === 20), true);
    await prompt.press('End'); await prompt.pressSequentially(' Keep the coat red.');
    assert.match(await page.evaluate(() => window.store.getState().params.prompt), /Keep the coat red\./);
    assert.equal(await prompt.evaluate(node => node.scrollHeight > node.clientHeight), true, 'Long frame-edit prompts scroll internally');
    await keyboard(330, 140, true);
    await visibleInput(prompt, 'Frame-edit prompt after another viewport shift');
    await page.screenshot({path: path.join(output, `animate-image-keyboard-${width}.png`)});
    await dismissKeyboard();
    assert.equal(await sidebar.getByText('Generate an image first, then click Apply.', {exact: true}).isVisible(), true);
    await sidebar.getByRole('button', {name: 'Return unchanged', exact: true}).click();
    assert.equal(await page.evaluate(() => window.store.getState().studioVideoWorkflow), 'animate');
    assert.equal(await page.evaluate(() => window.store.getState().params.video_guide), '/uploads/control.mp4');
    // Viggle also has text fields in its own scrolling controls.
    await sidebar.getByRole('button', {name: 'Use a character', exact: true}).click();
    const appearance = sidebar.getByRole('textbox', {name: 'Appearance (optional)', exact: true});
    await appearance.fill('A dark leather coat.');
    await keyboard(350, 100, false);
    await visibleInput(appearance, 'Animate appearance field', 80);
    await appearance.press('End'); await appearance.pressSequentially(' Red scarf.');
    assert.match(await page.evaluate(() => window.store.getState().params.viggle_character.appearance_prompt), /Red scarf\./);
    await dismissKeyboard();
  }
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  console.log('Animate keyboard: real frame-editor transition, prompt visibility/typing, return action and appearance controls passed');
}

module.exports = {assertAnimateKeyboard};
