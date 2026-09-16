const assert = require('node:assert/strict');
const path = require('node:path');

// Exercise real range dragging, including the single/multi-window transition.
// The caller supplies an isolated Sidebar with intercepted API requests.
async function assertDurationPopup(page, sidebar, output) {
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const dialog = page.getByRole('dialog', {name: 'Duration & windows'});
  const duration = () => dialog.getByRole('slider', {name: 'Duration model duration'});
  const close = () => dialog.press('Escape');
  const sameBox = (actual, expected, message) => {
    for (const key of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(actual[key] - expected[key]) < 1, `${message}: ${key} ${expected[key]} -> ${actual[key]}`);
  };
  const fitsSidebar = async () => {
    const panel = await dialog.boundingBox(), side = await sidebar.boundingBox();
    assert.ok(panel.x >= side.x && panel.x + panel.width <= side.x + side.width, 'Duration stays within the sidebar width');
    assert.ok(panel.y >= side.y && panel.y + panel.height <= side.y + side.height, 'Duration fits the visible sidebar height');
    const trigger = await sidebar.getByRole('button', {name: /^Duration:/}).boundingBox();
    assert.ok(panel.y + panel.height <= trigger.y - 5, 'Duration opens directly above its indicator');
    assert.ok(await dialog.evaluate(node => node.scrollWidth <= node.clientWidth), 'Duration has no horizontal overflow');
  };
  const drag = async (slider, label) => {
    await slider.focus(); await slider.press('Home'); await settle();
    await slider.scrollIntoViewIfNeeded();
    const before = await slider.boundingBox(), panelBefore = await dialog.boundingBox();
    const values = [];
    const xAt = fraction => before.x + 8 + fraction * (before.width - 16);
    await page.mouse.move(xAt(0), before.y + before.height / 2);
    await page.mouse.down();
    try {
      for (const fraction of [0.01, 0.03, 0.1, 0.5, 1, 0.1, 0.02, 0]) {
        await page.mouse.move(xAt(fraction), before.y + before.height / 2);
        await settle();
        sameBox(await dialog.boundingBox(), panelBefore, `${label} keeps the popup still`);
        sameBox(await slider.boundingBox(), before, `${label} stays under the pointer`);
        values.push(Number(await slider.inputValue()));
      }
    } finally { await page.mouse.up(); }
    assert.ok(Math.max(...values) > Math.min(...values), `${label} actually updates the selected value`);
    sameBox(await slider.boundingBox(), before, `${label} stays in place on release`);
  };

  for (const viewport of [{width: 1360, height: 900}, {width: 767, height: 844}, {width: 440, height: 740}, {width: 390, height: 650}, {width: 320, height: 600}]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => {
      window.resetFixture();
      window.store.getState().setParam('resolution', '1280x720');
    });
    await settle();
    for (const [label, prefix] of [['Resolution', /^Resolution:/], ['Aspect ratio', /^Aspect ratio:/]]) {
      const trigger = sidebar.getByRole('button', {name: prefix});
      await trigger.click(); await settle();
      const menu = page.getByRole('menu', {name: label, exact: true});
      const box = await menu.boundingBox(), anchor = await trigger.boundingBox();
      assert.ok(box.width < 150, `${label} sizes to its short option labels`);
      assert.ok(Math.abs(box.x - anchor.x) < 1, `${label} starts directly above its button`);
      assert.ok(Math.abs(box.y + box.height - anchor.y + 6) < 1, `${label} has a small gap above its button`);
      assert.equal(await menu.getByRole('heading').count(), 0, 'Compact options have no extra heading');
      await page.screenshot({path: path.join(output, `clip-${label.split(' ')[0].toLowerCase()}-${viewport.width}.png`)});
      await menu.press('Escape');
    }
    await page.evaluate(() => {
      const s = window.store.getState();
      s.setSlidingWindowLocked(true); s.setSlidingWindowSeconds(277 / 24); s.setDurationSeconds(8);
    });
    await sidebar.getByRole('button', {name: /^Duration:/}).click();
    await settle();
    await fitsSidebar();
    const auto = dialog.getByRole('switch', {name: 'Automatic duration'});
    assert.equal(await dialog.getByRole('button', {name: 'Auto', exact: true}).count(), 0, 'Auto is a toggle instead of a planning tab');
    for (const name of ['10m', '15m', '30m', '60m', 'Custom']) assert.equal(await dialog.getByRole('button', {name, exact: true}).count(), 1, `Retained preset: ${name}`);
    for (const name of ['1 window', '30s', '1m', '2m', '3m', '4m', '5m']) assert.equal(await dialog.getByRole('button', {name, exact: true}).count(), 0, `Short duration uses the slider: ${name}`);
    const timeBeforeAuto = await duration().boundingBox();
    await auto.click(); await settle();
    assert.equal(await auto.getAttribute('aria-checked'), 'true');
    const timecode = dialog.getByRole('textbox', {name: 'Duration timecode'});
    const longPreset = dialog.getByRole('button', {name: '10m', exact: true});
    assert.equal(await duration().isEnabled(), true, 'Auto still allows direct slider manipulation');
    assert.equal(await timecode.isEnabled(), true, 'Custom time entry can take over from Auto');
    assert.equal(await longPreset.isEnabled(), true, 'Auto leaves presets clickable');
    assert.ok(await duration().evaluate(node => Number(getComputedStyle(node.parentElement).opacity) < 1), 'Auto dims the slider');
    assert.ok(await longPreset.evaluate(node => Number(getComputedStyle(node.parentElement).opacity) < 1), 'Auto dims the presets');
    await page.screenshot({path: path.join(output, `clip-duration-auto-${viewport.width}.png`)});
    // Grab the auto thumb before moving it: the first pointer-down activates Time.
    const thumbBox = await duration().boundingBox();
    const thumbValue = Number(await duration().inputValue());
    const thumbMax = Number(await duration().getAttribute('max'));
    await page.mouse.move(thumbBox.x + 8 + thumbValue / thumbMax * (thumbBox.width - 16), thumbBox.y + thumbBox.height / 2);
    await page.mouse.down();
    try {
      await settle();
      assert.equal(await auto.getAttribute('aria-checked'), 'false', 'Grabbing the thumb turns Auto off immediately');
      assert.equal(await duration().evaluate(node => getComputedStyle(node.parentElement).opacity), '1', 'Grabbed slider becomes active');
      sameBox(await duration().boundingBox(), timeBeforeAuto, 'Auto takeover keeps the slider under the pointer');
      await page.mouse.move(thumbBox.x + thumbBox.width * 0.4, thumbBox.y + thumbBox.height / 2);
      await settle();
      assert.ok(Number(await duration().inputValue()) > thumbValue, 'The same drag changes duration');
    } finally { await page.mouse.up(); }
    await auto.click(); await settle();
    const autoIndex = Number(await duration().inputValue());
    await duration().press('ArrowRight'); await settle();
    assert.equal(await auto.getAttribute('aria-checked'), 'false', 'Keyboard adjustment turns Auto off');
    assert.equal(Number(await duration().inputValue()), autoIndex + 1, 'Keyboard takeover retains the requested native step');
    await auto.click(); await settle();
    await longPreset.click(); await settle();
    assert.equal(await auto.getAttribute('aria-checked'), 'false', 'Preset selection turns Auto off');
    assert.ok(await page.evaluate(() => window.store.getState().durationSeconds > 590), 'Manual preset survives automatic-duration reconciliation');
    assert.equal(await longPreset.evaluate(node => getComputedStyle(node.parentElement).opacity), '1', 'Preset selection restores active styling');
    await auto.click(); await settle();
    await dialog.getByRole('button', {name: 'Custom', exact: true}).click(); await settle();
    assert.equal(await auto.getAttribute('aria-checked'), 'false', 'Custom also takes over from Auto');
    assert.ok(await timecode.evaluate(node => document.activeElement === node), 'Custom focuses the timecode');
    await timecode.fill('00:00:40'); await timecode.press('Enter'); await settle();
    assert.equal(await page.evaluate(() => window.store.getState().durationSeconds), 40, 'Custom time is applied');
    await drag(duration(), `Time slider at ${viewport.width}px`);
    await duration().press('End'); await settle();
    const overlap = dialog.locator('input[aria-label="Window overlap"]');
    assert.equal(await overlap.isVisible(), false, 'Overlap starts collapsed in Time mode');
    await dialog.locator('summary').filter({hasText: 'Window overlap'}).click();
    assert.equal(await overlap.isVisible(), true);
    const oldOverlap = await overlap.inputValue();
    await overlap.press('ArrowRight'); await settle();
    assert.notEqual(await overlap.inputValue(), oldOverlap, 'Expanded overlap still updates its value');
    await dialog.locator('summary').filter({hasText: 'Window overlap'}).click();
    const continuity = dialog.getByRole('checkbox', {name: /Carry motion and sound between windows/});
    await continuity.scrollIntoViewIfNeeded();
    await continuity.uncheck();
    assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_sequence_continuity), false, 'Scrolled continuation controls still work');
    await continuity.check();
    await duration().scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(output, `duration-${viewport.width}-multi.png`)});
    // Changing the window limit can itself switch single/multi-window mode.
    await page.evaluate(() => window.store.getState().setDurationSeconds(8));
    await settle();
    await drag(dialog.getByRole('slider', {name: 'Window length'}), `Window slider at ${viewport.width}px`);
    await page.evaluate(() => window.store.getState().setSlidingWindowLocked(false));
    await settle();
    await drag(duration(), `Time slider with automatic windows at ${viewport.width}px`);
    await fitsSidebar();
    await close();
  }
  for (const [id, workflow] of [['minimax_h3', 'frames'], ['minimax_h3', 'extend'], ['ltx2_22B_distilled_1_1', 'frames']]) {
    await page.setViewportSize({width: 390, height: 740});
    await page.evaluate(args => window.resetFixture(...args), [id, 'video', workflow]);
    await sidebar.getByRole('button', {name: /^Duration:/}).click();
    await settle();
    await fitsSidebar();
    await drag(duration(), `${id} ${workflow} Time slider`);
    const auto = dialog.getByRole('switch', {name: 'Automatic duration'});
    const autoBox = await auto.boundingBox();
    const tabBox = await dialog.getByRole('button', {name: 'Window', exact: true}).boundingBox();
    await dialog.getByRole('button', {name: 'Window', exact: true}).click();
    await settle();
    assert.equal(await auto.isVisible(), true, 'Auto is available in Window mode');
    const windowAutoBox = await auto.boundingBox();
    for (const key of ['y', 'height']) assert.ok(Math.abs(windowAutoBox[key] - autoBox[key]) < 1, `Auto header retains its ${key} between Time and Window`);
    sameBox(await dialog.getByRole('button', {name: 'Window', exact: true}).boundingBox(), tabBox, 'Planning tabs do not shift when switching to Window');
    await drag(dialog.getByRole('slider', {name: 'Exact window count'}), `${id} ${workflow} window count`);
    assert.equal(await dialog.locator('input[aria-label="Window overlap"]').isVisible(), false, 'Overlap also starts collapsed in Window mode');
    await auto.click(); await settle();
    assert.equal(await auto.getAttribute('aria-checked'), 'true', 'Window view can return directly to Auto');
    assert.equal(await duration().isVisible(), true, 'Auto shows its dimmed time recommendation');
    sameBox(await dialog.getByRole('button', {name: 'Window', exact: true}).boundingBox(), tabBox, 'Planning tabs stay fixed when Auto is enabled from Window');
    await close();
  }
  await page.evaluate(() => window.resetFixture());
  await settle();
  // A keyboard-open visual viewport still bounds the fixed-size popup.
  await sidebar.getByRole('button', {name: /^Duration:/}).click();
  await page.evaluate(() => {
    Object.defineProperty(window.visualViewport, 'height', {configurable: true, value: 370});
    Object.defineProperty(window.visualViewport, 'offsetTop', {configurable: true, value: 120});
    window.visualViewport.dispatchEvent(new Event('resize')); window.visualViewport.dispatchEvent(new Event('scroll'));
  });
  await settle();
  await fitsSidebar();
  await drag(duration(), 'Time slider above the keyboard');
  await close();
  await page.evaluate(() => {
    delete window.visualViewport.height; delete window.visualViewport.offsetTop;
    window.visualViewport.dispatchEvent(new Event('resize')); window.visualViewport.dispatchEvent(new Event('scroll'));
  });
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  console.log('Clip settings: anchored menus, stable Auto header in both tabs, direct pointer/keyboard/preset/custom takeover from Auto, collapsed overlap, stable dragging, window controls, continuation and keyboard viewport passed');
}

module.exports = {assertDurationPopup};
