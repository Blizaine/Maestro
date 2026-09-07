const assert = require('node:assert/strict');
const path = require('node:path');

// Exercise real range dragging, including the single/multi-window transition.
// The caller supplies an isolated Sidebar with intercepted API requests.
async function assertDurationPopup(page, sidebar, output) {
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const dialog = page.getByRole('dialog', {name: 'Duration & windows'});
  const duration = () => dialog.getByRole('slider', {name: 'Duration model duration'});
  const close = () => dialog.getByRole('button', {name: 'Close Duration & windows'}).click();
  const sameBox = (actual, expected, message) => {
    for (const key of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(actual[key] - expected[key]) < 1, `${message}: ${key} ${expected[key]} -> ${actual[key]}`);
  };
  const fitsSidebar = async () => {
    const panel = await dialog.boundingBox(), side = await sidebar.boundingBox();
    assert.ok(panel.x >= side.x && panel.x + panel.width <= side.x + side.width, 'Duration stays within the sidebar width');
    assert.ok(panel.y >= side.y && panel.y + panel.height <= side.y + side.height, 'Duration fits the visible sidebar height');
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
    await page.evaluate(() => {
      const s = window.store.getState();
      s.setSlidingWindowLocked(true); s.setSlidingWindowSeconds(277 / 24); s.setDurationSeconds(8);
    });
    await sidebar.getByRole('button', {name: /^Duration:/}).click();
    await settle();
    await fitsSidebar();
    await drag(duration(), `Time slider at ${viewport.width}px`);
    await duration().press('End'); await settle();
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
    await dialog.getByRole('button', {name: 'Window', exact: true}).click();
    await settle();
    await drag(dialog.getByRole('slider', {name: 'Exact window count'}), `${id} ${workflow} window count`);
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
  console.log('Duration popup: sidebar bounds, stable range dragging, scrollable continuation and keyboard viewport passed');
}

module.exports = {assertDurationPopup};
