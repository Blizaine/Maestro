const assert = require('node:assert/strict');
const path = require('node:path');

// Exercise real wheel input over the textarea and reference area, at the short
// window sizes that previously trapped controls in separate scrolling regions.
exports.assertComposerScrolling = async (page, sidebar, output) => {
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const body = page.getByTestId('studio-body-scroll');
  const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt', exact: true});
  const script = Array.from({length: 70}, (_, i) => `${i + 1}. A traveler pauses, then describes the next scene.\n\tKeep the framing consistent.`).join('\n\n') + '\n';
  for (const viewport of [{width: 1280, height: 520}, {width: 1000, height: 420}, {width: 390, height: 844}, {width: 320, height: 568}]) {
    await page.setViewportSize(viewport);
    for (const [id, mode, workflow] of [['minimax_h3', 'video', 'frames'], ['minimax_h3_ref2va', 'video', 'references'], ['flux2_klein_9b', 'image', 'frames'], ['minimax_h3_voice_audio', 'audio', 'frames']]) {
      await page.evaluate(args => window.resetFixture(...args), [id, mode, workflow]);
      await settle();
      const hardware = sidebar.getByTitle('Show hardware status', {exact: true});
      if (await hardware.count()) await hardware.click();
      await prompt.fill(script);
      await settle();
      const footer = await page.getByTestId('studio-generate-bar').boundingBox();
      const settings = await page.getByTestId('studio-settings-strip').boundingBox();
      assert.ok(footer.y + footer.height <= viewport.height && settings.y > 0, 'Actions stay within the viewport');
      assert.ok(await prompt.evaluate(node => node.scrollHeight <= node.clientHeight + 1 && node.scrollTop === 0), 'The complete script fits the textarea');
      assert.ok(await body.evaluate(node => node.scrollHeight > node.clientHeight), 'The combined sidebar area scrolls');
      assert.deepEqual(await body.evaluate(node => [...node.querySelectorAll('*')].filter(child => child.scrollHeight > child.clientHeight + 1 && ['auto', 'scroll'].includes(getComputedStyle(child).overflowY)).map(child => child.className)), [], 'No nested scrolling region inside the composer');
      await prompt.evaluate(node => {node.blur();});
      await body.evaluate(node => {node.scrollTop = 0;});
      await settle();
      const header = await page.getByTestId('studio-workflow-header').boundingBox();
      const area = await body.boundingBox();
      assert.ok(header.y >= area.y && header.y < area.y + area.height, 'The mode controls are reachable at the top');
      await page.mouse.move(area.x + area.width / 2, area.y + area.height / 2);
      await page.mouse.wheel(0, 600);
      await page.waitForTimeout(150);
      const start = await body.evaluate(node => node.scrollTop);
      assert.ok(start > 0, 'Wheel input scrolls the body');
      assert.ok((await page.getByTestId('studio-workflow-header').boundingBox()).y < header.y, 'Mode controls scroll with the prompt and inputs');
      // At this point the pointer is over the long prompt, not its scrollbar.
      await page.mouse.wheel(0, 600);
      await page.waitForTimeout(150);
      assert.ok(await body.evaluate((node, previous) => node.scrollTop > previous, start), 'Wheel input over the textarea reaches the outer scroller');
      assert.equal(await prompt.evaluate(node => node.scrollTop), 0, 'Wheel input does not create hidden textarea scrolling');
      assert.deepEqual(await page.getByTestId('studio-generate-bar').boundingBox(), footer, 'Generate remains fixed while the body scrolls');
      assert.deepEqual(await page.getByTestId('studio-settings-strip').boundingBox(), settings, 'Settings remain fixed while the body scrolls');
      assert.equal(await page.evaluate(() => window.scrollY), 0, 'Scrolling does not move the underlying document/gallery');
      await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
      await settle();
      const toolbar = await sidebar.getByRole('group', {name: 'Prompt controls', exact: true}).boundingBox();
      assert.ok(toolbar.y >= area.y && toolbar.y + toolbar.height <= area.y + area.height + 1, 'Enhancement actions are reachable after the final prompt line');
      await prompt.fill('A short scene.');
      await settle();
      assert.ok((await prompt.boundingBox()).height < 500, 'Deleting text shrinks the prompt again');
      await prompt.evaluate(node => node.blur());
      await body.evaluate(node => {node.scrollTop = 0;});
      if (workflow === 'frames' && mode === 'video') await page.screenshot({path: path.join(output, `composer-scroll-${viewport.width}x${viewport.height}.png`)});
    }
  }
  await sidebar.getByTitle('Collapse', {exact: true}).click();
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  assert.ok((await prompt.boundingBox()).height >= 240, 'Short prompts still fill the available desktop writing area');
  console.log('Combined sidebar scrolling: four workflows, short desktop/mobile viewports, expanded hardware, wheel over prompt, growing/shrinking text and pinned actions passed');
};
