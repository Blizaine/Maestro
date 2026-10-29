const assert = require('node:assert/strict');
const path = require('node:path');

// Exercise the single prompt/input scroller and the pinned footer at short
// window sizes, with both system monitor layouts.
exports.assertComposerScrolling = async (page, sidebar, output) => {
  const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const body = page.getByTestId('studio-body-scroll');
  const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt', exact: true});
  const footerRoot = page.locator('[data-studio-footer]');
  const settingsStrip = page.getByTestId('studio-settings-strip');
  const generateBar = page.getByTestId('studio-generate-bar');
  const hardwareBar = sidebar.locator('.studio-hardware');
  const script = Array.from({length: 70}, (_, i) => `${i + 1}. A traveler pauses, then describes the next scene.\n\tKeep the framing consistent.`).join('\n\n') + '\n';
  const workflows = [
    ['minimax_h3', 'video', 'frames'],
    ['minimax_h3_ref2va', 'video', 'references'],
    ['flux2_klein_9b', 'image', 'frames'],
    ['minimax_h3_voice_audio', 'audio', 'frames'],
    ['minimax_h3_fused_turbo', 'video', 'extend'],
  ];
  const setHardwareExpanded = async expanded => {
    const collapsed = await hardwareBar.getByTitle('Show hardware status', {exact: true}).count() > 0;
    if (collapsed === expanded) {
      await hardwareBar.getByTitle(collapsed ? 'Show hardware status' : 'Collapse', {exact: true}).click();
      await settle();
    }
    assert.equal(await hardwareBar.getByTitle(expanded ? 'Collapse' : 'Show hardware status', {exact: true}).count(), 1,
      `System monitor is ${expanded ? 'expanded' : 'collapsed'}`);
  };
  const assertFooterAccess = async (viewport, expanded) => {
    const footer = await footerRoot.boundingBox();
    const settings = await settingsStrip.boundingBox();
    const generate = await generateBar.boundingBox();
    const hardware = await hardwareBar.boundingBox();
    assert.ok(footer && settings && generate && hardware, 'Footer and system monitor have visible layout boxes');
    assert.equal(await hardwareBar.isVisible(), true, 'System monitor remains visible');
    assert.ok(footer.y >= 0 && footer.y + footer.height <= viewport.height, 'Footer stays in the viewport');
    assert.ok(hardware.y >= 0 && hardware.y + hardware.height <= viewport.height, 'System monitor stays in the viewport');
    assert.ok(Math.abs(footer.y + footer.height - hardware.y) <= 1,
      `Footer sits directly above the ${expanded ? 'expanded' : 'collapsed'} system monitor`);
    assert.ok(settings.y >= footer.y && settings.y + settings.height <= footer.y + footer.height,
      'Settings strip stays inside the pinned footer');
    assert.ok(generate.y >= footer.y && generate.y + generate.height <= footer.y + footer.height,
      'Model and generation actions stay inside the pinned footer');

    const structure = await page.evaluate(() => {
      const scroller = document.querySelector('[data-testid="studio-body-scroll"]');
      const footer = document.querySelector('[data-studio-footer]');
      const settings = document.querySelector('[data-testid="studio-settings-strip"]');
      const generate = document.querySelector('[data-testid="studio-generate-bar"]');
      return {
        footerOutsideScroller: !!scroller && !!footer && !scroller.contains(footer),
        settingsOutsideScroller: !!scroller && !!settings && !scroller.contains(settings),
        generateOutsideScroller: !!scroller && !!generate && !scroller.contains(generate),
      };
    });
    assert.deepEqual(structure, {
      footerOutsideScroller: true,
      settingsOutsideScroller: true,
      generateOutsideScroller: true,
    }, 'Footer, settings, and Generate are outside the prompt/input scroller');

    const model = sidebar.getByRole('button', {name: 'Choose model', exact: true});
    const generateAction = generateBar.getByRole('button').filter({hasText: /^(Generate|Need |Choose canvas|Loading…)/});
    const queueAction = generateBar.locator('button[aria-label="Add current Studio settings to the queue"], button[aria-label="Add to queue unavailable"]');
    const generationActions = [['Model selector', model], ['Generate action', generateAction], ['Queue action', queueAction]];
    for (const [label, action] of generationActions) {
      assert.equal(await action.isVisible(), true, `${label} remains visible (count ${await action.count()})`);
      const box = await action.boundingBox();
      assert.ok(box && box.y >= generate.y && box.y + box.height <= generate.y + generate.height + 1,
        'Each model/generation action stays within the visible action bar');
    }

    const settingsActions = [
      sidebar.getByRole('button', {name: 'Open recipes', exact: true}),
      sidebar.getByRole('button', {name: /^Advanced settings/}),
      ...['Resolution', 'Aspect ratio', 'Duration'].map(label => sidebar.getByRole('button', {name: new RegExp(`^${label}:`)})),
    ];
    for (const action of settingsActions) {
      if (await action.count() === 0) continue;
      assert.equal(await action.isVisible(), true, 'Settings controls remain visible');
      const box = await action.boundingBox();
      assert.ok(box && box.y >= settings.y && box.y + box.height <= settings.y + settings.height + 1,
        'Each settings control stays inside the visible settings strip');
    }
  };
  for (const viewport of [{width: 1280, height: 520}, {width: 1000, height: 420}, {width: 390, height: 844}, {width: 320, height: 568}]) {
    await page.setViewportSize(viewport);
    for (const [id, mode, workflow] of workflows) {
      for (const hardwareExpanded of [false, true]) {
        await page.evaluate(args => window.resetFixture(...args), [id, mode, workflow]);
        await settle();
        await setHardwareExpanded(hardwareExpanded);
        await prompt.fill(script);
        await settle();
        await assertFooterAccess(viewport, hardwareExpanded);
        const footer = await generateBar.boundingBox();
        const settings = await settingsStrip.boundingBox();
        assert.ok(await prompt.evaluate(node => node.scrollHeight <= node.clientHeight + 1 && node.scrollTop === 0), 'The complete script fits the textarea');
        assert.ok(await body.evaluate(node => node.scrollHeight > node.clientHeight), 'The prompt/input body scrolls independently of the footer');
        assert.deepEqual(await body.evaluate(node => [...node.querySelectorAll('*')].filter(child => child.scrollHeight > child.clientHeight + 1 && ['auto', 'scroll'].includes(getComputedStyle(child).overflowY)).map(child => child.className)), [], 'No nested scrolling region inside the composer');
        await prompt.evaluate(node => {node.blur();});
        await body.evaluate(node => {node.scrollTop = 0;});
        await settle();
        const header = await page.getByTestId('studio-workflow-header').boundingBox();
        const area = await body.boundingBox();
        assert.ok(header.y >= area.y && header.y < area.y + area.height, 'The mode controls are reachable at the top');
        await prompt.scrollIntoViewIfNeeded();
        await settle();
        const headerBeforeWheel = await page.getByTestId('studio-workflow-header').boundingBox();
        const scrollBeforeWheel = await body.evaluate(node => node.scrollTop);
        const pointOverPrompt = async () => {
          const box = await prompt.boundingBox();
          const top = Math.max(box.y, area.y);
          const bottom = Math.min(box.y + box.height, area.y + area.height);
          assert.ok(bottom > top, `Long prompt is visible in the body scroller (prompt ${box.y}..${box.y + box.height}, body ${area.y}..${area.y + area.height})`);
          return {x: box.x + box.width / 2, y: top + (bottom - top) / 2};
        };
        const promptPoint = await pointOverPrompt();
        await page.mouse.move(promptPoint.x, promptPoint.y);
        await page.mouse.wheel(0, 400);
        await page.waitForTimeout(150);
        const start = await body.evaluate(node => node.scrollTop);
        assert.ok(start > scrollBeforeWheel, 'Real wheel input over the long prompt scrolls the body');
        assert.ok((await page.getByTestId('studio-workflow-header').boundingBox()).y < headerBeforeWheel.y, 'Mode controls scroll with the prompt and inputs');
        const secondPromptPoint = await pointOverPrompt();
        await page.mouse.move(secondPromptPoint.x, secondPromptPoint.y);
        await page.mouse.wheel(0, 400);
        await page.waitForTimeout(150);
        assert.ok(await body.evaluate((node, previous) => node.scrollTop > previous, start), 'Further wheel input over the textarea continues to scroll the outer body');
        assert.equal(await prompt.evaluate(node => node.scrollTop), 0, 'Wheel input does not create hidden textarea scrolling');
        assert.deepEqual(await generateBar.boundingBox(), footer, 'Model and Generate/queue actions remain fixed while the body scrolls');
        assert.deepEqual(await settingsStrip.boundingBox(), settings, 'Settings remain fixed while the body scrolls');
        await assertFooterAccess(viewport, hardwareExpanded);
        assert.equal(await page.evaluate(() => window.scrollY), 0, 'Scrolling does not move the underlying document/gallery');
        await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
        await settle();
        const toolbar = await sidebar.getByRole('group', {name: 'Prompt controls', exact: true}).boundingBox();
        assert.ok(toolbar.y >= area.y && toolbar.y + toolbar.height <= area.y + area.height + 1, 'Enhancement actions are reachable after the final prompt line');
        if (workflow === 'frames' && mode === 'video' && hardwareExpanded) {
          await page.screenshot({path: path.join(output, `composer-scroll-${viewport.width}x${viewport.height}.png`)});
        }
        await prompt.fill('A short scene.');
        await settle();
        assert.ok((await prompt.boundingBox()).height < 500, 'Deleting text shrinks the prompt again');
        await prompt.evaluate(node => node.blur());
        await body.evaluate(node => {node.scrollTop = 0;});
      }
    }
  }
  await setHardwareExpanded(false);
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  assert.ok((await prompt.boundingBox()).height >= 240, 'Short prompts still fill the available desktop writing area');
  console.log('Combined sidebar scrolling: five workflows, short desktop/mobile viewports, expanded/collapsed hardware, real wheel input over the long prompt, and pinned accessible actions passed');
};
