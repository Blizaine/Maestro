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
  const outputSettings = page.getByTestId('studio-output-settings');
  const generateBar = page.getByTestId('studio-generate-bar');
  const promptTools = sidebar.getByRole('group', {name: 'Prompt controls', exact: true});
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
    const output = await outputSettings.boundingBox();
    const generate = await generateBar.boundingBox();
    const hardware = await hardwareBar.boundingBox();
    const tools = await promptTools.boundingBox();
    assert.ok(footer && settings && output && generate && hardware && tools, 'Footer rows, prompt controls, and system monitor have visible layout boxes');
    assert.equal(await hardwareBar.isVisible(), true, 'System monitor remains visible');
    assert.ok(footer.y >= 0 && footer.y + footer.height <= viewport.height, 'Footer stays in the viewport');
    assert.ok(hardware.y >= 0 && hardware.y + hardware.height <= viewport.height, 'System monitor stays in the viewport');
    assert.ok(Math.abs(footer.y + footer.height - hardware.y) <= 1,
      `Footer sits directly above the ${expanded ? 'expanded' : 'collapsed'} system monitor`);
    assert.ok(settings.y >= footer.y && settings.y + settings.height <= footer.y + footer.height,
      'The top settings row stays inside the pinned footer');
    assert.ok(output.x >= settings.x && output.x + output.width <= settings.x + settings.width
      && output.y >= settings.y && output.y + output.height <= settings.y + settings.height + 1,
      'Output settings stay inside the top settings row');
    assert.ok(generate.y >= footer.y && generate.y + generate.height <= footer.y + footer.height,
      'Recipes, Model Browser, model, and generation actions stay inside the bottom footer row');
    assert.ok(settings.y + settings.height <= generate.y + 1,
      'The settings row sits above the Recipes and generation row');
    assert.ok(tools.x >= settings.x && tools.x + tools.width <= settings.x + settings.width + 1
      && tools.y >= settings.y && tools.y + tools.height <= settings.y + settings.height + 1,
      'Prompt enhancement controls stay in the top pinned row');
    assert.ok(await footerRoot.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'The pinned footer has no horizontal overflow');
    assert.ok(await settingsStrip.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'The top footer row has no horizontal overflow');
    const outputSize = await outputSettings.evaluate(node => ({scrollWidth: node.scrollWidth, clientWidth: node.clientWidth}));
    assert.ok(outputSize.scrollWidth <= outputSize.clientWidth + 1,
      `Output chips have no horizontal overflow (${outputSize.scrollWidth}/${outputSize.clientWidth}) at ${viewport.width}x${viewport.height}`);
    assert.ok(await generateBar.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'The bottom footer row has no horizontal overflow');

    const structure = await page.evaluate(() => {
      const scroller = document.querySelector('[data-testid="studio-body-scroll"]');
      const footer = document.querySelector('[data-studio-footer]');
      const settings = document.querySelector('[data-testid="studio-settings-strip"]');
      const output = document.querySelector('[data-testid="studio-output-settings"]');
      const generate = document.querySelector('[data-testid="studio-generate-bar"]');
      const tools = document.querySelector('[role="group"][aria-label="Prompt controls"]');
      return {
        footerOutsideScroller: !!scroller && !!footer && !scroller.contains(footer),
        settingsOutsideScroller: !!scroller && !!settings && !scroller.contains(settings),
        outputInsideSettings: !!settings && !!output && settings.contains(output),
        generateOutsideScroller: !!scroller && !!generate && !scroller.contains(generate),
        promptToolsOutsideScroller: !!scroller && !!tools && !scroller.contains(tools),
        promptToolsInsideFooter: !!footer && !!tools && footer.contains(tools),
      };
    });
    assert.deepEqual(structure, {
      footerOutsideScroller: true,
      settingsOutsideScroller: true,
      outputInsideSettings: true,
      generateOutsideScroller: true,
      promptToolsOutsideScroller: true,
      promptToolsInsideFooter: true,
    }, 'Both pinned footer rows and prompt controls are outside the prompt/input scroller');

    const recipes = sidebar.getByRole('button', {name: 'Open recipes', exact: true});
    const browser = sidebar.getByRole('button', {name: 'Open model browser', exact: true});
    const model = sidebar.getByRole('button', {name: 'Choose model', exact: true});
    const generateAction = generateBar.getByRole('button').filter({hasText: /^(Generate|Need |Choose canvas|Loading…)/});
    const queueAction = generateBar.locator('button[aria-label="Add current Studio settings to the queue"], button[aria-label="Add to queue unavailable"]');
    const generationActions = [
      ['Recipes shortcut', recipes], ['Model Browser shortcut', browser], ['Model selector', model],
      ['Generate action', generateAction], ['Queue action', queueAction],
    ];
    const actionBoxes = [];
    for (const [label, action] of generationActions) {
      assert.equal(await action.isVisible(), true, `${label} remains visible (count ${await action.count()})`);
      const box = await action.boundingBox();
      assert.ok(box && box.x >= generate.x && box.x + box.width <= generate.x + generate.width + 1
        && box.y >= generate.y && box.y + box.height <= generate.y + generate.height + 1,
        `${label} stays fully inside the bottom action row`);
      actionBoxes.push(box);
    }
    assert.ok(actionBoxes.every((box, index) => index === 0 || actionBoxes[index - 1].x + actionBoxes[index - 1].width <= box.x + 1),
      'Recipes, Model Browser, model, Generate, and Queue stay in left-to-right order without overlapping');

    const settingsActions = [
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
        const tools = await promptTools.boundingBox();
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
        assert.deepEqual(await promptTools.boundingBox(), tools, 'Prompt enhancement controls remain fixed while the body scrolls');
        await assertFooterAccess(viewport, hardwareExpanded);
        assert.equal(await page.evaluate(() => window.scrollY), 0, 'Scrolling does not move the underlying document/gallery');
        await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
        await settle();
        const toolbar = await promptTools.boundingBox();
        assert.ok(toolbar.y >= settings.y && toolbar.y + toolbar.height <= settings.y + settings.height + 1,
          'Enhancement actions stay reachable in the pinned settings row after the body scrolls');
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

  // Crowded References content and a long duration must leave both compact
  // footer rows reachable at narrow phones and short desktop windows.
  await page.evaluate(() => window.resetFixture('minimax_h3_ref2va', 'video', 'references'));
  await settle();
  await page.evaluate(() => {
    const store = window.store.getState();
    store.setParam('_duration_planning_mode', 'duration');
    store.setParam('minimax_h3_references', [1, 2, 3].map(index => ({
      type: 'image', path: `/uploads/reference-${index}.png`, url: '/picture.svg', filename: `Reference ${index}`,
    })));
    store.setDurationSeconds(109.8);
  });
  await settle();
  assert.equal(await page.evaluate(() => window.store.getState().params.minimax_h3_references.length), 3,
    'The crowded References fixture contains all three supported images');
  await prompt.fill('A traveler looks toward the distant mountain.\nKeep both faces visible in the final frame.');
  await page.evaluate(() => window.store.getState().setEnhanceOnGeneration(true));
  await settle();
  const badge = sidebar.locator('.studio-prompt-field > button[aria-label="Remove Enhance on generation"]');
  await badge.waitFor();
  const badgeClearance = await page.evaluate(() => {
    const field = document.querySelector('.studio-prompt-field');
    const prompt = field?.querySelector('textarea[aria-label="Generation prompt"]');
    const badge = field?.querySelector(':scope > button[aria-label="Remove Enhance on generation"]');
    const mirror = field?.querySelector('[data-prompt-mirror]');
    const text = mirror?.firstChild?.textContent?.trimEnd() || '';
    const lastLineStart = text.lastIndexOf('\n') + 1;
    const range = document.createRange();
    if (mirror?.firstChild) { range.setStart(mirror.firstChild, lastLineStart); range.setEnd(mirror.firstChild, text.length); }
    const line = range.getBoundingClientRect();
    const badgeBox = badge?.getBoundingClientRect();
    const style = prompt && getComputedStyle(prompt);
    return {insideField: !!field && !!prompt && !!badge && field.contains(badge),
      bottomPadding: style ? parseFloat(style.paddingBottom) : 0,
      badgeHeight: badgeBox?.height || 0,
      lastLineClear: !!line && !!badgeBox && (line.bottom <= badgeBox.top || line.top >= badgeBox.bottom
        || line.right <= badgeBox.left || line.left >= badgeBox.right)};
  });
  assert.equal(badgeClearance.insideField, true, 'The enabled badge stays attached to the References prompt');
  assert.ok(badgeClearance.bottomPadding >= badgeClearance.badgeHeight + 8,
    'The prompt reserves bottom padding for the enabled badge');
  assert.equal(badgeClearance.lastLineClear, true, 'The badge does not cover the prompt’s final line');
  await setHardwareExpanded(false);
  const durationChip = sidebar.getByRole('button', {name: /^Duration:/});
  for (const viewport of [{width: 1280, height: 520}, {width: 1000, height: 420}, {width: 390, height: 844}, {width: 320, height: 568}]) {
    await page.setViewportSize(viewport);
    await settle();
    await assertFooterAccess(viewport, false);
    const durationLabel = await durationChip.getAttribute('aria-label');
    const durationTitle = await durationChip.getAttribute('title');
    const durationText = (await durationChip.innerText()).replace(/\s+/g, ' ').trim();
    assert.match(durationText, /1:49\.\d/, 'The long duration is shown in a compact time format');
    assert.match(durationLabel, /1m 49\.\d+s/, 'The duration accessible name keeps its complete readable value');
    assert.match(durationTitle, /1m 49\.\d+s/, 'The duration tooltip keeps its complete readable value');
    assert.ok((await durationChip.boundingBox()).width < 100, 'The long duration chip stays compact');
    assert.ok(await sidebar.evaluate(node => node.scrollWidth <= node.clientWidth + 1),
      `Crowded References content does not create horizontal overflow at ${viewport.width}x${viewport.height}`);
    if (viewport.width <= 390) {
      const characters = sidebar.getByRole('button', {name: /^Characters/});
      assert.equal(await characters.isVisible(), true, 'The compact Characters shortcut remains visible in the crowded top row');
      const characterBox = await characters.boundingBox(), settings = await settingsStrip.boundingBox();
      assert.ok(characterBox.y >= settings.y && characterBox.y + characterBox.height <= settings.y + settings.height + 1,
        'Characters stays in the pinned top row on phones');
    }
    if (viewport.width === 1280 || viewport.width === 390) {
      const pinnedFooter = await footerRoot.boundingBox();
      await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
      await settle();
      await badge.scrollIntoViewIfNeeded();
      await settle();
      assert.equal(await badge.isVisible(), true, 'The enabled prompt badge remains visible for visual review');
      const badgeBox = await badge.boundingBox(), bodyBox = await body.boundingBox();
      assert.ok(badgeBox && bodyBox && badgeBox.y >= bodyBox.y
        && badgeBox.y + badgeBox.height <= bodyBox.y + bodyBox.height + 1,
      'The prompt and enabled badge are visible inside the scrolling body for the screenshot');
      assert.deepEqual(await footerRoot.boundingBox(), pinnedFooter,
        'Bringing the prompt badge into view leaves both pinned footer rows anchored');
      await page.screenshot({path: path.join(output, `composer-references-badge-${viewport.width}px.png`)});
    }
  }

  await setHardwareExpanded(false);
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await settle();
  assert.ok((await prompt.boundingBox()).height >= 240, 'Short prompts still fill the available desktop writing area');
  console.log('Combined sidebar scrolling: five workflows, crowded References, compact long duration, short desktop/mobile viewports, expanded/collapsed hardware, real wheel input, and two pinned footer rows passed');
};
