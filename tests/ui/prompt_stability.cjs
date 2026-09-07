const assert = require('node:assert/strict');

// Called by sidebar_redesign.cjs with real model metadata and intercepted APIs.
// Browser writing extensions inject measured shadow overlays into the prompt.
// Exercise those DOM changes with visible native scrollbars, not just React updates.
exports.assertPromptStability = async (page, sidebar) => {
  const geometry = node => {
    const bounds = node.getBoundingClientRect();
    return [bounds.x, bounds.y, bounds.width, bounds.height, node.clientWidth, node.clientHeight];
  };
  const workflows = [
    ['minimax_h3', 'video', 'frames'],
    ['minimax_h3_ref2va', 'video', 'references'],
    ['flux2_klein_9b', 'image', 'frames'],
    ['minimax_h3_voice_audio', 'audio', 'frames'],
  ];
  for (const viewport of [{width: 1536, height: 682}, {width: 390, height: 844}]) {
    await page.setViewportSize(viewport);
    for (const fixture of workflows) {
      await page.evaluate(args => {
        window.resetFixture(...args);
        if (args[1] === 'image') {
          const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="tan"/></svg>';
          window.store.setState({imageRefs: [new File([svg], 'source.svg', {type: 'image/svg+xml'}), new File([svg], 'person.svg', {type: 'image/svg+xml'})]});
        }
      }, fixture);
      const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt', exact: true});
      await prompt.fill('');
      await page.waitForTimeout(150);
      const hardware = sidebar.getByTitle('Show hardware status', {exact: true});
      if (await hardware.count()) await hardware.click();
      assert.equal(await sidebar.getByTitle('Collapse', {exact: true}).count(), 1, 'Exercise the expanded hardware bar');
      await page.waitForTimeout(150);
      const empty = await prompt.evaluate(geometry);
      const toolbar = sidebar.getByRole('group', {name: 'Prompt controls', exact: true});
      const toolbarBounds = await toolbar.boundingBox();
      assert.ok(toolbarBounds.y >= empty[1] + empty[3], 'Prompt tools sit below the writing area: ' + fixture[0]);
      assert.equal(await sidebar.locator('.studio-composer').getByText('Prompt', {exact: true}).count(), 0, 'The dock has no redundant Prompt heading');
      await prompt.pressSequentially('this is a test', {delay: 30});
      assert.deepEqual(await prompt.evaluate(geometry), empty, 'First keystrokes keep the prompt frame steady: ' + fixture[0]);
      assert.deepEqual(await toolbar.boundingBox(), toolbarBounds, 'Enhancement controls appear without moving the toolbar');

      await prompt.evaluate(node => {
        const host = document.createElement('grammarly-extension');
        host.id = 'test-writing-overlay';
        host.style.cssText = 'position:absolute;top:0;left:0;width:0;height:0;pointer-events:none';
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:absolute;top:1px;left:1px;';
        host.attachShadow({mode: 'open'}).append(overlay);
        node.parentElement.prepend(host);
      });
      try {
        for (const long of [false, true]) {
          if (long) {
            await prompt.fill(Array.from({length: 40}, (_, n) => 'Line ' + n + ': Preserve the pose, framing, lighting, materials and detailed scenery.').join('\n'));
            assert.ok(await prompt.evaluate(node => node.scrollHeight > node.clientHeight), 'Long prompt scrolls internally');
            await prompt.evaluate(node => {node.setSelectionRange(10, 25); node.scrollTop = 100;});
          }
          assert.deepEqual(await prompt.evaluate(geometry), empty, 'Scrollbar appearance must not rewrap the prompt or change its frame');
          const before = await prompt.evaluate(node => [node.selectionStart, node.selectionEnd, node.scrollTop]);
          const measurements = await page.evaluate(async () => {
            const node = document.querySelector('[aria-label="Generation prompt"]');
            const overlay = document.getElementById('test-writing-overlay').shadowRoot.firstElementChild;
            const samples = [];
            for (let tick = 0; tick < 6; tick++) {
              // The extension alternates stale large measurements and corrected
              // bounds. This used to add both outer scrollbars on every update.
              overlay.style.width = tick % 2 ? '0px' : window.innerWidth + 'px';
              overlay.style.height = tick % 2 ? '0px' : window.innerHeight + 'px';
              const s = window.store.getState();
              window.store.setState({systemStats: {...s.systemStats, cpu: {percent: tick * 17}}});
              await new Promise(resolve => setTimeout(resolve, 250));
              const rect = node.getBoundingClientRect(), scroller = document.querySelector('.studio-composer-content');
              samples.push({geometry: [rect.x, rect.y, rect.width, rect.height, node.clientWidth, node.clientHeight],
                outerWidth: scroller.scrollWidth, outerClientWidth: scroller.clientWidth,
                selection: [node.selectionStart, node.selectionEnd, node.scrollTop]});
            }
            return samples;
          });
          for (const sample of measurements) {
            assert.deepEqual(sample.geometry, empty, 'Writing overlay and status polling must not resize the prompt: ' + fixture[0]);
            assert.ok(sample.outerWidth <= sample.outerClientWidth + 1, 'Injected overlays must not add an outer horizontal scrollbar');
            assert.deepEqual(sample.selection, before, 'Background updates preserve the caret, selection and scroll position');
          }
        }
      } finally {
        await page.evaluate(() => document.getElementById('test-writing-overlay')?.remove());
      }
      await prompt.fill('this is a test');
      assert.deepEqual(await prompt.evaluate(geometry), empty, 'Removing the scrollbar keeps the text width steady');
    }
  }
  await sidebar.getByTitle('Collapse', {exact: true}).click();
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(() => window.resetFixture());
  await page.waitForTimeout(150);
  console.log('Frames, References, Image and Speech prompt stability passed: first keystrokes, long scripts, writing-extension overlays, polling, caret and scrolling at desktop/mobile sizes');
};
