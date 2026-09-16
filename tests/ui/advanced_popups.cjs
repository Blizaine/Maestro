const assert = require('node:assert/strict');
const path = require('node:path');

// All LoRA data and interactions stay on sidebar_redesign's isolated origin.
async function assertAdvancedPopups(page, sidebar, output) {
  const filename = 'StudioStyle.safetensors';
  const guideText = 'Use a moderate adapter weight to preserve the scene lighting and composition. '.repeat(70);
  const routeLoras = async route => {
    const endpoint = new URL(route.request().url()).pathname;
    const body = endpoint.endsWith('/details') ? {loras: [{filename, guide: guideText, has_guide: true}]}
      : endpoint.includes('/guide/') ? {guide: guideText} : {loras: [filename]};
    return route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
  };
  await page.route('**/api/v1/loras/**', routeLoras);
  const settle = () => page.waitForTimeout(150);
  const trigger = sidebar.getByRole('button', {name: /^Advanced settings/});
  const panel = page.getByRole('dialog', {name: 'Advanced settings', exact: true});
  const section = key => panel.getByTestId('advanced-' + key);
  const summary = key => section(key).locator(':scope > summary');
  const disclose = async (key, open = true) => {
    if (await section(key).evaluate(node => node.open) !== open) await summary(key).click();
    await settle();
  };
  const fitsAdvanced = async () => {
    const bounds = await panel.boundingBox(), anchor = await trigger.boundingBox(), side = await sidebar.boundingBox();
    assert.ok(bounds.x >= side.x && bounds.x + bounds.width <= side.x + side.width, 'Advanced stays within the sidebar');
    assert.ok(bounds.y >= side.y && Math.abs(bounds.y + bounds.height - anchor.y + 6) < 1, 'Advanced opens directly above its button on every screen size');
    const generate = await page.getByTestId('studio-generate-bar').boundingBox();
    assert.ok(bounds.y + bounds.height < generate.y, 'Advanced does not cover the generation controls');
  };
  const guide = page.getByRole('tooltip', {name: 'Guide for StudioStyle'});
  const info = panel.getByRole('button', {name: 'Guide for StudioStyle'});
  const fitsGuide = async () => {
    const box = await guide.boundingBox();
    const viewport = await page.evaluate(() => ({left: visualViewport.offsetLeft, top: visualViewport.offsetTop, width: visualViewport.width, height: visualViewport.height}));
    assert.ok(box.x >= viewport.left + 7 && box.x + box.width <= viewport.left + viewport.width - 7, 'LoRA guide fits the visible screen horizontally');
    assert.ok(box.y >= viewport.top + 7 && box.y + box.height <= viewport.top + viewport.height - 7, 'Long LoRA guide fits above the keyboard and below the screen top');
    assert.ok(await guide.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Guide prose wraps without horizontal overflow');
    assert.ok(await guide.evaluate(node => {
      const rect = node.getBoundingClientRect();
      return node.contains(document.elementFromPoint(rect.left + 15, rect.top + 15));
    }), 'Guide is visible and receives pointer input outside the parent panel bounds');
  };
  try {
    for (const width of [1360, 767, 440, 390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.evaluate(async () => {
        window.resetFixture();
        const s = window.store.getState();
        window.store.setState({slidingWindowLocked: true, voiceCloneEnabled: false,
          params: {...s.params, seed: -1, activated_loras: [], negative_prompt: '', override_attention: '', skip_steps_cache_type: '',
            minimax_h3_turbo_mode: false, temporal_upsampling: '', custom_settings: {}, face_refiner: {enabled: false},
            minimax_h3_text_encoder: s.modelOptions.minimax_h3_text_encoder_default,
            minimax_h3_reference_detail: s.modelOptions.omni_reference_detail_default ?? 'match',
            video_prompt_type: '', image_refs: [], self_refiner_setting: 0}});
        await window.store.getState().loadLoras(s.params.model_type);
      });
      await settle();
      await trigger.click(); await settle();
      for (const key of ['loras', 'performance', 'finishing', 'generation']) await disclose(key, false);
      await fitsAdvanced();
      assert.equal(await panel.locator('summary [aria-label$=" active"]').count(), 0, 'Empty section badges are hidden');
      assert.equal(await trigger.getAttribute('aria-label'), 'Advanced settings', 'Duration-only window override is not counted in Advanced');
      await disclose('loras');
      await panel.getByRole('button', {name: 'StudioStyle', exact: true}).click();
      await disclose('loras', false);
      assert.equal(await summary('loras').getByLabel('1 active', {exact: true}).textContent(), '1', 'Collapsed LoRAs show one active adapter');
      await disclose('finishing');
      await panel.getByRole('checkbox', {name: 'Refine faces after generation'}).check();
      await disclose('finishing', false);
      assert.equal(await summary('finishing').getByLabel('1 active', {exact: true}).textContent(), '1');
      await disclose('performance');
      const detail = panel.getByLabel('Reference detail', {exact: true});
      const defaultDetail = await detail.inputValue();
      await detail.selectOption(defaultDetail === 'match' ? 'max' : 'match');
      await disclose('performance', false);
      assert.equal(await summary('performance').getByLabel('1 active', {exact: true}).textContent(), '1');
      await page.evaluate(() => window.store.getState().setParam('seed', 17)); await settle();
      assert.equal(await summary('generation').getByLabel('1 active', {exact: true}).textContent(), '1');
      assert.equal(await trigger.getAttribute('aria-label'), 'Advanced settings, 4 active', 'Total equals the section counts');
      await page.screenshot({path: path.join(output, `advanced-badges-${width}.png`)});
      await disclose('loras');
      await info.click(); await settle();
      await fitsGuide();
      assert.equal(await panel.isVisible(), true, 'Opening guide keeps Advanced open');
      assert.deepEqual(await page.evaluate(() => window.store.getState().params.activated_loras), [filename], 'Reading the guide does not toggle its LoRA');
      await guide.hover(); await page.mouse.wheel(0, 240); await settle();
      assert.ok(await guide.evaluate(node => node.scrollTop > 0), 'Long guide can be scrolled');
      assert.equal(await panel.isVisible(), true, 'Scrolling the nested guide keeps Advanced open');
      await page.screenshot({path: path.join(output, `lora-guide-${width}.png`)});
      await guide.press('Escape'); await settle();
      assert.equal(await guide.isVisible(), false, 'Escape dismisses the guide');
      assert.equal(await panel.isVisible(), true, 'Escape leaves the parent Advanced panel open');
      await info.click(); await settle();
      const outsidePoint = await panel.evaluate(node => {
        const rect = node.getBoundingClientRect(), tooltip = node.querySelector('[role="tooltip"]');
        return [[rect.left + 4, rect.top + rect.height / 2], [rect.right - 4, rect.top + rect.height / 2],
          [rect.left + rect.width / 2, rect.bottom - 4]].find(([x, y]) => {
            const target = document.elementFromPoint(x, y);
            return node.contains(target) && !tooltip.contains(target);
          });
      });
      assert.ok(outsidePoint, 'A portion of the parent remains available outside the guide');
      await page.mouse.click(...outsidePoint); await settle();
      assert.equal(await guide.isVisible(), false, 'Tapping outside the guide dismisses it');
      if (width === 390) {
        await panel.getByPlaceholder('Search LoRAs...').focus();
        await page.evaluate(() => {
          Object.defineProperty(visualViewport, 'height', {configurable: true, value: 370});
          Object.defineProperty(visualViewport, 'offsetTop', {configurable: true, value: 120});
          visualViewport.dispatchEvent(new Event('resize')); visualViewport.dispatchEvent(new Event('scroll'));
        });
        await settle(); await fitsAdvanced();
        await info.click(); await settle(); await fitsGuide();
        await guide.press('Escape');
        await page.evaluate(() => {
          delete visualViewport.height; delete visualViewport.offsetTop;
          visualViewport.dispatchEvent(new Event('resize')); visualViewport.dispatchEvent(new Event('scroll'));
        });
        await settle();
      }
      await panel.getByRole('button', {name: 'StudioStyle', exact: true}).click();
      await disclose('loras', false);
      assert.equal(await summary('loras').locator('[aria-label$=" active"]').count(), 0, 'Badge disappears when the last adapter is disabled');
      assert.equal(await trigger.getAttribute('aria-label'), 'Advanced settings, 3 active');
      await panel.getByRole('button', {name: 'Close Advanced settings'}).click();
    }
  } finally {
    await page.unroute('**/api/v1/loras/**', routeLoras);
    await page.setViewportSize({width: 1360, height: 900});
    await page.evaluate(async () => {
      window.resetFixture();
      await window.store.getState().loadLoras(window.store.getState().params.model_type);
    });
  }
  console.log('Advanced: section counts and total, mobile anchoring above Generate, bounded/scrolled LoRA guides, nested dismissal and keyboard viewport passed');
}

module.exports = {assertAdvancedPopups};
