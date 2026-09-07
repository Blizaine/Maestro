// Called by sidebar_redesign.cjs with every backend mutation intercepted.
const assert = require('node:assert/strict');

async function assertExplicitEnhancement(page, sidebar, requests, llmRequests) {
  const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt'});
  const main = sidebar.getByRole('button', {name: 'Enhance prompt with AI Faithful'});
  const choices = sidebar.getByRole('button', {name: 'Prompt enhancement options'});
  const enhance = async (style = 'faithful') => {
    const before = llmRequests.length;
    const response = page.waitForResponse(response => /\/api\/v1\/llm\/(enhance-prompt|plan-h3-sequence|plan-h3-windows)$/.test(response.url()));
    if (style === 'faithful') await main.click();
    else {
      await choices.click();
      const menu = page.getByRole('menu', {name: 'Enhance prompt', exact: true});
      assert.deepEqual(await menu.getByRole('menuitem').allTextContents(), ['AI Faithful', 'AI Creative']);
      await menu.getByRole('menuitem', {name: 'AI Creative', exact: true}).click();
    }
    await response;
    await page.waitForFunction(() => !window.store.getState().isEnhancing);
    assert.equal(llmRequests.length, before + 1, 'Only an explicit click requests enhancement');
    assert.equal(llmRequests.at(-1).planning_style, style);
    assert.equal(await page.evaluate(() => window.store.getState().promptEnhanceError), null);
  };
  const reset = async (id, mode = 'video', workflow = 'frames') => {
    await page.evaluate(args => {
      window.resetFixture(...args);
      const s = window.store.getState();
      if (args[2] === 'references') s.setParam('minimax_h3_references', [{type: 'image', path: '/uploads/portrait.png', url: '/picture.svg'}]);
      if (args[1] === 'video' && args[2] === 'frames') s.setParam('image_start', '/uploads/frame.png');
    }, [id, mode, workflow]);
    await page.waitForTimeout(150);
  };

  // Single-pass text is used verbatim by both submission actions, even with
  // legacy Auto/Creative state or a saved deferred payload present.
  for (const [id, mode, workflow] of [
    ['minimax_h3_ref2va_fused_turbo', 'video', 'references'],
    ['minimax_h3_fused_turbo', 'video', 'frames'],
    ['flux2_klein_9b', 'image', 'frames'],
    ['ltx2_22B_distilled_1_1', 'video', 'frames'],
  ]) {
    await reset(id, mode, workflow);
    await prompt.fill('Keep my exact words.\nThis second paragraph is part of the same shot.');
    const verbatim = await prompt.inputValue(), before = llmRequests.length;
    await page.evaluate(() => {
      const s = window.store.getState();
      window.store.setState({params: {...s.params, minimax_h3_sequence_prompt_mode: 'creative', minimax_h3_window_storyboard: true,
        minimax_h3_reference_sequence: s.modelOptions.omni_reference === true,
        ltx_window_prompt_mode: 'auto', _deferred_prompt_enhance: {prompt: 'An old hidden prompt'}}});
    });
    for (const action of ['queue', 'now']) {
      const submitted = requests.length;
      await page.evaluate(action => window.store.getState().startGeneration(action), action);
      assert.equal(requests.length, submitted + 1, `${id}: ${action} submits without a hidden enhance`);
      assert.equal(requests.at(-1).prompt, verbatim);
      assert.equal(requests.at(-1)._deferred_prompt_enhance, undefined);
      assert.equal(llmRequests.length, before);
    }
    await page.evaluate(() => window.store.setState({isGenerating: false, jobs: []}));
    await enhance('creative');
    assert.match(await prompt.inputValue(), /^Enhanced creative/);
    await enhance();
    assert.match(await prompt.inputValue(), /^Enhanced faithful/, 'Main button remains Faithful after choosing Creative');
  }

  // Auto duration describes the visible story; hidden legacy prompt modes
  // cannot reinterpret paragraphs as windows or inflate Creative timing.
  await reset('minimax_h3_ref2va_fused_turbo', 'video', 'references');
  await prompt.fill('Blaine walks into a studio.\nHe sits at his desk.');
  await page.evaluate(() => window.store.getState().setParam('_duration_planning_mode', 'auto'));
  await page.waitForTimeout(200);
  const automaticDuration = await page.evaluate(() => window.store.getState().durationSeconds);
  for (const mode of ['manual', 'creative', 'auto']) {
    await page.evaluate(mode => window.store.getState().setParam('minimax_h3_sequence_prompt_mode', mode), mode);
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => window.store.getState().durationSeconds), automaticDuration);
  }

  // H3 long-form enhancement produces an editable plan. Generate/Queue reuse
  // the reviewed exact prompts and never invoke another LLM request.
  for (const [id, workflow, endpoint] of [
    ['minimax_h3_ref2va_fused_turbo', 'references', 'plan-h3-sequence'],
    ['minimax_h3_fused_turbo', 'frames', 'plan-h3-windows'],
  ]) {
    await reset(id, 'video', workflow);
    await page.evaluate(() => window.store.getState().setDurationSeconds(30));
    await page.waitForTimeout(150);
    await prompt.fill('A tutorial that develops over three scenes.');
    const before = llmRequests.length, submitted = requests.length;
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.length, submitted, 'A brief without window prompts is not silently planned');
    assert.equal(llmRequests.length, before);
    assert.match(await page.evaluate(() => window.store.getState().promptEnhanceError), /press Enhance/);
    await enhance('creative');
    assert.ok(llmRequests.at(-1).endpoint.endsWith(endpoint));
    await sidebar.getByRole('button', {name: /Exact H3 prompts/}).click();
    const review = page.getByRole('dialog', {name: 'H3 window prompts'});
    await review.locator('textarea[title]').first().fill('My exact first window revision.');
    await review.getByRole('button', {name: 'Done', exact: true}).click();
    const afterEnhance = llmRequests.length;
    for (const action of ['queue', 'now']) {
      await page.evaluate(action => window.store.getState().startGeneration(action), action);
      assert.equal(requests.at(-1)._h3_window_plan_reviewed, true);
      assert.equal(requests.at(-1).h3_window_prompts[0], 'My exact first window revision.');
      assert.equal(requests.at(-1).minimax_h3_sequence_prompt_mode, 'creative');
      assert.equal(llmRequests.length, afterEnhance);
    }
    // The same duration also supports hand-written lines without an AI plan.
    const count = requests.at(-1).h3_window_prompts.length;
    await page.evaluate(count => {
      const s = window.store.getState(); s.clearH3WindowPlan();
      s.setParam('prompt', Array.from({length: count}, (_, i) => `Manual window ${i + 1}.`).join('\n'));
    }, count);
    await page.evaluate(() => window.store.getState().startGeneration('queue'));
    assert.equal(requests.at(-1)._h3_window_plan_reviewed, undefined);
    assert.equal(requests.at(-1).h3_window_prompts.length, count);
    assert.equal(requests.at(-1).h3_window_prompts[0], 'Manual window 1.');
    assert.equal(llmRequests.length, afterEnhance);
  }

  await reset('ltx2_22B_distilled_1_1');
  await page.evaluate(() => window.store.getState().setDurationSeconds(30));
  await page.waitForTimeout(150);
  await prompt.fill('Blaine presents the three stages of a project.');
  await enhance();
  const enhanced = await prompt.inputValue(), before = llmRequests.length;
  assert.ok(enhanced.split('\n').length > 1, 'LTX enhancement exposes each window line');
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(requests.at(-1).ltx_window_prompt_mode, 'manual');
  assert.equal(requests.at(-1).prompt, enhanced);
  assert.deepEqual(requests.at(-1).ltx_window_prompts, enhanced.split('\n'));
  assert.equal(llmRequests.length, before);

  await reset('minimax_h3_voice_audio', 'audio');
  const speechResponse = page.waitForResponse(response => response.url().endsWith('/api/v1/llm/enhance-prompt'));
  await sidebar.getByRole('button', {name: 'Speech enhancement options'}).click();
  await sidebar.getByRole('button', {name: /Write (2-Person Dialogue|Dialogue \(2 speakers\))/}).first().click();
  await speechResponse;
  await page.waitForFunction(() => !window.store.getState().isEnhancing);
  assert.equal(llmRequests.at(-1).tts_enhance_mode, 'dialogue');
  assert.equal(await prompt.inputValue(), 'Blaine: Welcome to Maestro.');
  console.log('Explicit Faithful/Creative, unchanged Generate/Queue text, H3 reviewed/manual windows, LTX lines and TTS enhancement passed');
}
module.exports = {assertExplicitEnhancement};
