// Called by sidebar_redesign.cjs with every backend mutation intercepted.
const assert = require('node:assert/strict');
const path = require('node:path');
const screenshotPath = name => path.resolve(__dirname, '../../.codex-tmp/sidebar-validation', name);

async function assertExplicitEnhancement(page, sidebar, requests, llmRequests) {
  const prompt = sidebar.getByRole('textbox', {name: 'Generation prompt'});
  const main = sidebar.getByRole('button', {name: 'Enhance prompt', exact: true});
  const choices = sidebar.getByRole('button', {name: 'Prompt enhancement options'});
  const isArmed = () => page.evaluate(() => window.shouldEnhanceOnGeneration(window.store.getState()));
  const enhance = async (fromMenu = false) => {
    const style = await page.evaluate(() => String(window.store.getState().modelOptions?.architecture || '').startsWith('minimax_h3') ? 'adaptive' : 'faithful');
    const before = llmRequests.length;
    const response = page.waitForResponse(response => /\/api\/v1\/llm\/(enhance-prompt|plan-h3-sequence|plan-h3-windows)$/.test(response.url()));
    if (!fromMenu) await main.click();
    else {
      await choices.click();
      const menu = page.getByRole('menu', {name: 'Enhance prompt', exact: true});
      assert.deepEqual(await menu.getByRole('menuitem').allTextContents(), ['Enhance now']);
      await menu.getByRole('menuitemcheckbox', {name: 'Enhance on generation', exact: true}).waitFor();
      await menu.getByRole('menuitem', {name: 'Enhance now', exact: true}).click();
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
    ['krea2_turbo', 'image', 'frames'],
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
      if (mode === 'image') assert.equal(requests.at(-1).multi_prompts_gen_type, 2, 'Image paragraphs reach the engine as one prompt');
      assert.equal(requests.at(-1)._deferred_prompt_enhance, undefined);
      assert.equal(llmRequests.length, before);
    }
    await page.evaluate(() => window.store.setState({isGenerating: false, jobs: []}));
    await enhance(true);
    assert.match(await prompt.inputValue(), /^Enhanced (adaptive|faithful)/);
    await enhance();
    assert.match(await prompt.inputValue(), /^Enhanced (adaptive|faithful)/, 'Wand and Enhance now use the same writer');
    if (mode === 'image') {
      const provenance = await page.evaluate(() => window.store.getState().params._prompt_enhancement);
      assert.equal(provenance.original_prompt, verbatim, 'Repeated enhancement retains the source');
      assert.equal(provenance.enhanced_prompt, await prompt.inputValue());
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      assert.deepEqual(requests.at(-1)._prompt_enhancement, provenance, 'Generation carries source into saved image metadata');
      await prompt.fill('A different image brief.');
      assert.equal(await page.evaluate(() => window.store.getState().params._prompt_enhancement), undefined, 'A new brief cannot inherit an unrelated source');
      await page.evaluate(provenance => {
        const s = window.store.getState();
        window.store.setState({params: {...s.params, _prompt_enhancement: provenance}});
      }, provenance);
      await page.evaluate(() => window.store.getState().startGeneration('queue'));
      assert.equal(requests.at(-1)._prompt_enhancement, undefined, 'Recipe/default replacements cannot attach a stale source');
    }
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
    await page.setViewportSize({width: 390, height: 720});
    await page.evaluate(() => window.store.getState().setSidebarOpen(true));
    await sidebar.getByRole('button', {name: 'Generate', exact: true}).click();
    assert.equal(await page.evaluate(() => window.store.getState().sidebarOpen), true,
      'An unsubmitted draft keeps mobile controls and its validation message visible');
    assert.equal(requests.length, submitted);
    await sidebar.getByText(/This .* sequence needs .* press Enhance/).waitFor();
    await page.setViewportSize({width: 1360, height: 900});
    await enhance();
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
      assert.equal(requests.at(-1).minimax_h3_sequence_prompt_mode, 'adaptive');
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

  // H3 Frames planning and submission share one canonical schedule. Restored
  // overlap 17 normalizes to 18; the 336-frame story stays three windows
  // through a harmless same-model options refresh and preserves reviewed edits.
  await reset('minimax_h3_fused_turbo', 'video', 'frames');
  await page.evaluate(() => {
    const s = window.store.getState();
    window.store.setState({durationSeconds: 14, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 17,
      startImage: null, endImage: null,
      params: {...s.params, prompt: 'A brief story told in three connected scenes.', image_mode: 0,
        image_start: undefined, image_end: undefined, video_length: 336, sliding_window_size: 124,
        sliding_window_overlap: 17, minimax_h3_multi_window: true, minimax_h3_window_storyboard: true,
        custom_settings: {}, _duration_planning_mode: 'duration'}});
  });
  const timingPlanCalls = llmRequests.length;
  await enhance();
  const timingRequest = llmRequests.at(-1);
  assert.equal(timingRequest.endpoint, '/api/v1/llm/plan-h3-windows');
  assert.deepEqual({total: timingRequest.total_frames, window: timingRequest.window_frames,
    overlap: timingRequest.overlap_frames, discard: timingRequest.discard_frames,
    multi: timingRequest.minimax_h3_multi_window, custom: timingRequest.custom_settings},
  {total: 336, window: 124, overlap: 18, discard: 0, multi: true, custom: {}});
  assert.equal(llmRequests.length, timingPlanCalls + 1);
  assert.deepEqual(await page.evaluate(() => {
    const s = window.store.getState();
    return {frames: s.params.video_length, duration: s.durationSeconds,
      overlap: s.params.sliding_window_overlap, windows: s.h3WindowPlan?.windows.map(w => [w.start_frame, w.end_frame])};
  }), {frames: 336, duration: 14, overlap: 18, windows: [[0, 124], [124, 230], [230, 336]]});

  await page.evaluate(() => window.store.getState().updateH3WindowPrompt(0, 'Keep this reviewed opening exactly.'));
  await page.evaluate(async () => window.store.getState().loadModelOptions('minimax_h3_fused_turbo'));
  assert.deepEqual(await page.evaluate(() => {
    const s = window.store.getState();
    return {frames: s.params.video_length, duration: s.durationSeconds,
      prompt: s.h3WindowPlan?.windows[0]?.prompt, overlap: s.slidingWindowOverlap};
  }), {frames: 336, duration: 14, prompt: 'Keep this reviewed opening exactly.', overlap: 18},
  'A same-model options refresh keeps a valid reviewed Frames plan and its timing');
  const beforeReviewedSubmit = requests.length;
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(requests.length, beforeReviewedSubmit + 1);
  assert.deepEqual({frames: requests.at(-1).video_length, window: requests.at(-1).sliding_window_size,
    overlap: requests.at(-1).sliding_window_overlap, discard: requests.at(-1).sliding_window_discard_last_frames,
    count: requests.at(-1).h3_window_prompts?.length, reviewed: requests.at(-1)._h3_window_plan_reviewed,
    opening: requests.at(-1).h3_window_prompts?.[0]},
  {frames: 336, window: 124, overlap: 18, discard: 0, count: 3, reviewed: true,
    opening: 'Keep this reviewed opening exactly.'});

  // A restored reviewed plan with the same window count but different spans
  // must fail preflight rather than submit prompts against a changed schedule.
  await page.evaluate(() => {
    const s = window.store.getState();
    const plan = s.h3WindowPlan;
    window.store.setState({h3WindowPlan: {...plan, windows: plan.windows.map((window, index) => (
      index === 0 ? {...window, end_frame: window.end_frame + 1} : window
    ))}});
  });
  const beforeStaleSubmit = requests.length;
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(requests.length, beforeStaleSubmit);
  assert.match(await page.evaluate(() => window.store.getState().promptEnhanceError), /timing.*Press Enhance/);

  // An enabled multi-window toggle is not enough by itself: a single selected
  // pass still uses the native 345-frame length repair.
  await page.evaluate(() => {
    const s = window.store.getState();
    window.store.setState({jobs: [], isGenerating: false, h3WindowPlan: null,
      durationSeconds: 14, slidingWindowSeconds: 345 / 24, slidingWindowOverlap: 18,
      params: {...s.params, prompt: 'One native pass.', video_length: 336, sliding_window_size: 345,
        sliding_window_overlap: 18, minimax_h3_multi_window: true, minimax_h3_window_storyboard: true,
        image_mode: 0}});
  });
  const beforeSinglePass = requests.length;
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(requests.length, beforeSinglePass + 1);
  assert.equal(requests.at(-1).video_length, 345, 'A single native pass still rounds 336 to 345');

  // A real rolling request one frame above the native ceiling must bypass both
  // the max+1 clamp and lattice snap. Manual prompts keep this path LLM-free.
  await page.evaluate(() => {
    const s = window.store.getState();
    window.store.setState({jobs: [], isGenerating: false, h3WindowPlan: null,
      durationSeconds: 346 / 24, slidingWindowSeconds: 124 / 24, slidingWindowOverlap: 18,
      params: {...s.params, prompt: 'Window one.\nWindow two.\nWindow three.\nWindow four.',
        video_length: 346, sliding_window_size: 124, sliding_window_overlap: 18,
        minimax_h3_multi_window: true, minimax_h3_window_storyboard: false, image_mode: 0}});
  });
  const beforeLongRollingSubmit = requests.length;
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(requests.length, beforeLongRollingSubmit + 1);
  assert.equal(requests.at(-1).video_length, 346, 'A true 346-frame rolling timeline is not clamped to 345');

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

  // Defer writing into the same frozen job while a generation owns the GPU.
  // Arming is per submission; later Studio edits and later jobs are separate.
  for (const action of ['queue', 'now']) {
    await reset('minimax_h3_ref2va_fused_turbo', 'video', 'references');
    await page.evaluate(() => window.store.getState().setDurationSeconds(28));
    await prompt.fill('A new mountain duel. No dialogue.');
    await page.evaluate(() => window.store.setState({isGenerating: true}));
    await choices.click();
    await page.getByRole('menuitemcheckbox', {name: 'Enhance on generation', exact: true}).click();
    await sidebar.getByRole('button', {name: 'Remove Enhance on generation'}).waitFor();
    const calls = llmRequests.length, before = requests.length;
    const accepted = page.waitForResponse(response => response.url().endsWith('/api/v1/generate'));
    await sidebar.getByRole('button', {name: action === 'queue' ? 'Add current Studio settings to the queue' : 'Generate', exact: true}).click();
    await accepted;
    await page.waitForFunction(() => !window.store.getState().enhanceOnGeneration);
    assert.equal(requests.length, before + 1);
    assert.equal(llmRequests.length, calls, 'No browser-side LLM call before the queued job gets its turn');
    assert.equal(requests.at(-1)._enhance_on_generation, true);
    assert.equal(requests.at(-1)._queue_mode, action === 'queue' ? 'held' : 'now');
    assert.equal(requests.at(-1).prompt, 'A new mountain duel. No dialogue.');
    assert.equal(requests.at(-1).minimax_h3_sequence_prompt_mode, 'adaptive');
    assert.equal(await isArmed(), false);
    await prompt.fill('A later Studio edit.');
    assert.equal(requests.at(-1).prompt, 'A new mountain duel. No dialogue.');
  }
  await reset('flux2_klein_9b', 'image');
  await choices.click();
  await page.getByRole('menuitemcheckbox', {name: 'Enhance on generation'}).click();
  await page.route('**/api/v1/generate', route => route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({detail: 'Test submission failure'})}));
  await page.evaluate(() => window.store.getState().startGeneration('queue'));
  assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), true, 'Failed submission keeps its instruction');
  await page.unroute('**/api/v1/generate');
  await page.setViewportSize({width: 390, height: 720});
  const badge = sidebar.getByRole('button', {name: 'Remove Enhance on generation'});
  await page.waitForFunction(() => {
    const rect = document.querySelector('[aria-label="Remove Enhance on generation"]')?.getBoundingClientRect();
    return rect && rect.x >= 0 && rect.right <= window.innerWidth;
  });
  const rect = await badge.boundingBox();
  assert.ok(rect && rect.x >= 0 && rect.x + rect.width <= 390, 'Armed instruction fits mobile');
  await page.screenshot({path: screenshotPath('mobile-enhance-on-generation.png')});
  await badge.click();
  assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGeneration), false);
  await page.setViewportSize({width: 1360, height: 900});

  // The default is an opt-in preference. The badge can skip one accepted
  // submission without changing it; rejected submissions retain that skip.
  await reset('flux2_klein_9b', 'image');
  await prompt.fill('');
  assert.equal(await choices.isEnabled(), true, 'Preferences are available before writing');
  await choices.click();
  const defaultOption = page.getByRole('menuitemcheckbox', {name: 'Use by default', exact: true});
  assert.equal(await defaultOption.getAttribute('aria-checked'), 'false', 'Fresh installs do not opt in');
  assert.equal(await page.getByRole('menuitem', {name: 'Enhance now', exact: true}).isEnabled(), false);
  await defaultOption.click();
  assert.equal(await defaultOption.getAttribute('aria-checked'), 'true');
  await defaultOption.press('Escape');
  await page.setViewportSize({width: 390, height: 720});
  await page.waitForTimeout(150);
  await choices.click();
  const defaultMenu = page.getByRole('menu', {name: 'Enhance prompt', exact: true});
  const menuBounds = await defaultMenu.boundingBox();
  const sidebarBounds = await sidebar.boundingBox();
  const anchorBounds = await choices.boundingBox();
  assert.ok(menuBounds && menuBounds.x >= sidebarBounds.x && menuBounds.x + menuBounds.width <= sidebarBounds.x + sidebarBounds.width);
  assert.ok(menuBounds.y + menuBounds.height <= anchorBounds.y, 'Default preference opens above the wand');
  await page.screenshot({path: screenshotPath('mobile-enhance-default.png')});
  await defaultOption.press('Escape');
  await page.setViewportSize({width: 1360, height: 900});
  await prompt.fill('A quiet mountain lake.');
  assert.equal(await isArmed(), true);
  const defaultCalls = llmRequests.length;
  for (const action of ['queue', 'now']) {
    await sidebar.getByRole('button', {name: 'Remove Enhance on generation'}).click();
    assert.equal(await isArmed(), false);
    await page.route('**/api/v1/generate', route => route.fulfill({status: 503, json: {detail: 'Test rejection'}}));
    await page.evaluate(action => window.store.getState().startGeneration(action), action);
    assert.equal(await isArmed(), false, 'Rejected submission keeps the one-time skip');
    assert.equal(await page.evaluate(() => window.store.getState().enhanceOnGenerationDefault), true);
    await page.unroute('**/api/v1/generate');
    await page.evaluate(action => window.store.getState().startGeneration(action), action);
    assert.equal(requests.at(-1)._enhance_on_generation, undefined, 'Accepted skipped job uses visible text');
    assert.equal(await isArmed(), true, 'The default resumes after the skipped submission is accepted');
    await page.evaluate(action => window.store.getState().startGeneration(action), action);
    assert.equal(requests.at(-1)._enhance_on_generation, true);
    assert.equal(requests.at(-1)._queue_mode, action === 'queue' ? 'held' : 'now');
    assert.equal(await isArmed(), true, 'Opt-in default remains enabled for later jobs');
  }
  assert.equal(llmRequests.length, defaultCalls, 'Deferred enhancement never calls the writer from the browser');

  // Enhancement already performed now satisfies the default for both plain
  // text and multi-window plans, even when the same draft is submitted twice.
  for (const [id, mode, workflow, duration] of [
    ['flux2_klein_9b', 'image', 'frames', 5],
    ['ltx2_22B_distilled_1_1', 'video', 'frames', 30],
    ['minimax_h3_fused_turbo', 'video', 'frames', 30],
    ['minimax_h3_ref2va_fused_turbo', 'video', 'references', 30],
  ]) {
    await reset(id, mode, workflow);
    await page.evaluate(duration => {
      const s = window.store.getState(); s.setEnhanceOnGenerationDefault(true); s.setDurationSeconds(duration);
    }, duration);
    await page.waitForTimeout(150);
    await enhance();
    assert.equal(await isArmed(), false, 'Enhance now satisfies the default for its draft');
    const calls = llmRequests.length;
    for (const action of ['queue', 'now']) {
      await page.evaluate(action => window.store.getState().startGeneration(action), action);
      assert.equal(requests.at(-1)._enhance_on_generation, undefined, 'A completed draft is not enhanced twice');
      assert.equal(await isArmed(), false);
    }
    assert.equal(llmRequests.length, calls);
    await prompt.fill('A completely new brief.');
    assert.equal(await isArmed(), true, 'A new brief follows the saved default');
    await choices.click();
    await defaultOption.click();
    assert.equal(await isArmed(), false, 'Turning off the default restores manual generation');
    await defaultOption.press('Escape');
  }

  // A late response belongs to the submitted job, not a newer Studio choice.
  await reset('flux2_klein_9b', 'image');
  await page.evaluate(() => window.store.getState().setEnhanceOnGenerationDefault(true));
  let acceptSubmission, submissionStarted;
  const pendingSubmission = new Promise(resolve => {submissionStarted = resolve});
  await page.route('**/api/v1/generate', async route => {
    submissionStarted();
    await new Promise(resolve => {acceptSubmission = resolve});
    return route.fulfill({json: {job_id: 'late-acceptance', status: 'held'}});
  });
  await page.evaluate(() => {window.pendingSubmit = window.store.getState().startGeneration('queue')});
  await pendingSubmission;
  await sidebar.getByRole('button', {name: 'Remove Enhance on generation'}).click();
  acceptSubmission();
  await page.evaluate(() => window.pendingSubmit);
  assert.equal(await isArmed(), false, 'Late acceptance does not clear a newly chosen skip');
  await page.unroute('**/api/v1/generate');

  await reset('flux2_klein_9b', 'image');
  await page.evaluate(() => {
    const s = window.store.getState(); s.setEnhanceOnGenerationDefault(true);
    window.store.setState({studioImageWorkflow: 'inpaint'});
  });
  assert.equal(await isArmed(), false, 'Specialized image editing does not inherit generation enhancement');

  // Saved queue drafts remain reviewable after cancellation, completion, and
  // reload. Review/retry routes are intercepted just like generation itself.
  await page.evaluate(() => window.mountQueue());
  const saved = {id: 'enhanced-test', status: 'failed', progress: 0, phase: '',
    created_at: Date.now() / 1000, params: {model_type: 'minimax_h3_ref2va_fused_turbo', prompt: 'Original scene'},
    enhancement: {version: 1, state: 'review', warnings: ['Review the fallback draft.'], error: 'The writer needs a review before generation.'}};
  const savedData = {enhancement: {...saved.enhancement, original_prompt: 'Original scene', enhanced_prompt: 'Saved enhanced scene'},
    original_params: saved.params, prepared: {params: {...saved.params, prompt: 'Saved enhanced scene'}}};
  let retryAction, rejectRetry = true;
  await page.route('**/api/v1/jobs/enhanced-test/enhancement', route => route.fulfill({json: savedData}));
  await page.route('**/api/v1/jobs/enhanced-test/retry', route => {
    retryAction = route.request().postDataJSON().action;
    if (rejectRetry) return route.fulfill({status: 503, json: {detail: 'Submission unavailable; retry.'}});
    return route.fulfill({json: {job_id: 'retry-test', status: 'queued'}});
  });
  await page.evaluate(job => window.store.setState({jobs: [job], isGenerating: false}), saved);
  await page.getByRole('button', {name: /^Generation queue,/}).first().click();
  await page.getByRole('button', {name: 'Needs attention — review prompts'}).click();
  const jobReview = page.getByRole('dialog', {name: 'Review enhanced prompts'});
  await jobReview.getByRole('status').getByText('Generation is paused for review. Your draft is saved.', {exact: true}).waitFor();
  await jobReview.getByText('Review the fallback draft.', {exact: true}).waitFor();
  await jobReview.getByText('Saved enhanced scene', {exact: true}).waitFor();
  await page.setViewportSize({width: 390, height: 720});
  const reviewBounds = await jobReview.boundingBox();
  assert.ok(reviewBounds && reviewBounds.x >= 0 && reviewBounds.x + reviewBounds.width <= 390, 'Saved draft fits mobile');
  await page.screenshot({path: screenshotPath('mobile-enhanced-job-review.png')});
  await jobReview.getByRole('button', {name: 'Generate full job with this draft'}).click();
  await jobReview.getByText('Submission unavailable; retry.', {exact: true}).waitFor();
  assert.equal(await jobReview.isVisible(), true, 'Rejected submission leaves an actionable error in the review');
  assert.equal(await page.evaluate(() => window.store.getState().jobs.some(j => j.id === 'retry-test')), false);
  rejectRetry = false;
  let historyRequested = false;
  await page.route('**/api/v1/jobs', route => {
    historyRequested = true;
    return route.fulfill({status: 503, json: {detail: 'History refresh failed'}});
  });
  await page.route('**/api/v1/status/retry-test', route => route.fulfill({json: {
    job_id: 'retry-test', status: 'queued', progress: 0, step: 0, total_steps: 0, phase: '',
    message: 'Waiting for GPU', output_files: [], enhancement: {...saved.enhancement, state: 'complete'},
  }}));
  await jobReview.getByRole('button', {name: 'Generate full job with this draft'}).click();
  assert.equal(retryAction, 'accept_draft', 'A fallback runs only after explicit review');
  await jobReview.waitFor({state: 'hidden'});
  await page.waitForFunction(() => window.store.getState().jobs.some(j => j.id === 'retry-test' && j.status === 'queued'));
  assert.equal(await page.evaluate(() => window.store.getState().isGenerating), true);
  await page.getByText('Queued for generation', {exact: true}).waitFor();
  await page.waitForFunction(() => window.store.getState().jobs.find(j => j.id === 'retry-test')?.message === 'Waiting for GPU');
  assert.equal(historyRequested, false, 'An accepted job does not depend on a second queue history request');
  await page.unroute('**/api/v1/jobs');
  await page.unroute('**/api/v1/status/retry-test');
  await page.setViewportSize({width: 1360, height: 900});
  await page.evaluate(job => {
    window.store.setState({jobs: [{...job, status: 'running', phase: 'Enhancing'}], isGenerating: true});
    window.store.getState().stopGeneration(job.id);
  }, saved);
  assert.equal(await page.evaluate(() => window.store.getState().jobs.find(j => j.id === 'enhanced-test')?.status), 'cancelled');
  await page.unroute('**/api/v1/jobs/enhanced-test/enhancement');
  await page.unroute('**/api/v1/jobs/enhanced-test/retry');
  await page.evaluate(() => {window.queueRoot.unmount(); window.queueNode.remove()});

  await reset('minimax_h3_voice_audio', 'audio');
  const speechResponse = page.waitForResponse(response => response.url().endsWith('/api/v1/llm/enhance-prompt'));
  await sidebar.getByRole('button', {name: 'Speech enhancement options'}).click();
  await sidebar.getByRole('button', {name: /Write (2-Person Dialogue|Dialogue \(2 speakers\))/}).first().click();
  await speechResponse;
  await page.waitForFunction(() => !window.store.getState().isEnhancing);
  assert.equal(llmRequests.at(-1).tts_enhance_mode, 'dialogue');
  assert.equal(await prompt.inputValue(), 'Blaine: Welcome to Maestro.');
  console.log('Unified Enhance, deferred queue/now, failed submission, mobile badge, reviewed/manual H3 windows, LTX and TTS passed');
}
module.exports = {assertExplicitEnhancement};
