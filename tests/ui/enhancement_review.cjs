const assert = require('node:assert/strict');
const path = require('node:path');

async function assertEnhancementReview(page, output, llmRequests) {
  const plan = {source_prompt: 'A courier in a garden. No dialogue.', signature: 'saved-six',
    model_type: 'minimax_h3_ref2va_fused_turbo', resolution: '864x480', planning_style: 'adaptive',
    plan_kind: 'reference_sequence', total_frames: 1980, window_frames: 345, window_count: 6,
    native_continuation: true, retryable_windows: [4], camera_checkpoint: {version: 1, context: {image_paths: []}},
    planning_warnings: ["Window 4's camera plan needs review."], planned_by: 'deterministic_fallback',
    windows: Array.from({length: 6}, (_, i) => ({index: i + 1, title: `Scene ${i + 1}`,
      start_seconds: i * 14, end_seconds: (i + 1) * 14, start_frame: i * 327, end_frame: i * 327 + 345,
      opening_state: 'In the garden', closing_state: 'In the garden', prompt: `Saved prompt ${i + 1}`}))};
  plan.window_prompts = plan.windows.map(w => w.prompt);
  const params = {prompt: plan.source_prompt, model_type: plan.model_type, video_length: 1980};
  const job = {id: 'six-windows', status: 'failed', progress: 0, step: 0, totalSteps: 0,
    phase: '', message: '', outputFiles: [], error: 'Needs review', params,
    enhancement: {version: 1, state: 'review', warnings: plan.planning_warnings}};
  const data = {enhancement: {...job.enhancement, original_prompt: plan.source_prompt},
    original_params: params, prepared: {params: {...params, h3_window_prompts: plan.window_prompts}, h3_window_plan: plan}};
  const actions = [];
  let reject = true;
  await page.route('**/api/v1/jobs/six-windows/enhancement', route => route.fulfill({json: data}));
  await page.route('**/api/v1/jobs/six-windows/retry', route => {
    actions.push(route.request().postDataJSON().action);
    return reject ? route.fulfill({status: 503, json: {detail: 'Mock submission unavailable'}})
      : route.fulfill({json: {job_id: 'repair-queued', status: 'queued'}});
  });
  const mount = async () => {
    await page.evaluate(job => window.mountReview(job), job);
    const dialog = page.getByRole('dialog', {name: 'Review enhanced prompts'});
    await dialog.getByRole('button', {name: 'Retry window 4 & generate', exact: true}).waitFor();
    return dialog;
  };
  let dialog = await mount();
  assert.equal(await dialog.locator('summary').filter({hasText: /^Window [1-6]/}).count(), 6);
  await dialog.getByText('Needs review', {exact: true}).waitFor();
  await dialog.getByRole('button', {name: 'Generate all 6 windows with this draft', exact: true}).click();
  await dialog.getByRole('alert').getByText('Mock submission unavailable').waitFor();
  assert.equal(actions.at(-1), 'accept_draft');
  await dialog.getByRole('button', {name: 'Retry window 4 & generate', exact: true}).click();
  await dialog.getByText('Mock submission unavailable', {exact: true}).waitFor();
  assert.equal(actions.at(-1), 'retry');
  await dialog.getByText('Other options', {exact: true}).click();
  await dialog.getByRole('button', {name: 'Rewrite all prompts & generate', exact: true}).click();
  await dialog.getByText('Mock submission unavailable', {exact: true}).waitFor();
  assert.equal(actions.at(-1), 'refresh');
  await dialog.getByRole('button', {name: 'Generate all 6 windows from original prompt', exact: true}).click();
  await dialog.getByText('Mock submission unavailable', {exact: true}).waitFor();
  assert.equal(actions.at(-1), 'as_written');
  await dialog.getByText('Other options', {exact: true}).click();
  await page.setViewportSize({width: 390, height: 720});
  await dialog.getByRole('button', {name: 'Generate all 6 windows with this draft', exact: true}).scrollIntoViewIfNeeded();
  const bounds = await dialog.boundingBox();
  assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390 && bounds.y >= 0 && bounds.y + bounds.height <= 721);
  await page.screenshot({path: path.join(output, 'mobile-six-window-review.png')});
  const beforeEdit = actions.length;
  await page.evaluate(() => {
    window.store.setState({loadSettingsFromOutput: async () => {window.loadedReviewParams = window.store.getState().selectedOutputMeta.params;}});
  });
  await dialog.getByRole('button', {name: 'Edit prompts in Studio', exact: true}).click();
  await dialog.waitFor({state: 'hidden'});
  assert.equal(actions.length, beforeEdit, 'Editing never enqueues a job');
  assert.deepEqual(await page.evaluate(() => window.loadedReviewParams.h3_window_prompts), plan.window_prompts);
  reject = false;
  await page.route('**/api/v1/status/repair-queued', route => route.fulfill({json: {
    job_id: 'repair-queued', status: 'queued', progress: 0, message: 'Waiting', output_files: [], enhancement: {state: 'pending'},
  }}));
  dialog = await mount();
  await dialog.getByRole('button', {name: 'Retry window 4 & generate', exact: true}).click();
  await dialog.waitFor({state: 'hidden'});
  assert.equal(await page.evaluate(() => window.store.getState().jobs.some(j => j.id === 'repair-queued')), true);
  await page.setViewportSize({width: 1360, height: 900});

  // The interactive repair button passes the frozen plan, while the ordinary
  // Enhance action intentionally requests a fresh plan.
  await page.evaluate(plan => {
    window.resetFixture();
    const s = window.store.getState();
    window.store.setState({durationSeconds: 82.5, slidingWindowSeconds: 345 / 24, slidingWindowLocked: true,
      h3WindowPlan: plan, params: {...s.params, prompt: plan.source_prompt, video_length: 1980,
        minimax_h3_reference_sequence: true, minimax_h3_sequence_prompt_mode: 'adaptive', minimax_h3_window_storyboard: true}});
  }, plan);
  await page.evaluate(() => window.store.getState().enhancePrompt(undefined, 'adaptive', true));
  assert.deepEqual(llmRequests.at(-1).retry_plan, plan);
  await page.evaluate(() => window.store.getState().enhancePrompt(undefined, 'adaptive'));
  assert.equal(llmRequests.at(-1).retry_plan, undefined);
  console.log('PASS six-window review: full-job generation, targeted retry, full rewrite, source opt-out, edit-only, mobile layout and interactive repair.');
}
module.exports = {assertEnhancementReview};
