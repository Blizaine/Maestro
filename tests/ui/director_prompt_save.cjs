const assert = require('node:assert/strict');

async function assertDirectorPromptSave(page, promptUpdates) {
  const pipeline = {
    pipeline_id: 'prompt-save-test', pipeline_type: 'short_film', status: 'completed',
    created_at: 1, total_time_sec: 1, scene_description: 'Prompt save fixture',
    shot_image_policy: 'generate', llm_log: {},
    clips: [{index: 0, tag: null, image_prompt: 'Original image', video_prompt: 'Original video',
      window_prompts: ['Original video'], start_image_filename: 'frame.png', video_filename: 'clip.mp4',
      window_count: 1, keyframe_prompts: [], keyframe_filenames: []}],
  };
  await page.evaluate(pipeline => {
    window.directorPipelineFixture = pipeline;
    window.store.setState({dashboardOpen: true, dashboardLoading: false,
      dashboardPipelineList: [{id: pipeline.pipeline_id, created_at: pipeline.created_at,
        pipeline_type: pipeline.pipeline_type, clip_count: 1, status: pipeline.status}],
      dashboardSelectedPipeline: pipeline});
    window.mountDashboard();
  }, pipeline);

  const dashboard = page.getByText('Dashboard', {exact: true}).locator('..').locator('..');
  const shot = page.getByText('Shot 1', {exact: false}).locator('..').locator('..');
  const imageEdit = shot.getByTitle('Edit prompt').first();
  await imageEdit.click();
  assert.equal(await imageEdit.isDisabled(), true, 'Active pencil cannot reset the draft');
  const imageArea = shot.locator('textarea').first();
  await imageArea.fill('Saved image prompt');
  await shot.getByTitle('Save image prompt').click();
  await page.waitForFunction(() => window.directorPromptUpdates?.length === 1);
  assert.deepEqual(promptUpdates[0].body, {image_prompt: 'Saved image prompt'});
  await shot.getByTitle('Edit prompt').nth(1).click();
  const videoArea = shot.locator('textarea').first();
  await videoArea.fill('Saved video prompt');
  await shot.getByTitle('Save video prompt').click();
  await page.waitForFunction(() => window.directorPromptUpdates?.length === 2);
  assert.deepEqual(promptUpdates[1].body, {video_prompt: 'Saved video prompt'});
  assert.equal(await page.getByText('Saved image prompt', {exact: true}).count(), 1);
  assert.equal(await page.getByText('Saved video prompt', {exact: true}).count(), 1);
  await page.evaluate(() => window.store.setState({dashboardOpen: false}));
  await dashboard.waitFor({state: 'detached'}).catch(() => {});
}

module.exports = {assertDirectorPromptSave};
