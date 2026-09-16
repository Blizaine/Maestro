const assert = require('node:assert/strict');
const path = require('node:path');

// Runs against the real queue component and store in sidebar.test. Every API
// request is intercepted; no user's history, media or generation is touched.
async function assertQueueHistory(page, output) {
  const job = (id, status) => ({id, status, progress: 0, step: 0, totalSteps: 0,
    phase: '', message: id, outputFiles: [], error: null,
    enhancement: {version: 1, state: 'complete', warnings: []}});
  const project = (id, status) => ({id, status, scene_description: id, message: '',
    created_at: 1, completed_at: status === 'completed' ? 2 : null,
    pipeline_type: 'short_film', image_model: 'image', video_model: 'video'});
  const jobs = [job('old-failure', 'failed'), ...Array.from({length: 24}, (_, i) => job(`done-${i}`, 'completed')),
    job('working', 'running'), job('waiting', 'queued'), job('held', 'held'), job('cancelled', 'cancelled')];
  let queue = {version: 1, paused: true, running: false, entries: [project('held-a', 'held'),
    project('director-done', 'completed'), project('held-b', 'held'), project('director-failed', 'failed')]};
  let storedJobs = [...jobs], failOne = true, deferredClear = false, finishClear;
  const deletes = [], reorders = [];
  const directorRoute = '**/api/v1/director/queue**';
  const jobsRoute = '**/api/v1/jobs**';
  await page.route(directorRoute, async route => {
    const url = new URL(route.request().url());
    if (route.request().method() === 'DELETE') {
      const id = decodeURIComponent(url.pathname.split('/').pop());
      assert.equal(url.searchParams.get('completed_only'), 'true', 'History deletion requests the completed-only guard');
      assert.equal(queue.entries.find(entry => entry.id === id)?.status, 'completed');
      deletes.push(id); queue = {...queue, entries: queue.entries.filter(entry => entry.id !== id)};
      return route.fulfill({json: {removed: true}});
    }
    if (url.pathname.endsWith('/reorder')) {
      const ids = route.request().postDataJSON().entry_ids;
      reorders.push(ids); queue = {...queue, entries: ids.map(id => queue.entries.find(entry => entry.id === id))};
    }
    return route.fulfill({json: queue});
  });
  await page.route(jobsRoute, async route => {
    const url = new URL(route.request().url());
    const id = decodeURIComponent(url.pathname.split('/').pop());
    if (route.request().method() === 'DELETE') {
      assert.equal(storedJobs.find(item => item.id === id)?.status, 'completed', 'Only successful jobs are cleared');
      deletes.push(id);
      if (id === 'done-0' && !deferredClear) {
        deferredClear = true;
        await new Promise(resolve => {finishClear = resolve});
      }
      if (id === 'done-2' && failOne) {failOne = false; return route.fulfill({status: 503, json: {detail: 'Try again'}})}
      storedJobs = storedJobs.filter(item => item.id !== id);
      return route.fulfill({json: {dismissed: true}});
    }
    if (url.pathname.endsWith('/enhancement')) return route.fulfill({json: {
      enhancement: {version: 1, state: 'complete', original_prompt: 'Original scene', enhanced_prompt: 'Saved draft', warnings: []},
      original_params: {prompt: 'Original scene'}, prepared: {params: {prompt: 'Saved draft'}},
    }});
    return route.fulfill({json: {jobs: storedJobs.map(item => ({...item, job_id: item.id, output_files: [], total_steps: 0}))}});
  });
  try {
    await page.evaluate(({jobs, queue}) => {
      window.store.setState({jobs, directorQueue: queue, pipelineId: null, pipelineStatus: null, isEnhancing: false, isGenerating: true});
      window.mountQueue();
    }, {jobs, queue});
    const toggle = page.getByRole('button', {name: 'Generation queue, 7 items'}).first();
    await toggle.click();
    const panel = page.getByRole('dialog', {name: 'Generation queue', exact: true});
    const completed = panel.getByRole('region', {name: 'Completed jobs'});
    const header = completed.getByRole('button', {name: 'Completed 25', exact: true});
    await header.waitFor();
    assert.equal(await header.getAttribute('aria-expanded'), 'false');
    assert.equal(await completed.getByRole('button', {name: 'View prompts'}).count(), 0);
    await panel.getByText('old-failure', {exact: true}).waitFor();
    await panel.getByText('director-failed', {exact: true}).waitFor();
    await panel.getByText('cancelled', {exact: true}).waitFor();

    // Hidden completed Director entries must not absorb an up/down click.
    await panel.getByText('held-b', {exact: true}).locator('../..').getByTitle('Move up', {exact: true}).click();
    await page.waitForFunction(() => window.store.getState().directorQueue?.entries[0]?.id === 'held-b');
    assert.deepEqual(reorders.at(-1), ['held-b', 'director-done', 'held-a', 'director-failed']);

    await page.setViewportSize({width: 390, height: 720});
    const box = await panel.boundingBox();
    assert.ok(box.x >= 0 && box.x + box.width <= 390 && box.y + box.height <= 720, 'Queue fits mobile');
    await header.scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(output, 'mobile-queue-completed-collapsed.png')});
    await header.click();
    assert.equal(await completed.getByRole('button', {name: 'View prompts'}).count(), 24);
    assert.equal(await completed.locator('[id] > div').first().getByText('done-23', {exact: true}).count(), 1, 'Newest saved job first');
    await completed.getByRole('button', {name: 'View prompts'}).first().click();
    const review = page.getByRole('dialog', {name: 'Job prompts'});
    await review.getByText('Saved draft', {exact: true}).waitFor();
    await review.getByText('Original prompt', {exact: true}).click();
    await review.getByText('Original scene', {exact: true}).waitFor();
    await review.getByRole('button', {name: 'Close job prompts'}).click();
    await toggle.click();
    await completed.getByRole('button', {name: 'View prompts'}).first().scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(output, 'mobile-queue-completed-expanded.png')});

    await completed.getByRole('button', {name: 'Clear completed', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('[aria-label="Completed jobs"] button[title^="Clear completed"]')?.disabled);
    for (let attempt = 0; !finishClear && attempt < 100; attempt++) await page.waitForTimeout(10);
    assert.ok(finishClear, 'The clear operation reached the mock backend');
    // A different job finishes while the server is clearing the snapshot.
    const justFinished = job('finished-during-clear', 'completed');
    storedJobs.push(justFinished);
    await page.evaluate(item => window.store.setState(state => ({jobs: [...state.jobs, item]})), justFinished);
    finishClear();
    await panel.getByRole('alert').getByText('Could not clear 1 completed entry. Please try again.', {exact: true}).waitFor();
    assert.equal(await completed.getByRole('button', {name: 'Completed 2', exact: true}).count(), 1);
    const remaining = await page.evaluate(() => window.store.getState().jobs.map(item => item.id));
    assert.deepEqual(remaining, ['old-failure', 'done-2', 'working', 'waiting', 'held', 'cancelled', 'finished-during-clear']);
    assert.equal(deletes.length, 25, 'Clear includes all 24 saved jobs and the completed Director entry');

    await completed.getByRole('button', {name: 'Clear completed', exact: true}).click();
    await page.waitForFunction(() => !window.store.getState().jobs.some(item => item.status === 'completed'));
    assert.equal(await panel.getByRole('region', {name: 'Completed jobs'}).count(), 0);
    assert.equal(await panel.getByRole('alert').count(), 0);
    await panel.getByText('old-failure', {exact: true}).waitFor();
    await panel.getByText('director-failed', {exact: true}).waitFor();
    assert.ok(await page.evaluate(() => window.store.getState().isGenerating), 'Clearing history does not stop active generation');

    // The mock backend retains deletion across the same reconnect used at startup.
    await page.evaluate(async () => {
      window.store.setState({jobs: []});
      await window.store.getState().reconnectJobs();
      await window.store.getState().loadDirectorQueue();
    });
    assert.equal(await page.evaluate(() => window.store.getState().jobs.some(item => item.status === 'completed')), false);
    assert.equal(await page.evaluate(() => window.store.getState().directorQueue.entries.some(item => item.status === 'completed')), false);
    console.log('Queue history: collapsed completed list, mobile prompts, reordering, safe clear, partial failure, concurrent completion and reconnect passed');
  } finally {
    finishClear?.();
    await page.evaluate(() => {window.queueRoot?.unmount(); window.queueNode?.remove()});
    await page.unroute(directorRoute); await page.unroute(jobsRoute);
    await page.setViewportSize({width: 1360, height: 900});
  }
}

module.exports = {assertQueueHistory};
