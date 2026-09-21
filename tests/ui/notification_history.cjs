// Real React + Zustand notification regression tests with an isolated API.
// Run: node tests/ui/notification_history.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT ||
  'playwright');

(async () => {
  const bundle = await esbuild.build({
    stdin: { contents: `
      import React, { StrictMode, useEffect } from 'react';
      import { createRoot } from 'react-dom/client';
      import { useStore } from './src/stores/useStore';
      import { NotificationCoordinator } from './src/components/NotificationCoordinator';
      import { NotificationToastHost } from './src/components/NotificationToastHost';
      import { subscribeMaestroAlerts } from './src/lib/notifications';
      window.store = useStore;
      window.alerts = [];
      subscribeMaestroAlerts(alert => window.alerts.push(alert));
      function Ready() { useEffect(() => { window.ready = true; }, []); return null; }
      createRoot(document.getElementById('root')).render(<StrictMode>
        <NotificationCoordinator/><NotificationToastHost/><Ready/>
      </StrictMode>);
    `, resolveDir: path.join(root, 'ui'), loader: 'tsx' },
    bundle: true, write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent',
  });
  const browser = await playwright.chromium.launch({headless: true,
    ...(process.env.MAESTRO_CHROME || process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    const statusRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    const job = (id, status, outputs = 1) => ({
      job_id: id, status, progress: status === 'completed' ? 100 : 0,
      step: 0, total_steps: 0, phase: status, message: status, error: null,
      output_files: Array.from({length: outputs}, (_, i) => `${id}-${i}.mp4`),
      enhancement: {status: 'completed'}, created_at: 1,
    });
    let history = [job('old-sequence', 'completed', 6), ...['a', 'b', 'c'].map(
      id => job(`old-${id}`, 'completed')), job('old-failure', 'failed'), job('old-cancelled', 'cancelled')];
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/v1/jobs') return route.fulfill({json: {jobs: history}});
      if (url.pathname.startsWith('/api/v1/jobs/')) {
        const id = url.pathname.split('/').pop();
        statusRequests.push(id);
        return route.fulfill({json: history.find(item => item.job_id === id) || job(id, 'completed')});
      }
      if (route.request().method() === 'POST') {
        return route.fulfill({status: 500, json: {detail: 'Simulated submission failure'}});
      }
      if (url.pathname.startsWith('/api/')) return route.fulfill({json: {files: [], total: 0}});
      return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
    });
    const open = async () => {
      await page.goto('http://notifications.test');
      // A new browser session has no deduplication history to hide this bug.
      await page.evaluate(() => sessionStorage.clear());
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      await page.waitForFunction(() => window.ready);
    };
    const alerts = () => page.evaluate(() => window.alerts);
    await open();
    await page.evaluate(() => window.store.getState().reconnectJobs());
    assert.equal(await page.evaluate(() => window.store.getState().jobs.length), 6, 'History remains in the queue');
    assert.deepEqual(await alerts(), [], 'Opening Maestro must not replay historical completions, failures or cancellations');
    assert.equal(await page.getByRole('status').count(), 0, 'No stale completion toast covers the UI');
    await page.waitForTimeout(2200);
    assert.deepEqual(statusRequests, [], 'Terminal history does not start unnecessary status polling');
    console.log('Saved Studio history opens quietly and remains available');

    history.push(job('live-a', 'running', 0), job('live-b', 'queued', 0));
    await open();
    await page.evaluate(() => window.store.getState().reconnectJobs());
    assert.deepEqual(await alerts(), [], 'Reopening with active jobs and saved history stays quiet');
    await page.evaluate(() => {
      window.store.setState(s => ({jobs: s.jobs.map(job => job.id === 'live-a'
        ? {...job, status: 'completed', outputFiles: ['new.mp4']} : job)}));
      window.store.setState({sidebarOpen: true});
    });
    assert.deepEqual((await alerts()).map(a => a.title), ['Generation complete'], 'An observed completion notifies once');
    await page.evaluate(() => {
      window.store.setState(s => ({jobs: s.jobs.map(job => job.id === 'live-b'
        ? {...job, status: 'failed', error: 'New error'} : job)}));
    });
    const batch = await alerts();
    assert.deepEqual(batch.map(a => a.title), ['Generation complete', 'Generation failed', 'Queue finished with errors']);
    assert.equal(batch[2].body, '1 finished, 1 failed', 'Summary excludes all historical jobs');
    await page.evaluate(() => window.store.getState().reconnectJobs());
    assert.equal((await alerts()).length, 3, 'Refreshing queue history does not repeat alerts');
    console.log('Active Studio jobs, live failures and queue summaries still notify');

    await open();
    await page.evaluate(() => {
      window.store.setState({directorQueue: {entries: [{id: 'old-director', status: 'completed'}]}});
      window.store.setState({directorQueue: {entries: [
        {id: 'old-director', status: 'completed'}, {id: 'late-history', status: 'failed'},
      ]}});
      window.store.setState({pipelineId: 'old-project', pipelineStatus: {id: 'old-project', status: 'completed'}});
      window.store.setState({pipelineId: 'other-project', pipelineStatus: {id: 'other-project', status: 'failed'}});
    });
    assert.deepEqual(await alerts(), [], 'Director restore and switching saved projects stay quiet');
    await page.evaluate(() => {
      window.store.setState({pipelineId: 'new-project', pipelineStatus: null, pipelinePolling: true});
      window.store.setState({pipelineStatus: {id: 'new-project', status: 'completed'}});
      window.store.setState({pipelinePolling: false});
      window.store.setState({directorQueue: {entries: [{id: 'new-queue', pipeline_id: 'new-project', status: 'running'}]}});
      window.store.setState({directorQueue: {entries: [{id: 'new-queue', pipeline_id: 'new-project', status: 'completed'}]}});
    });
    assert.deepEqual((await alerts()).map(a => a.title), ['Director project complete'], 'Fast Director completion still notifies, and its queue view does not duplicate it');
    console.log('Director history is quiet; genuine pipeline and queue transitions still notify');

    await open();
    await page.evaluate(async () => {
      window.store.setState({toolsSourcePath: 'test.mp4', toolsTool: 'film_grain', filmGrainIntensity: 1});
      await window.store.getState().runTool();
    });
    assert.deepEqual((await alerts()).map(a => a.title), ['Generation failed'], 'Local pre-submission failures must remain visible');
    assert.deepEqual(errors, [], 'No browser runtime errors');
    console.log('Submission errors still alert; all notification history checks passed');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
