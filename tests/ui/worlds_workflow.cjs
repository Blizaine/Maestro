// Run: node tests/ui/worlds_workflow.cjs
// Mounts the real Worlds explorer and API client, with all API traffic
// intercepted in Chromium. No generation or live service is contacted.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const WORLD_ID = 'world-owned-by-create-receipt';
const DESCRIPTION = 'A quiet valley beneath a glass-blue moon';

function sha256Canonical(value) {
  const stable = Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)));
  return crypto.createHash('sha256').update(JSON.stringify(stable), 'utf8').digest('hex');
}

function assertPublicRequest(request, expectedFields) {
  assert.match(request.client_request_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    'Each operation gets a canonical version-4 request UUID');
  assert.equal(request.request_digest, sha256Canonical(expectedFields),
    'The browser computes the digest from the exact normalized public fields');
  assert.deepEqual(Object.keys(request).sort(), [...Object.keys(expectedFields), 'client_request_id', 'request_digest'].sort(),
    'The request contains only its typed public operation fields and durable identity');
  for (const serverOnly of ['model_type', 'custom_settings', 'world_output', 'video_length', 'num_inference_steps', 'resolution', 'workspace', 'custom_guide']) {
    assert.equal(Object.hasOwn(request, serverOnly), false, `${serverOnly} remains server-owned`);
  }
}

async function buildExplorerBundle() {
  const source = [
    "import * as React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {WorldExplorer} from './src/components/Worlds/WorldExplorer';",
    'function Harness() {',
    '  const [open, setOpen] = React.useState(false);',
    '  return React.createElement(React.Fragment, null,',
    '    React.createElement("button", {id: "open-worlds", onClick: () => setOpen(true)}, "Open World Explorer"),',
    '    React.createElement(WorldExplorer, {open, onClose: () => setOpen(false)}),',
    '  );',
    '}',
    'window.mountWorldExplorer = () => createRoot(document.getElementById("root")).render(React.createElement(Harness));',
    'window.mountWorldExplorer();',
  ].join('\n');
  const result = await esbuild.build({
    stdin: {contents: source, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  return result.outputFiles[0].text;
}

function jsonResponse(route, value, status = 200) {
  return route.fulfill({status, contentType: 'application/json', body: JSON.stringify(value)});
}

async function attachInterceptedApp(page, bundle, state) {
  await page.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== 'http://maestro.test') return route.abort('blockedbyclient');

    if (url.pathname === '/' && request.method() === 'GET') {
      state.documentLoads += 1;
      return route.fulfill({
        contentType: 'text/html',
        body: '<!doctype html><html><body><div id="root"></div><script src="/explorer.js"></script></body></html>',
      });
    }
    if (url.pathname === '/explorer.js' && request.method() === 'GET') {
      return route.fulfill({contentType: 'text/javascript', body: bundle});
    }

    if (url.pathname === '/api/v1/worlds' && request.method() === 'GET') {
      return jsonResponse(route, {worlds: state.worlds});
    }
    if (url.pathname === '/api/v1/worlds/requests' && request.method() === 'POST') {
      const body = request.postDataJSON();
      state.posts.push(body);
      if (body.operation === 'create') {
        state.createRequestId = body.client_request_id;
        // Model an accepted request whose POST response was lost. The client
        // must recover the owner through GET and must not POST it again.
        return route.abort('failed');
      }
      if (body.operation === 'move') {
        state.moveRequestId = body.client_request_id;
        return jsonResponse(route, {
          client_request_id: body.client_request_id,
          request_digest: body.request_digest,
          state: 'queued',
          job_id: 'job-owned-move',
          job_status: 'queued',
          cancel_state: 'none',
        });
      }
      return jsonResponse(route, {detail: 'Unexpected operation'}, 400);
    }

    const cancelMatch = url.pathname.match(/^\/api\/v1\/worlds\/requests\/([^/]+)\/cancel$/);
    if (cancelMatch && request.method() === 'POST') {
      const requestId = decodeURIComponent(cancelMatch[1]);
      state.cancels.push(requestId);
      return jsonResponse(route, {
        client_request_id: requestId,
        request_digest: state.posts.find(item => item.client_request_id === requestId)?.request_digest,
        state: 'queued',
        job_id: 'job-owned-move',
        job_status: 'queued',
        cancel_state: 'sent',
      });
    }

    const receiptMatch = url.pathname.match(/^\/api\/v1\/worlds\/requests\/([^/]+)$/);
    if (receiptMatch && request.method() === 'GET') {
      const requestId = decodeURIComponent(receiptMatch[1]);
      state.receiptGets[requestId] = (state.receiptGets[requestId] || 0) + 1;
      if (requestId === state.createRequestId) {
        if (state.documentLoads < 2) {
          return jsonResponse(route, {
            client_request_id: requestId,
            request_digest: state.posts.find(item => item.client_request_id === requestId).request_digest,
            state: 'queued',
            job_id: 'job-owned-create',
            job_status: 'queued',
          });
        }
        state.worlds = [state.publishedWorld];
        return jsonResponse(route, {
          client_request_id: requestId,
          request_digest: state.posts.find(item => item.client_request_id === requestId).request_digest,
          state: 'published',
          job_id: 'job-owned-create',
          job_status: 'completed',
          world: state.publishedWorld,
        });
      }
      if (requestId === state.moveRequestId) {
        return jsonResponse(route, {
          client_request_id: requestId,
          request_digest: state.posts.find(item => item.client_request_id === requestId).request_digest,
          state: 'queued',
          job_id: 'job-owned-move',
          job_status: 'queued',
          cancel_state: state.cancels.length ? 'sent' : 'none',
        });
      }
      return jsonResponse(route, {detail: 'Unknown request'}, 404);
    }

    const worldMatch = url.pathname.match(/^\/api\/v1\/worlds\/([^/]+)$/);
    if (worldMatch && request.method() === 'GET') {
      const worldId = decodeURIComponent(worldMatch[1]);
      const world = state.worlds.find(item => item.world_id === worldId);
      return world
        ? jsonResponse(route, {world})
        : jsonResponse(route, {detail: 'Unknown world'}, 404);
    }

    return route.abort('blockedbyclient');
  });
  await page.goto('http://maestro.test/');
  await page.getByRole('button', {name: 'Open World Explorer'}).click();
  await page.getByRole('dialog', {name: /World Explorer/}).waitFor({state: 'visible'});
}

async function main() {
  const bundle = await buildExplorerBundle();
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {})});
  try {
    const state = {
      documentLoads: 0,
      posts: [],
      cancels: [],
      receiptGets: {},
      worlds: [],
      createRequestId: null,
      moveRequestId: null,
      publishedWorld: {
        world_id: WORLD_ID,
        title: 'First saved valley',
        description: DESCRIPTION,
        segment_count: 1,
        recording_available: true,
      },
    };
    const page = await browser.newPage({viewport: {width: 1365, height: 900}});
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await attachInterceptedApp(page, bundle, state);

    const description = page.getByRole('textbox', {name: 'Create a world'});
    const create = page.getByRole('button', {name: 'Create world'});
    const acknowledgement = page.getByRole('checkbox', {name: /I will use the decoder only under these terms/});
    assert.equal(await acknowledgement.isChecked(), false, 'License acknowledgement starts unchecked');
    assert.equal(await create.isDisabled(), true, 'Creating a world is disabled until the terms are acknowledged');
    assert.equal(state.posts.length, 0, 'No generation request is submitted before acknowledgement');

    await acknowledgement.check();
    await description.fill(`  ${DESCRIPTION}  `);
    assert.equal(await create.isEnabled(), true, 'Acknowledgement enables the populated create form');
    await create.click();
    await page.getByText('Owned request: queued').waitFor({state: 'visible', timeout: 5000});
    assert.equal(state.posts.length, 1, 'An ambiguous create POST is sent exactly once');
    assert.deepEqual(state.receiptGets[state.createRequestId], 1, 'Ambiguous POST is reconciled with a receipt GET');

    const createFields = {
      acknowledge_pixelumm_noncommercial: true,
      description: DESCRIPTION,
      operation: 'create',
    };
    const createRequest = state.posts[0];
    assertPublicRequest(createRequest, createFields);
    assert.equal(createRequest.description, DESCRIPTION, 'The client trims the description before hashing and sending');
    const durableCreate = JSON.parse(await page.evaluate(() => localStorage.getItem('maestro.worlds.pending-request.v1')));
    assert.equal(durableCreate.request.client_request_id, createRequest.client_request_id,
      'The exact request UUID is saved before the ambiguous response is reconciled');
    assert.equal(durableCreate.request.request_digest, createRequest.request_digest,
      'The exact digest is saved alongside the request UUID');

    // A full browser reload retains localStorage. The second document sees a
    // published durable receipt and must not repeat the create POST.
    await page.reload();
    await page.getByRole('button', {name: 'Open World Explorer'}).click();
    await page.getByRole('heading', {name: 'First saved valley'}).waitFor({state: 'visible', timeout: 5000});
    assert.equal(state.posts.length, 1, 'Reload recovery performs GET only; it does not resubmit create');
    assert.ok(state.receiptGets[state.createRequestId] >= 2, 'Reload reads the same request UUID receipt');
    assert.equal(await page.evaluate(() => localStorage.getItem('maestro.worlds.pending-request.v1')), null,
      'A published receipt clears the durable pending request');

    // The published world is selected from the owned receipt. Forward must
    // continue that exact record; the client does not choose a model or path.
    await page.getByRole('button', {name: 'Forward'}).click();
    await page.getByText('Owned request: queued').waitFor({state: 'visible', timeout: 5000});
    assert.equal(state.posts.length, 2, 'The deliberate Forward action creates one new request');
    const moveRequest = state.posts[1];
    state.moveRequest = moveRequest;
    const moveFields = {
      acknowledge_pixelumm_noncommercial: true,
      action: 'Forward',
      duration_ms: 800,
      operation: 'move',
      world_id: WORLD_ID,
    };
    assertPublicRequest(moveRequest, moveFields);
    assert.equal(moveRequest.world_id, state.publishedWorld.world_id, 'Forward belongs to the exact published parent world');

    await page.getByRole('button', {name: 'Cancel this job'}).click();
    await page.getByRole('button', {name: 'Cancel sent'}).waitFor({state: 'visible', timeout: 5000});
    assert.deepEqual(state.cancels, [state.moveRequestId], 'Cancellation targets only the currently owned move request');
    const pendingAfterCancel = JSON.parse(await page.evaluate(() => localStorage.getItem('maestro.worlds.pending-request.v1')));
    assert.equal(pendingAfterCancel.request.client_request_id, state.moveRequestId);
    assert.equal(pendingAfterCancel.cancelAttempted, true, 'The one-shot cancel intent is persisted');

    await page.reload();
    await page.getByRole('button', {name: 'Open World Explorer'}).click();
    await page.getByRole('button', {name: 'Cancel sent'}).waitFor({state: 'visible', timeout: 5000});
    assert.equal(state.posts.length, 2, 'Reload does not repeat create or move POSTs');
    assert.deepEqual(state.cancels, [state.moveRequestId], 'Reload does not repeat the cancel POST');
    assert.equal(await page.getByRole('button', {name: 'Cancel sent'}).isDisabled(), true,
      'The restored one-shot cancellation is disabled');
    assert.deepEqual(pageErrors, [], 'The desktop workflow has no browser runtime errors');
    await page.close();

    // Exercise real dialog focus handling at a narrow mobile viewport.
    const mobile = await browser.newPage({viewport: {width: 390, height: 844}});
    const mobileErrors = [];
    mobile.on('pageerror', error => mobileErrors.push(error.message));
    const mobileState = {
      documentLoads: 0,
      posts: [],
      cancels: [],
      receiptGets: {},
      worlds: [],
      createRequestId: null,
      moveRequestId: null,
      publishedWorld: state.publishedWorld,
    };
    await attachInterceptedApp(mobile, bundle, mobileState);
    assert.equal(await mobile.evaluate(() => innerWidth), 390, 'The focus check uses a mobile-sized viewport');
    await mobile.waitForFunction(() => document.activeElement?.getAttribute('aria-label') === 'Close World Explorer');
    await mobile.keyboard.press('Escape');
    await mobile.getByRole('dialog').waitFor({state: 'detached'});
    assert.equal(await mobile.evaluate(() => document.activeElement?.id), 'open-worlds',
      'Closing the mobile explorer restores focus to its opening control');
    assert.deepEqual(mobileErrors, [], 'The mobile focus workflow has no browser runtime errors');
    await mobile.close();
  } finally {
    await browser.close();
  }
}

main().then(() => console.log('Worlds desktop workflow, durable recovery, ownership, cancellation, and mobile focus checks passed.')).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
