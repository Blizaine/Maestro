// Run with: node tests/ui/read_requests.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));

async function loadReadRequest() {
  const bundle = await esbuild.build({
    stdin: {
      contents: fs.readFileSync(path.join(root, 'ui/src/api/readRequest.ts'), 'utf8'),
      resolveDir: root,
      sourcefile: 'readRequest.ts',
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
  });
  const compiled = { exports: {} };
  new Function('module', 'exports', bundle.outputFiles[0].text)(compiled, compiled.exports);
  return compiled.exports;
}

function jsonResponse(value, ok = true) {
  return { ok, json: async () => value };
}

async function flush() {
  await new Promise(resolve => setImmediate(resolve));
}

(async () => {
  const { getJson } = await loadReadRequest();
  const originalFetch = global.fetch;
  const unhandledRejections = [];
  const recordUnhandled = reason => unhandledRejections.push(reason);
  process.on('unhandledRejection', recordUnhandled);

  try {
    let requestCount = 0;
    let releaseRequest;
    global.fetch = () => {
      requestCount += 1;
      return new Promise(resolve => { releaseRequest = resolve; });
    };

    const firstRead = getJson('/api/v1/models', 'Failed to fetch models', 500);
    const concurrentRead = getJson('/api/v1/models', 'Failed to fetch models', 500);
    await flush();
    assert.equal(requestCount, 1, 'concurrent identical reads share one fetch');
    releaseRequest(jsonResponse({ revision: 1 }));
    const results = await Promise.all([firstRead, concurrentRead]);
    assert.deepEqual(results, [{ revision: 1 }, { revision: 1 }]);

    global.fetch = async () => {
      requestCount += 1;
      return jsonResponse({ revision: 2 });
    };
    assert.deepEqual(await getJson('/api/v1/models', 'Failed to fetch models', 500), { revision: 2 });
    assert.equal(requestCount, 2, 'a later read gets a fresh response instead of a cached body');

    let failureAttempts = 0;
    global.fetch = async () => {
      failureAttempts += 1;
      return failureAttempts === 1
        ? jsonResponse({ detail: 'temporary failure' }, false)
        : jsonResponse({ recovered: true });
    };
    await assert.rejects(
      getJson('/api/v1/system-config', 'Failed to fetch system config', 500),
      error => error.message === 'Failed to fetch system config',
    );
    assert.deepEqual(await getJson('/api/v1/system-config', 'Failed to fetch system config', 500), { recovered: true });
    assert.equal(failureAttempts, 2, 'an HTTP failure releases the in-flight entry for recovery');

    const modelResponses = new Map();
    const modelCalls = [];
    global.fetch = url => {
      modelCalls.push(url);
      return new Promise(resolve => modelResponses.set(url, resolve));
    };
    const modelA = getJson('/api/v1/model-options/model-a', 'Failed to fetch model options', 500);
    const modelB = getJson('/api/v1/model-options/model-b', 'Failed to fetch model options', 500);
    await flush();
    assert.deepEqual(modelCalls.sort(), [
      '/api/v1/model-options/model-a',
      '/api/v1/model-options/model-b',
    ], 'different model keys use independent requests');
    modelResponses.get('/api/v1/model-options/model-b')(jsonResponse({ model: 'b' }));
    modelResponses.get('/api/v1/model-options/model-a')(jsonResponse({ model: 'a' }));
    assert.deepEqual(await Promise.all([modelA, modelB]), [{ model: 'a' }, { model: 'b' }]);

    let timeoutAttempts = 0;
    global.fetch = (_url, { signal }) => {
      timeoutAttempts += 1;
      return new Promise((_, reject) => {
        signal.addEventListener('abort', () => {
          const error = new Error('fetch aborted');
          error.name = 'AbortError';
          reject(error);
        }, { once: true });
      });
    };
    await assert.rejects(
      getJson('/api/v1/llm/models', 'Failed to fetch LLM models', 20),
      error => error.message === 'Failed to fetch LLM models' && error.name === 'TimeoutError',
    );
    await flush();
    assert.equal(timeoutAttempts, 1);
    global.fetch = async () => {
      timeoutAttempts += 1;
      return jsonResponse({ models: [] });
    };
    assert.deepEqual(await getJson('/api/v1/llm/models', 'Failed to fetch LLM models', 20), { models: [] });
    assert.equal(timeoutAttempts, 2, 'a timed-out request releases the in-flight entry for recovery');
    assert.deepEqual(unhandledRejections, [], 'aborting a timed-out fetch does not leave an unhandled rejection');

    console.log('Read requests: coalescing, freshness, failure recovery, model keys, and timeout recovery passed');
  } finally {
    process.off('unhandledRejection', recordUnhandled);
    global.fetch = originalFetch;
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
