// CPU-only lifecycle/clock regressions; no browser or running app required.
const assert = require('node:assert/strict');
const path = require('node:path');
const esbuild = require(path.resolve(__dirname, '../../ui/node_modules/esbuild'));

(async () => {
  const bundle = await esbuild.build({entryPoints: [path.resolve(__dirname, '../../ui/src/lib/appStartup.ts')],
    bundle: true, platform: 'node', format: 'cjs', write: false});
  const compiled = {exports: {}};
  new Function('module', 'exports', bundle.outputFiles[0].text)(compiled, compiled.exports);
  const {createAppStartup} = compiled.exports;
  const originalSet = global.setTimeout, originalClear = global.clearTimeout;
  let now = 0, nextId = 0;
  const timers = new Map();
  global.setTimeout = (callback, delay) => { const id = ++nextId; timers.set(id, {at: now + delay, callback}); return id; };
  global.clearTimeout = id => timers.delete(id);
  const tick = async () => {
    const [id, timer] = [...timers].sort((left, right) => left[1].at - right[1].at)[0] || [];
    assert.ok(timer, 'a bounded retry/background task should be pending');
    timers.delete(id); now = timer.at; timer.callback();
    for (let index = 0; index < 8; index++) await Promise.resolve();
  };
  try {
    let calls = 0, ready = false, background = 0, release;
    const pending = new Promise(resolve => {release = resolve;});
    const startup = createAppStartup({loadControls: async () => {calls++; await pending; ready = true;},
      controlsReady: () => ready, loadBackground: () => background++, onError: () => assert.fail('unexpected startup error')});
    const first = startup.run(), second = startup.run();
    assert.equal(calls, 1, 'reconnect notifications share the active bootstrap');
    release(); await Promise.all([first, second]);
    assert.equal(background, 0, 'controls can paint before optional indexing starts');
    await tick(); assert.equal(background, 1);
    await startup.retry(); assert.equal(background, 1, 'a later recovery does not rescan background data');
    startup.dispose();

    calls = 0; ready = false; let errors = 0, recovered = false;
    const retrying = createAppStartup({loadControls: async () => {calls++; if (!recovered) throw new Error('server starting'); ready = true;},
      controlsReady: () => ready, loadBackground: () => background++, onError: () => errors++});
    await retrying.run();
    while (timers.size) await tick();
    assert.equal(calls, 5, 'automatic retry is bounded');
    assert.equal(errors, 5);
    assert.equal(now, 15150, 'failed controls use 1/2/4/8-second backoff');
    recovered = true; await retrying.retry(); await tick();
    assert.equal(calls, 6, 'manual/online recovery can reconnect after the automatic bound');
    assert.equal(background, 2);
    retrying.dispose();

    let lateRelease, lateErrors = 0;
    const late = createAppStartup({loadControls: async () => new Promise(resolve => {lateRelease = resolve;}),
      controlsReady: () => false, loadBackground: () => assert.fail('disposed mount started background work'), onError: () => lateErrors++});
    const completion = late.run(); late.dispose(); lateRelease(); await completion;
    assert.equal(lateErrors, 0, 'unmounted/StrictMode cleanup cannot report stale errors');
    assert.equal(timers.size, 0, 'cleanup removes every startup timer');
    console.log('Startup controller: coalescing, priority, bounded retries, recovery and disposal passed');
  } finally {global.setTimeout = originalSet; global.clearTimeout = originalClear;}
})().catch(error => {console.error(error); process.exitCode = 1;});
