// Core v4 theme migration regression. Runs with Node + the UI's esbuild dev dependency;
// browser/prepaint/CSS/OS integration remains covered by graphite_theme.cjs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const esbuild = require(path.resolve(__dirname, '../../ui/node_modules/esbuild'));

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'ui/index.html'), 'utf8');
const prepaint = html.match(/<div id="root"><\/div>\s*<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(prepaint, 'index.html keeps its pre-mount theme script next to the app root');

const runtime = esbuild.buildSync({
  absWorkingDir: path.join(root, 'ui'),
  stdin: {
    contents: `
      import {FAMILIES, getStoredPrefs, resolveTheme, applyThemePrefs, onOsThemeChange} from './src/lib/theme';
      window.themeApi = {FAMILIES, getStoredPrefs, resolveTheme, applyThemePrefs, onOsThemeChange};
    `,
    resolveDir: path.join(root, 'ui'),
    loader: 'ts',
  },
  bundle: true,
  write: false,
  format: 'iife',
  platform: 'browser',
  logLevel: 'silent',
}).outputFiles[0].text;

const KEYS = {
  marker: 'maestro-theme-v4-migrated',
  mode: 'maestro-theme-mode',
  family: 'maestro-theme-family',
  legacy: 'maestro-theme',
  interim: 'maestro-theme-dark',
};

class MemoryStorage {
  constructor(values = {}, options = {}) {
    this.values = new Map(Object.entries(values));
    this.failGetKey = options.failGetKey;
    this.failSetKey = options.failSetKey;
    this.writes = [];
  }

  getItem(key) {
    if (this.failGetKey === key) throw new Error(`blocked getItem(${key})`);
    return this.values.has(key) ? this.values.get(key) : null;
  }

  setItem(key, value) {
    this.writes.push(key);
    if (this.failSetKey === key) throw new Error(`blocked setItem(${key})`);
    this.values.set(key, String(value));
  }
}

function openPage({storage = new MemoryStorage(), colorScheme = 'dark', runPrepaint = true} = {}) {
  const listeners = [];
  const attrs = new Map();
  const htmlElement = {
    getAttribute: key => attrs.has(key) ? attrs.get(key) : null,
    setAttribute: (key, value) => attrs.set(key, String(value)),
    removeAttribute: key => attrs.delete(key),
    classList: {add() {}, remove() {}},
  };
  const meta = {
    content: '#111316',
    setAttribute(key, value) { if (key === 'content') this.content = String(value); },
    getAttribute(key) { return key === 'content' ? this.content : null; },
  };
  const document = {
    documentElement: htmlElement,
    querySelector(selector) { return selector === 'meta[name="theme-color"]' ? meta : null; },
  };
  const window = {
    document,
    setTimeout() { return 0; },
    matchMedia() {
      return {
        matches: colorScheme === 'light',
        addEventListener(event, listener) { if (event === 'change') listeners.push(listener); },
      };
    },
  };
  window.window = window;
  const context = vm.createContext({window, document, localStorage: storage, console});
  if (runPrepaint) vm.runInContext(prepaint, context, {filename: 'ui/index.html#prepaint'});
  const prepaintState = {theme: htmlElement.getAttribute('data-theme'), bg: meta.content};
  vm.runInContext(runtime, context, {filename: 'theme.ts#bundle'});

  return {
    storage,
    window,
    prepaintState,
    runtimeState() {
      const prefs = JSON.parse(JSON.stringify(window.themeApi.getStoredPrefs()));
      return {prefs, theme: window.themeApi.resolveTheme(prefs)};
    },
    setColorScheme(nextScheme) {
      colorScheme = nextScheme;
      for (const listener of listeners) listener({matches: nextScheme === 'light'});
    },
    domState() { return {theme: htmlElement.getAttribute('data-theme'), bg: meta.content}; },
  };
}

const graphiteDark = {theme: 'graphite-studio', bg: '#111316'};
const migratedDefault = {mode: 'dark', family: 'graphite-studio'};

// Existing explicit families, auto/light modes, legacy variants, interim keys,
// and malformed values all receive the one-time Graphite Studio dark reset.
const oldPreferences = [
  {[KEYS.mode]: 'light', [KEYS.family]: 'default', [KEYS.legacy]: 'daylight'},
  {[KEYS.mode]: 'auto', [KEYS.family]: 'onyx', [KEYS.legacy]: 'pearl'},
  {[KEYS.legacy]: 'ivory'},
  {[KEYS.interim]: 'golden-hour'},
  {[KEYS.mode]: 'sepia', [KEYS.family]: 'not-a-family', [KEYS.interim]: 'bad', [KEYS.legacy]: 'not-a-theme'},
];
for (const values of oldPreferences) {
  const page = openPage({storage: new MemoryStorage(values), colorScheme: 'light'});
  assert.deepEqual(page.prepaintState, graphiteDark, 'unmigrated prepaint ignores every old selection');
  assert.deepEqual(page.runtimeState(), {prefs: migratedDefault, theme: 'graphite-studio'});
  assert.equal(page.storage.getItem(KEYS.mode), 'dark');
  assert.equal(page.storage.getItem(KEYS.family), 'graphite-studio');
  assert.equal(page.storage.getItem(KEYS.legacy), 'graphite-studio');
  assert.equal(page.storage.getItem(KEYS.marker), '4');
}

// Runtime can complete the same one-time reset if the inline script could not
// execute before the app bundle (for example in an embedded webview).
{
  const storage = new MemoryStorage({[KEYS.mode]: 'auto', [KEYS.family]: 'onyx'});
  const page = openPage({storage, runPrepaint: false, colorScheme: 'light'});
  assert.deepEqual(page.runtimeState(), {prefs: migratedDefault, theme: 'graphite-studio'});
  assert.equal(storage.getItem(KEYS.marker), '4');
  assert.equal(storage.getItem(KEYS.mode), 'dark');
  assert.equal(storage.getItem(KEYS.family), 'graphite-studio');
}

// A marker fixed to the v4 major version preserves a user's saved selection
// on later patch loads. The old legacy fallback still works after migration.
{
  const storage = new MemoryStorage({
    [KEYS.marker]: '4', [KEYS.mode]: 'light', [KEYS.family]: 'onyx', [KEYS.legacy]: 'pearl',
  });
  const page = openPage({storage});
  assert.deepEqual(page.prepaintState, {theme: 'pearl', bg: '#f2f2f2'});
  assert.deepEqual(page.runtimeState(), {prefs: {mode: 'light', family: 'onyx'}, theme: 'pearl'});
}
{
  const storage = new MemoryStorage({[KEYS.marker]: '4', [KEYS.legacy]: 'daylight'});
  const page = openPage({storage});
  assert.deepEqual(page.prepaintState, {theme: 'daylight', bg: '#f4f5f7'});
  assert.deepEqual(page.runtimeState(), {prefs: {mode: 'light', family: 'default'}, theme: 'daylight'});
}
{
  const storage = new MemoryStorage({[KEYS.marker]: '4', [KEYS.interim]: 'onyx'});
  const page = openPage({storage});
  assert.deepEqual(page.prepaintState, {theme: 'onyx', bg: '#000000'});
  assert.deepEqual(page.runtimeState(), {prefs: {mode: 'dark', family: 'onyx'}, theme: 'onyx'});
}
for (const legacy of ['__proto__', 'constructor']) {
  const storage = new MemoryStorage({
    [KEYS.marker]: '4', [KEYS.mode]: 'invalid', [KEYS.family]: 'invalid', [KEYS.legacy]: legacy,
  });
  const page = openPage({storage});
  assert.deepEqual(page.prepaintState, graphiteDark, `${legacy} is not a valid legacy theme id`);
  assert.deepEqual(page.runtimeState(), {prefs: migratedDefault, theme: 'graphite-studio'});
}

// User changes after migration survive a reload representing a later v4 patch;
// auto mode then continues to follow the OS while retaining its chosen family.
{
  const storage = new MemoryStorage({[KEYS.family]: 'default', [KEYS.mode]: 'auto'});
  const firstPage = openPage({storage, colorScheme: 'dark'});
  assert.deepEqual(firstPage.prepaintState, graphiteDark);
  firstPage.window.themeApi.applyThemePrefs({mode: 'auto', family: 'onyx'});
  assert.equal(storage.getItem(KEYS.marker), '4');

  const laterPatchPage = openPage({storage, colorScheme: 'dark'});
  assert.deepEqual(laterPatchPage.prepaintState, {theme: 'onyx', bg: '#000000'});
  assert.deepEqual(laterPatchPage.runtimeState(), {prefs: {mode: 'auto', family: 'onyx'}, theme: 'onyx'});
  laterPatchPage.window.themeApi.applyThemePrefs(laterPatchPage.runtimeState().prefs);
  laterPatchPage.setColorScheme('light');
  assert.deepEqual(laterPatchPage.domState(), {theme: 'pearl', bg: '#f2f2f2'});
  assert.deepEqual(laterPatchPage.runtimeState(), {prefs: {mode: 'auto', family: 'onyx'}, theme: 'pearl'});
  assert.equal(storage.getItem(KEYS.marker), '4');
}

// Blocked storage and partial writes never crash theme setup or mark a partial
// first-load reset complete. The prepaint/runtime both stay on the dark default.
{
  const blocked = {
    getItem() { throw new Error('storage blocked'); },
    setItem() { throw new Error('storage blocked'); },
  };
  const page = openPage({storage: blocked, colorScheme: 'light'});
  assert.deepEqual(page.prepaintState, graphiteDark);
  assert.deepEqual(page.runtimeState(), {prefs: migratedDefault, theme: 'graphite-studio'});
  assert.doesNotThrow(() => page.window.themeApi.applyThemePrefs(migratedDefault));
  assert.deepEqual(page.domState(), graphiteDark);
}
for (const failedKey of [KEYS.family, KEYS.marker]) {
  const storage = new MemoryStorage({[KEYS.legacy]: 'pearl'}, {failSetKey: failedKey});
  const page = openPage({storage, colorScheme: 'light'});
  assert.deepEqual(page.prepaintState, graphiteDark);
  assert.deepEqual(page.runtimeState(), {prefs: migratedDefault, theme: 'graphite-studio'});
  assert.doesNotThrow(() => page.window.themeApi.applyThemePrefs(migratedDefault));
  assert.equal(storage.getItem(KEYS.marker), null, `${failedKey} failure leaves the migration marker absent`);
  if (failedKey === KEYS.family) {
    assert.equal(storage.writes.includes(KEYS.marker), false, 'marker write is skipped after an earlier theme write fails');
  } else {
    const markerAttempt = storage.writes.lastIndexOf(KEYS.marker);
    assert.ok(markerAttempt > storage.writes.lastIndexOf(KEYS.legacy), 'marker is attempted only after all theme keys');
  }
  assert.deepEqual(page.domState(), graphiteDark);
}

console.log('v4 theme migration, prepaint/runtime parity, persistence, legacy compatibility and partial storage passed.');
