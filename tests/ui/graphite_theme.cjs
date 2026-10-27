// Real theme registry, preference migration and prepaint script; network is mocked.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const htmlSource = fs.readFileSync(path.join(root, 'ui/index.html'), 'utf8');
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'ui/public/manifest.webmanifest'), 'utf8'));
  assert.equal(manifest.background_color, '#111316', 'Home-screen launch uses the new default canvas');
  assert.equal(manifest.theme_color, '#111316');
  const prepaint = htmlSource.match(/<div id="root"><\/div>\s*<script>([\s\S]*?)<\/script>/)?.[1];
  assert.ok(prepaint, 'index.html keeps its pre-mount theme script next to the app root');
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {contents: `
      import {FAMILIES, getStoredPrefs, resolveTheme, applyThemePrefs, onOsThemeChange} from './src/lib/theme';
      window.themeApi = {FAMILIES, getStoredPrefs, resolveTheme, applyThemePrefs, onOsThemeChange};
    `, resolveDir: path.join(root, 'ui'), loader: 'ts'},
    bundle: true, write: false, format: 'iife', platform: 'browser', logLevel: 'silent',
  });
  const browser = await chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
    args: ['--disable-gpu'],
  });
  const expectedTheme = (family, variant) => ({
    'graphite-studio': {dark: 'graphite-studio', light: 'graphite-daylight'},
    'golden-hour': {dark: 'golden-hour', light: 'ivory'},
    default: {dark: 'default', light: 'daylight'},
    onyx: {dark: 'onyx', light: 'pearl'},
  })[family][variant];
  const appearance = (family, variant) => {
    const theme = expectedTheme(family, variant);
    const htmlTheme = theme === 'default' ? null : theme;
    const familyInfo = {
      'graphite-studio': {dark: '#111316', light: '#f3f4f6'},
      'golden-hour': {dark: '#0a0a0a', light: '#f2ede2'},
      default: {dark: '#0a0a0f', light: '#f4f5f7'},
      onyx: {dark: '#000000', light: '#f2f2f2'},
    }[family];
    return {theme, htmlTheme, bg: familyInfo[variant]};
  };
  const prepaintAppearance = (family, variant) => {
    const {htmlTheme, bg} = appearance(family, variant);
    return {theme: htmlTheme, bg};
  };
  async function boot({values = {}, blocked = false, colorScheme = 'dark'} = {}) {
    const context = await browser.newContext({colorScheme});
    const page = await context.newPage();
    await page.route('**/*', route => route.fulfill({contentType: 'text/html', body:
      `<meta name="theme-color" content="#111316"><div id="root"></div><script>${prepaint}</script>`}));
    await page.addInitScript(({values, blocked}) => {
      if (blocked) {
        Object.defineProperty(window, 'localStorage', {configurable: true, get() { throw new Error('storage blocked'); }});
        return;
      }
      for (const [key, value] of Object.entries(values)) localStorage.setItem(key, value);
    }, {values, blocked});
    await page.goto('http://graphite-theme.test/');
    const prepaintState = await page.evaluate(() => ({
      theme: document.documentElement.getAttribute('data-theme'),
      bg: document.querySelector('meta[name="theme-color"]').content,
    }));
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    return {context, page, prepaintState};
  }
  const close = async ({context}) => context.close();

  try {
    // The new family is the picker/default family. Earlier family IDs stay valid,
    // and each former light CSS ID still migrates to its original family + light mode.
    {
      const {context, page, prepaintState} = await boot();
      const result = await page.evaluate(() => ({families: window.themeApi.FAMILIES.map(f => f.id), prefs: window.themeApi.getStoredPrefs()}));
      assert.deepEqual(result.families, ['graphite-studio', 'golden-hour', 'default', 'onyx']);
      assert.deepEqual(result.prefs, {mode: 'dark', family: 'graphite-studio'});
      assert.deepEqual(prepaintState, {theme: 'graphite-studio', bg: '#111316'});
      await close({context});
    }
    for (const family of ['golden-hour', 'default', 'onyx']) {
      const {context, page} = await boot({values: {'maestro-theme-family': family}});
      assert.deepEqual(await page.evaluate(() => window.themeApi.getStoredPrefs()), {mode: 'dark', family});
      await close({context});
    }
    for (const [legacy, family] of [['graphite-daylight', 'graphite-studio'], ['ivory', 'golden-hour'], ['daylight', 'default'], ['pearl', 'onyx']]) {
      const {context, page, prepaintState} = await boot({values: {'maestro-theme': legacy}});
      const prefs = await page.evaluate(() => window.themeApi.getStoredPrefs());
      assert.deepEqual(prefs, {mode: 'light', family});
      assert.deepEqual(prepaintState, prepaintAppearance(family, 'light'));
      await close({context});
    }
    {
      const {context, page, prepaintState} = await boot({values: {
        'maestro-theme-mode': 'sepia', 'maestro-theme-family': 'not-a-family',
        'maestro-theme-dark': 'bad', 'maestro-theme': 'not-a-theme',
      }});
      assert.deepEqual(await page.evaluate(() => window.themeApi.getStoredPrefs()), {mode: 'dark', family: 'graphite-studio'});
      assert.deepEqual(prepaintState, prepaintAppearance('graphite-studio', 'dark'));
      await close({context});
    }
    {
      const {context, page, prepaintState} = await boot({blocked: true, colorScheme: 'light'});
      assert.deepEqual(await page.evaluate(() => window.themeApi.getStoredPrefs()), {mode: 'dark', family: 'graphite-studio'});
      assert.deepEqual(prepaintState, prepaintAppearance('graphite-studio', 'dark'), 'blocked storage uses the graphite dark prepaint default');
      await page.evaluate(() => window.themeApi.applyThemePrefs(window.themeApi.getStoredPrefs()));
      assert.equal(await page.locator('meta[name="theme-color"]').getAttribute('content'), '#111316');
      await close({context});
    }

    // Exercise all explicit modes against both OS schemes. The actual document
    // theme and browser chrome tint must match both the runtime resolver and prepaint.
    for (const family of ['graphite-studio', 'golden-hour', 'default', 'onyx']) {
      for (const mode of ['dark', 'light', 'auto']) {
        for (const colorScheme of mode === 'auto' ? ['dark', 'light'] : ['dark']) {
          const {context, page, prepaintState} = await boot({
            values: {'maestro-theme-mode': mode, 'maestro-theme-family': family}, colorScheme,
          });
          const variant = mode === 'light' || (mode === 'auto' && colorScheme === 'light') ? 'light' : 'dark';
          const expected = appearance(family, variant);
          assert.deepEqual(prepaintState, prepaintAppearance(family, variant), `${family}/${mode}/${colorScheme} prepaint`);
          const runtime = await page.evaluate(() => {
            const prefs = window.themeApi.getStoredPrefs();
            return {prefs, theme: window.themeApi.resolveTheme(prefs)};
          });
          assert.deepEqual(runtime.prefs, {mode, family});
          assert.equal(runtime.theme, expected.theme);
          await page.evaluate(() => window.themeApi.applyThemePrefs(window.themeApi.getStoredPrefs()));
          assert.equal(await page.locator('meta[name="theme-color"]').getAttribute('content'), expected.bg);
          assert.deepEqual(await page.evaluate(() => ({
            mode: localStorage.getItem('maestro-theme-mode'),
            family: localStorage.getItem('maestro-theme-family'),
            legacy: localStorage.getItem('maestro-theme'),
          })), {mode, family, legacy: expected.theme}, `${family}/${mode} persists the resolved legacy theme`);
          await close({context});
        }
      }
    }

    // Auto mode responds to OS changes for both directions, updating the DOM,
    // browser tint and subscriber while retaining the selected family.
    for (const family of ['graphite-studio', 'golden-hour', 'default', 'onyx']) {
      const {context, page} = await boot({values: {'maestro-theme-mode': 'auto', 'maestro-theme-family': family}, colorScheme: 'dark'});
      await page.evaluate(() => {
        window.themeChanges = 0;
        window.themeApi.onOsThemeChange(() => window.themeChanges++);
        window.themeApi.applyThemePrefs(window.themeApi.getStoredPrefs());
      });
      await page.emulateMedia({colorScheme: 'light'});
      await page.waitForFunction(theme => document.documentElement.getAttribute('data-theme') === theme,
        expectedTheme(family, 'light') === 'default' ? null : expectedTheme(family, 'light'));
      assert.equal(await page.locator('meta[name="theme-color"]').getAttribute('content'), appearance(family, 'light').bg);
      assert.equal(await page.evaluate(() => window.themeChanges), 1);
      await page.emulateMedia({colorScheme: 'dark'});
      await page.waitForFunction(theme => document.documentElement.getAttribute('data-theme') === theme,
        expectedTheme(family, 'dark') === 'default' ? null : expectedTheme(family, 'dark'));
      assert.equal(await page.locator('meta[name="theme-color"]').getAttribute('content'), appearance(family, 'dark').bg);
      assert.equal(await page.evaluate(() => window.themeChanges), 2);
      await close({context});
    }

    // Check the emitted CSS against real fill/foreground utility combinations.
    // Bright Graphite actions need dark text; playback whites stay independent.
    const assets = path.join(root, 'ui/dist/assets');
    const stylesheet = fs.readFileSync(path.join(assets,
      fs.readdirSync(assets).find(name => /^index-.*\.css$/.test(name))), 'utf8');
    const luminance = color => {
      const rgb = color.startsWith('#')
        ? color.slice(1).match(/../g).map(value => parseInt(value, 16))
        : color.match(/[\d.]+/g).slice(0, 3).map(Number);
      return rgb.map(value => value / 255).reduce((sum, value, index) => sum
        + (value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4)
        * [.2126, .7152, .0722][index], 0);
    };
    for (const mode of ['dark', 'light']) {
      const {context, page} = await boot({values: {
        'maestro-theme-family': 'graphite-studio', 'maestro-theme-mode': mode,
      }});
      await page.addStyleTag({content: stylesheet + '*{transition:none!important}'});
      const colors = await page.evaluate(() => {
        window.themeApi.applyThemePrefs(window.themeApi.getStoredPrefs());
        const root = getComputedStyle(document.documentElement);
        const probes = document.createElement('div');
        probes.innerHTML = '<button class="bg-cta text-white">Generate</button><button class="bg-toggle-active text-white">Studio</button><button class="bg-accent-blue text-white">Action</button><button class="bg-accent-green text-white">Success</button><span class="bg-black/40 text-white">Playback</span>';
        document.body.append(probes);
        const foregrounds = [...probes.children].map(element => getComputedStyle(element).color);
        probes.remove();
        return {foregrounds, backgrounds: ['--color-gradient-cta-from', '--color-gradient-toggle-from',
          '--color-accent-blue', '--color-accent-green'].map(name => root.getPropertyValue(name).trim()),
          muted: root.getPropertyValue('--color-text-muted').trim(),
          active: root.getPropertyValue('--color-bg-active').trim()};
      });
      const pairings = colors.backgrounds.map((background, index) => [colors.foregrounds[index], background]);
      pairings.push([colors.muted, colors.active]);
      for (const [foreground, background] of pairings) {
        const a = luminance(foreground), b = luminance(background);
        assert.ok((Math.max(a, b) + .05) / (Math.min(a, b) + .05) >= 4.5,
          `${mode} readable foreground ${foreground} against ${background}`);
      }
      assert.equal(colors.foregrounds[4], 'rgb(255, 255, 255)', 'Playback overlay stays white');
      await page.evaluate(() => window.themeApi.applyThemePrefs({family: 'golden-hour', mode: 'dark'}));
      const retained = await page.evaluate(() => {
        const probe = document.createElement('button');probe.className = 'bg-accent-blue text-white';
        document.body.append(probe);const color = getComputedStyle(probe).color;probe.remove();return color;
      });
      assert.equal(retained, 'rgb(255, 255, 255)', 'Existing theme foregrounds are unchanged');
      await close({context});
    }

    const graphite = await (async () => {
      const {context, page} = await boot();
      const result = await page.evaluate(() => {
        const family = window.themeApi.FAMILIES[0];
        return {id: family.id, dark: family.dark.id, light: family.light.id,
          darkSwatch: family.dark.swatch, lightSwatch: family.light.swatch,
          goldenHourDescription: window.themeApi.FAMILIES.find(f => f.id === 'golden-hour').description};
      });
      await close({context});
      return result;
    })();
    assert.deepEqual(graphite, {
      id: 'graphite-studio', dark: 'graphite-studio', light: 'graphite-daylight',
      darkSwatch: {bg: '#111316', surface: '#191c20', accent: '#ff985f'},
      lightSwatch: {bg: '#f3f4f6', surface: '#fafbfc', accent: '#b34312'},
      goldenHourDescription: 'Warm cinematic palette — near-black surfaces and amber highlights at night; warm paper and burnt orange in daylight.',
    });
    console.log('Graphite Studio preference defaults, legacy migrations, prepaint/runtime parity, modes, OS changes, persistence and browser tint passed.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
