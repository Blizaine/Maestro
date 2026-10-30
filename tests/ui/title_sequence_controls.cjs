// Isolated browser regression for the shared animated-title controls.
// It mounts the real control and helper module; no generation endpoint is used.
// Run after the UI build: node tests/ui/title_sequence_controls.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');

const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT
  || (() => {
    try {
      return require.resolve('playwright', {paths: [path.join(root, 'ui')]});
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
      return require.resolve('playwright');
    }
  })();
const {chromium} = require(playwrightPath);

function appCss() {
  const html = fs.readFileSync(path.join(ui, 'dist/index.html'), 'utf8');
  const filename = /href="\/assets\/([^"]+\.css)"/.exec(html)?.[1];
  assert.ok(filename, 'Build the UI before running the title controls browser check.');
  const file = path.join(ui, 'dist/assets', filename);
  assert.ok(fs.existsSync(file), `Built stylesheet exists: ${file}`);
  return fs.readFileSync(file, 'utf8');
}

async function bundleControls() {
  const extensions = ['.tsx', '.ts', '.jsx', '.js', '.mjs', '.cjs', '.json', '.svg'];
  const resolveFile = candidate => {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return fs.realpathSync(candidate);
    for (const extension of extensions) {
      const withExtension = candidate + extension;
      if (fs.existsSync(withExtension) && fs.statSync(withExtension).isFile()) return fs.realpathSync(withExtension);
      const indexFile = path.join(candidate, `index${extension}`);
      if (fs.existsSync(indexFile) && fs.statSync(indexFile).isFile()) return fs.realpathSync(indexFile);
    }
    return null;
  };
  const resolver = {
    name: 'title-sequence-controls-workspace-resolver',
    setup(build) {
      build.onResolve({filter: /.*/}, args => {
        if (args.path.startsWith('node:')) return {path: args.path, external: true};
        if (args.path.startsWith('.') || path.isAbsolute(args.path)) {
          const resolved = resolveFile(path.resolve(args.resolveDir, args.path));
          return resolved ? {path: resolved, namespace: 'workspace-fs'}
            : {errors: [{text: `Unable to resolve ${args.path} from ${args.resolveDir}`}]};
        }
        return {path: fs.realpathSync(requireFromUi.resolve(args.path, {paths: [args.resolveDir, ui]})), namespace: 'workspace-fs'};
      });
      build.onLoad({filter: /.*/, namespace: 'workspace-fs'}, args => {
        const extension = path.extname(args.path);
        const loader = extension === '.tsx' ? 'tsx' : extension === '.ts' ? 'ts'
          : extension === '.jsx' ? 'jsx' : extension === '.json' ? 'json'
            : extension === '.svg' ? 'file' : 'js';
        return {contents: fs.readFileSync(args.path, 'utf8'), loader, resolveDir: path.dirname(args.path)};
      });
    },
  };
  const result = await esbuild.build({
    stdin: {
      contents: `
        import {useState} from 'react';
        import {createRoot} from 'react-dom/client';
        import {AnimatedTitleAdvancedControls} from './src/editor/AnimatedTitleAdvancedControls';
        import {DEFAULT_TITLE_SETTINGS, TITLE_EFFECTS} from './src/editor/editorAnimatedTitles';
        function Harness() {
          const [settings, setSettings] = useState({
            ...DEFAULT_TITLE_SETTINGS,
            text: 'CREATE\\nEXPLORE',
            effect: 'fire',
            direction: TITLE_EFFECTS.fire,
            seconds: 5.2,
          });
          const [busy, setBusy] = useState(false);
          const patch = value => setSettings(current => ({...current, ...value}));
          window.titleHarness = {getSettings: () => settings, getBusy: () => busy, setBusy, setSettings};
          return <main className="mx-auto flex w-full max-w-2xl flex-col gap-4 p-4">
            <AnimatedTitleAdvancedControls settings={settings} patch={patch} busy={busy}
              canvas={{width: 1920, height: 1080}} destination="editor" at={3.25} />
          </main>;
        }
        createRoot(document.getElementById('root')).render(<Harness />);
      `,
      resolveDir: ui,
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    format: 'iife',
    target: 'es2022',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    plugins: [resolver],
    logLevel: 'silent',
  });
  return result.outputFiles[0].text;
}

async function main() {
  const bundle = await bundleControls();
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32'
    ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  const page = await browser.newPage({viewport: {width: 1280, height: 900}});
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const screenshots = process.env.MAESTRO_TITLE_SEQUENCE_SCREENSHOTS;
  if (screenshots) fs.mkdirSync(screenshots, {recursive: true});

  try {
    await page.setContent(`<!doctype html><html data-theme="graphite-studio"><head><meta charset="utf-8"><style>${appCss()}</style></head><body><div id="root"></div><script>${bundle}</script></body></html>`);
    const singleText = page.getByLabel('Animated title text');
    await singleText.waitFor();
    assert.equal(await singleText.inputValue(), 'CREATE\nEXPLORE', 'Legacy title intents start in single-title mode.');
    assert.equal(await page.getByRole('button', {name: 'Single title'}).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByLabel('Title duration').inputValue(), '5.2');
    assert.equal(await page.getByLabel('Title animation preset').locator('option').count(), 21,
      'The legacy single-title selector keeps its custom choice and 20 presets.');
    const legacyDirection = await page.evaluate(() => window.titleHarness.getSettings().direction);
    assert.equal(await page.getByLabel('Title animation description').inputValue(), legacyDirection);

    await page.getByRole('button', {name: 'Phrase sequence'}).click();
    const sequenceDuration = page.getByLabel('Sequence duration');
    assert.equal(await sequenceDuration.locator('option').count(), 14,
      'Two phrases allow all 14 H3 native whole-clip durations.');
    assert.equal(await sequenceDuration.inputValue(), '124', 'The single-title duration selects its nearest native sequence clip.');
    assert.equal(await page.getByText('Sequence length', {exact: true}).count(), 0,
      'The native duration picker replaces the former read-only total field.');
    let initialSequence = await page.evaluate(() => window.titleHarness.getSettings());
    assert.equal(initialSequence.sequenceFrames, 124);
    assert.ok(initialSequence.phrases.every(phrase => Number.isInteger(phrase.seconds * 24)),
      'Aligned phrase slots use integer frame counts.');
    assert.equal(await page.getByRole('group', {name: 'Phrase 1'}).getByLabel('Phrase 1 text').inputValue(), 'CREATE',
      'Multiline single-title text converts into ordered phrase cards.');
    assert.equal(await page.getByRole('group', {name: 'Phrase 2'}).getByLabel('Phrase 2 text').inputValue(), 'EXPLORE');
    assert.equal(await page.getByLabel('Shared title direction').inputValue(), '',
      'Single-title fire motion does not leak into the phrase sequence.');
    const retainedSingleSettings = await page.evaluate(() => window.titleHarness.getSettings());
    assert.equal(retainedSingleSettings.text, 'CREATE\nEXPLORE', 'Sequence editing keeps the single-title draft intact.');
    assert.equal(retainedSingleSettings.direction, legacyDirection, 'Switching modes keeps the single-title direction in settings.');
    assert.equal(await page.getByText('Each phrase clears before the next one; the last clears to the selected green background.').count(), 1);
    assert.equal(await page.getByText('14/160 total text characters').count(), 1);
    assert.match(await page.getByRole('group', {name: 'Phrase 1'}).innerText(), /simple fade in, readable hold, and complete fade out/,
      'Blank per-phrase motion has a clear fade default.');

    let firstCard = page.getByRole('group', {name: 'Phrase 1'});
    await firstCard.locator('details > summary').click();
    await firstCard.getByLabel('Phrase 1 animation preset').selectOption('slide');
    assert.match(await firstCard.getByRole('textbox', {name: 'Phrase 1 animation'}).inputValue(), /Slide this phrase in/);
    await firstCard.getByRole('textbox', {name: 'Phrase 1 animation'}).fill('A custom phrase motion, then clear every letter.');
    assert.equal(await firstCard.getByLabel('Phrase 1 animation preset').inputValue(), 'custom');
    await firstCard.getByRole('textbox', {name: 'Phrase 1 animation'}).fill('');

    await firstCard.getByLabel('Phrase 1 duration').fill('3.1');
    await firstCard.getByLabel('Phrase 1 duration').blur();
    await firstCard.getByLabel('Phrase 1 text').fill('CREATE');
    await page.getByRole('group', {name: 'Phrase 2'}).getByLabel('Phrase 2 text').fill('EXPLORE');
    assert.match(await page.getByLabel('Approximate sequence timeline').innerText(), /Selected clip: 124 frames · 5\.167s at 24 fps/);
    assert.match(await page.getByLabel('Approximate sequence timeline').innerText(), /0\.000–3\.083s · 74 frames/);
    let editedSettings = await page.evaluate(() => window.titleHarness.getSettings());
    assert.equal(editedSettings.sequenceFrames, 124, 'Editing one phrase keeps the selected clip length.');
    assert.equal(Math.round(editedSettings.phrases.reduce((sum, phrase) => sum + phrase.seconds, 0) * 24), 124);
    await page.getByText('View prepared prompt').click();
    const sequencePrompt = await page.locator('pre').innerText();
    assert.match(sequencePrompt, /Phrase schedule for this/);
    assert.match(sequencePrompt, /The ONLY readable text is "CREATE"/);
    assert.match(sequencePrompt, /fade it out completely/);

    await firstCard.getByLabel('Phrase 1 duration').fill('0');
    await firstCard.getByLabel('Phrase 1 duration').blur();
    editedSettings = await page.evaluate(() => window.titleHarness.getSettings());
    assert.ok(editedSettings.phrases[0].seconds >= 1, 'Blur clamps an edited slot to its one-second frame minimum.');
    assert.equal(editedSettings.sequenceFrames, 124, 'A bounded phrase edit preserves the native clip length.');
    await firstCard.getByLabel('Phrase 1 duration').fill('3.1');
    await firstCard.getByLabel('Phrase 1 duration').blur();

    await firstCard.getByLabel('Phrase 1 text').fill('A'.repeat(80));
    await page.getByRole('group', {name: 'Phrase 2'}).getByLabel('Phrase 2 text').fill('B'.repeat(80));
    assert.equal(await page.getByText('80/80 characters · limit reached').count(), 2,
      'Phrase length counters expose the per-phrase limit.');
    assert.ok(await page.getByRole('alert').count() >= 1, 'The combined sequence character warning is visible.');
    await firstCard.getByLabel('Phrase 1 text').fill('CREATE');
    await page.getByRole('group', {name: 'Phrase 2'}).getByLabel('Phrase 2 text').fill('EXPLORE');

    await page.getByRole('button', {name: 'Move phrase 1 down'}).click();
    let currentPhrases = await page.evaluate(() => window.titleHarness.getSettings().phrases.map(phrase => phrase.text));
    assert.deepEqual(currentPhrases, ['EXPLORE', 'CREATE'], 'Up/down controls reorder the phrases.');
    assert.equal((await page.evaluate(() => window.titleHarness.getSettings())).sequenceFrames, 124,
      'Reordering keeps each phrase slot and the selected native clip length.');

    await page.getByRole('button', {name: '+ Add phrase'}).click();
    assert.equal(await page.getByTestId('title-phrase-card').count(), 3, 'Add phrase creates a third card.');
    await page.getByRole('group', {name: 'Phrase 3'}).getByLabel('Phrase 3 text').fill('INSPIRE');
    await page.getByRole('button', {name: 'Remove phrase 2'}).click();
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings().phrases.map(phrase => phrase.text)),
      ['EXPLORE', 'INSPIRE'], 'Remove deletes the selected card and preserves the remaining order.');
    assert.equal(await page.getByRole('button', {name: 'Remove phrase 1'}).isDisabled(), true);
    assert.equal(await page.getByRole('button', {name: 'Remove phrase 2'}).isDisabled(), true,
      'A two-phrase sequence cannot be reduced below its minimum.');

    await page.getByText('Paste lines to make phrases').click();
    const paste = page.getByLabel('Paste phrases, one per line');
    await paste.fill('CREATE\nEXPLORE\nINSPIRE');
    await page.getByRole('button', {name: 'Replace phrases from lines'}).click();
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings().phrases.map(phrase => phrase.text)),
      ['CREATE', 'EXPLORE', 'INSPIRE'], 'The explicit paste-lines action replaces cards in order.');
    await paste.fill('ONE\nTWO\nTHREE\nFOUR\nFIVE\nSIX\nSEVEN');
    await page.getByRole('button', {name: 'Replace phrases from lines'}).click();
    assert.match(await page.getByRole('alert').last().innerText(), /6|phrase/i,
      'More than six pasted lines show a visible error and do not replace the current cards.');
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings().phrases.map(phrase => phrase.text)),
      ['CREATE', 'EXPLORE', 'INSPIRE']);

    await paste.fill('CREATE\nEXPLORE\nINSPIRE');
    await page.getByRole('button', {name: 'Replace phrases from lines'}).click();
    while (await page.getByTestId('title-phrase-card').count() < 6) await page.getByRole('button', {name: '+ Add phrase'}).click();
    assert.equal(await page.getByTestId('title-phrase-card').count(), 6);
    assert.equal(await sequenceDuration.locator('option').count(), 12,
      'Six phrases filter native durations below the 144-frame minimum.');
    assert.ok(Number(await sequenceDuration.inputValue()) >= 144,
      'Adding phrases selects a native clip long enough for one second per card.');
    assert.equal(await page.getByRole('button', {name: '+ Add phrase'}).count(), 0,
      'The sequence controls cap the number of cards at six.');

    await page.evaluate(() => window.titleHarness.setBusy(true));
    await page.getByRole('button', {name: 'Phrase sequence'}).waitFor();
    assert.equal(await page.getByRole('button', {name: 'Single title'}).isDisabled(), true, 'Mode changes are blocked while busy.');
    assert.equal(await page.getByLabel('Phrase 1 text').isDisabled(), true, 'Phrase editing is blocked while busy.');
    assert.equal(await page.getByRole('button', {name: 'Remove phrase 1'}).isDisabled(), true, 'Card actions are blocked while busy.');
    await page.evaluate(() => window.titleHarness.setBusy(false));

    await page.getByRole('button', {name: 'Replace phrases from lines'}).click();
    assert.equal(await page.getByTestId('title-phrase-card').count(), 3,
      'The visual review uses three valid phrase cards after checking the six-card limit.');
    await sequenceDuration.selectOption('243');
    assert.equal(await sequenceDuration.inputValue(), '243');
    assert.match(await page.getByLabel('Approximate sequence timeline').innerText(), /Selected clip: 243 frames · 10\.125s at 24 fps/);
    await page.getByLabel('Phrase 1 duration').fill('3.333');
    await page.getByLabel('Phrase 1 duration').blur();
    const resizedSettings = await page.evaluate(() => window.titleHarness.getSettings());
    assert.equal(resizedSettings.sequenceFrames, 243, 'Changing a phrase slot preserves the chosen 243-frame clip.');
    assert.equal(Math.round(resizedSettings.phrases.reduce((sum, phrase) => sum + phrase.seconds, 0) * 24), 243,
      'All resized phrase slots still sum to the selected clip frames.');
    assert.ok(resizedSettings.phrases.every(phrase => Number.isInteger(phrase.seconds * 24)),
      'Every redistributed phrase slot remains frame-aligned.');
    assert.equal(Math.round(resizedSettings.phrases[0].seconds * 24), 80,
      'The edited 3.333-second phrase is rounded to its nearest 24fps frame.');

    if (screenshots) {
      await page.screenshot({path: path.join(screenshots, 'title-sequence-desktop.png'), fullPage: true});
      await page.setViewportSize({width: 390, height: 844});
      await page.screenshot({path: path.join(screenshots, 'title-sequence-mobile.png'), fullPage: true});
    } else {
      await page.setViewportSize({width: 390, height: 844});
    }
    const mobileWidth = await page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
      cards: [...document.querySelectorAll('[data-testid="title-phrase-card"]')].map(card => ({
        scroll: card.scrollWidth, client: card.clientWidth,
      })),
    }));
    assert.ok(mobileWidth.document <= mobileWidth.viewport + 1, `No horizontal page overflow on mobile: ${JSON.stringify(mobileWidth)}`);
    assert.ok(mobileWidth.cards.every(card => card.scroll <= card.client + 1), `Phrase cards fit mobile width: ${JSON.stringify(mobileWidth)}`);

    await page.evaluate(() => window.titleHarness.setSettings(current => ({
      ...current,
      mode: 'sequence',
      sequenceFrames: undefined,
      phrases: [
        {id: 'legacy-one', text: 'LEGACY', seconds: 2.6, animation: ''},
        {id: 'legacy-two', text: 'TIMING', seconds: 2.6, animation: ''},
      ],
    })));
    assert.equal(await sequenceDuration.inputValue(), '124', 'Legacy sequences derive the picker value from their phrase total.');
    assert.equal(await page.getByText('Sequence length', {exact: true}).count(), 0,
      'Legacy sequence settings remain valid without the removed read-only total field.');
    await page.getByRole('button', {name: '+ Add phrase'}).click();
    const migratedSettings = await page.evaluate(() => window.titleHarness.getSettings());
    assert.equal(migratedSettings.sequenceFrames, 124, 'The first edit of a legacy sequence stores an explicit native duration.');
    assert.ok(migratedSettings.phrases.every(phrase => Number.isInteger(phrase.seconds * 24)),
      'The first legacy sequence edit aligns all phrase slots to frames.');

    await page.getByRole('button', {name: 'Single title'}).click();
    assert.equal(await page.getByLabel('Animated title text').inputValue(), 'CREATE\nEXPLORE',
      'Returning to single mode restores the original single-title wording.');
    assert.equal(await page.getByLabel('Title animation description').inputValue(),
      await page.evaluate(() => window.titleHarness.getSettings().direction),
      'Returning to single mode restores the original single-title motion.');
    assert.equal(await page.getByLabel('Title duration').inputValue(), '5.2');
    assert.equal(await page.getByLabel('Title animation preset').locator('option').count(), 21,
      'The legacy full-title preset selector remains available in single mode.');

    assert.deepEqual(pageErrors, [], 'The mounted controls have no browser runtime errors.');
    console.log('Phrase sequence controls passed: native duration selection, legacy conversion, frame-aligned editing, ordered editing, presets, paste limits, busy state, six-card limit, and mobile overflow.');
    if (screenshots) console.log(`Screenshots saved in ${screenshots}`);
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
