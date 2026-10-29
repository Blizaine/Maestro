// Browser regression for the unified natural-language title composer and its
// isolated planning lifecycle. All planning/submission requests are synthetic.
// Run after the UI build: node tests/ui/title_composer.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');

const root = path.resolve(__dirname, '../..');
const ui = path.join(root, 'ui');
const requireFromUi = createRequire(path.join(ui, 'package.json'));
const esbuild = requireFromUi('esbuild');
const playwrightPath = process.env.MAESTRO_PLAYWRIGHT ||
  'C:/Users/bliza/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core';
const {chromium} = require(playwrightPath);
const ORIGIN = 'http://maestro.test';

function appCss() {
  const html = fs.readFileSync(path.join(ui, 'dist/index.html'), 'utf8');
  const filename = /href="\/assets\/([^"]+\.css)"/.exec(html)?.[1];
  assert.ok(filename, 'Build the UI before running the title composer browser check.');
  const file = path.join(ui, 'dist/assets', filename);
  assert.ok(fs.existsSync(file), `Built stylesheet exists: ${file}`);
  return fs.readFileSync(file, 'utf8');
}

async function bundleComposer() {
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
    name: 'title-composer-workspace-resolver',
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
        import {AnimatedTitleControls} from './src/editor/AnimatedTitleControls';
        import {DEFAULT_TITLE_SETTINGS, animatedTitlePrompt, titleComposerValidation, titlePlanIsCurrent} from './src/editor/editorAnimatedTitles';
        import {useTitleComposer} from './src/editor/useTitleComposer';
        function Composer({scope}) {
          const [settings, setSettings] = useState({...DEFAULT_TITLE_SETTINGS, brief: ''});
          const [submitted, setSubmitted] = useState([]);
          const [enhanceOnGenerate, setEnhanceOnGenerate] = useState(true);
          const patch = value => setSettings(current => ({...current, ...value}));
          const composer = useTitleComposer(settings, patch, scope);
          const validation = titleComposerValidation(settings, {width: 1920, height: 1080});
          let reviewReady = false;
          try { reviewReady = settings.titleReviewRequired === true && titlePlanIsCurrent(settings); } catch {}
          const submit = async value => setSubmitted(current => [...current, JSON.parse(JSON.stringify(value))]);
          window.titleHarness = {...window.titleHarness, settings, submitted, composer,
            getSettings: () => settings, getSubmitted: () => submitted,
            getSubmittedPrompt: index => animatedTitlePrompt(submitted[index]),
            setEnhanceOnGenerate,
            setSettings: value => setSettings(current => ({...current, ...value}))};
          return <main className="mx-auto flex w-full max-w-2xl flex-col gap-4 p-4">
            <AnimatedTitleControls settings={settings} patch={patch} busy={composer.busy}
              composer={composer} canvas={{width: 1920, height: 1080}} destination="editor" at={3.25} />
            {validation && <p role="alert">{validation}</p>}
            <button type="button" disabled={composer.busy || Boolean(validation)} onClick={() => void composer.generate(submit, {enhance: enhanceOnGenerate})}>
              {reviewReady ? 'Generate reviewed plan' : 'Generate title'}
            </button>
          </main>;
        }
        function App() {
          const [scope, setScope] = useState('project-a:workspace-a:open');
          const [mounted, setMounted] = useState(true);
          window.titleHarness = {...window.titleHarness, setScope, setMounted, getScope: () => scope};
          return mounted ? <Composer scope={scope} /> : <button type="button">Composer unmounted</button>;
        }
        createRoot(document.getElementById('root')).render(<App />);
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

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

async function main() {
  const bundle = await bundleComposer();
  const browser = await chromium.launch({headless: true, ...(process.platform === 'win32'
    ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  const page = await browser.newPage({viewport: {width: 1280, height: 900}});
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const screenshots = process.env.MAESTRO_TITLE_COMPOSER_SCREENSHOTS;
  if (screenshots) fs.mkdirSync(screenshots, {recursive: true});
  const planRequests = [];
  const responseQueue = [];
  let delayedPlan = null;

  await page.route(`${ORIGIN}/**`, async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/v1/llm/plan-title' && route.request().method() === 'POST') {
      planRequests.push(route.request().postDataJSON());
      if (delayedPlan) {
        const active = delayedPlan;
        delayedPlan = null;
        active.started.resolve();
        await active.release.promise;
        return route.fulfill({json: active.response});
      }
      return route.fulfill({json: responseQueue.shift() || {plan: null, warnings: [], raw_draft: '', review_required: false}});
    }
    if (route.request().isNavigationRequest()) {
      return route.fulfill({contentType: 'text/html', body: '<!doctype html><html data-theme="graphite-studio"><head><meta charset="utf-8"></head><body><div id="root"></div></body></html>'});
    }
    return route.fulfill({status: 204, body: ''});
  });

  try {
    await page.goto(`${ORIGIN}/`);
    await page.addStyleTag({content: appCss()});
    await page.addScriptTag({content: bundle});
    const brief = page.getByLabel('Describe your title animation');
    await brief.waitFor();

    assert.equal(await brief.inputValue(), '', 'The composer starts with a blank, user-owned description.');
    assert.equal(await page.getByLabel('Title style preset').locator('option').count(), 21,
      'Custom plus all 20 optional styles are available in the main composer.');
    assert.equal(await page.getByLabel('Title clip duration').locator('option').count(), 14,
      'The title duration uses the 14 whole-frame H3 choices.');
    assert.equal(await page.getByRole('button', {name: 'Single title'}).count(), 0,
      'Single/sequence mode stays in Advanced until the user asks for it.');
    assert.equal(await page.getByRole('button', {name: 'Phrase sequence'}).count(), 0,
      'Phrase sequence mode stays in Advanced until the user asks for it.');
    await brief.fill('Show ORBIT in crisp lettering, then let it dissolve into a blue glow.');
    await page.getByLabel('Title style preset').selectOption('neon');
    await page.getByLabel('Title background').first().selectOption('black');
    await page.getByLabel('Title clip duration').selectOption('158');
    await page.getByLabel('Title aspect ratio').first().selectOption('9:16');
    await page.getByLabel('Title resolution').first().selectOption('480p');

    responseQueue.push({
      plan: {sequence: false, phrases: [{text: 'ORBIT', animation: 'Trace it in, hold clearly, then dissolve away.', weight: 1}],
        look: 'Clear, crisp lettering', direction: 'Trace it in, hold clearly, then dissolve away.'},
      warnings: [], raw_draft: '{"plan":"single title"}', review_required: false, draft_history_id: 'abc123-title-draft',
    });
    await page.getByRole('button', {name: 'Generate title'}).click();
    await page.waitForFunction(() => window.titleHarness.getSubmitted().length === 1);
    let firstPlan = await page.evaluate(() => window.titleHarness.getSubmitted()[0]);
    assert.equal(planRequests.length, 1, 'Generate enhances a stale description once.');
    assert.equal(planRequests[0].brief, 'Show ORBIT in crisp lettering, then let it dissolve into a blue glow.');
    assert.match(planRequests[0].style_hint, /neon strokes/);
    assert.equal(planRequests[0].frames, 158);
    assert.equal(planRequests[0].background, 'black');
    assert.equal(firstPlan.text, 'ORBIT');
    assert.equal(firstPlan.mode, 'single');
    assert.equal(firstPlan.aspect, '9:16');
    assert.equal(firstPlan.resolution, '480p');
    assert.equal(firstPlan.stylePreset, 'neon');
    assert.equal(await page.getByTestId('title-plan-review').isVisible(), true,
      'The planned title appears in an expandable review summary.');
    assert.match(await page.getByTestId('title-plan-review').innerText(), /ORBIT/);
    assert.equal(await page.getByTestId('title-plan-review').evaluate(element => element.open), false,
      'A normal generated plan stays compact until expanded.');

    await brief.fill('Show ORBIT first, clear it completely, then show ARRIVE and fade it away.');
    responseQueue.push({
      plan: {sequence: true, phrases: [
        {text: 'ORBIT', animation: 'Trace in, hold, and clear completely.', weight: 1.25},
        {text: 'ARRIVE', animation: 'Reveal, hold, and fade out.', weight: 1.5},
      ], look: 'Clean white lettering', direction: 'Keep the camera still.'},
      warnings: ['The requested timing is approximate.'], raw_draft: '{"detected":["ORBIT","ARRIVE"]}',
      review_required: true, draft_history_id: 'def456-title-draft',
    });
    await page.getByRole('button', {name: 'Generate title'}).click();
    await page.waitForFunction(() => window.titleHarness.getSettings().titleReviewRequired === true);
    assert.equal(planRequests.length, 2, 'The changed description requests exactly one updated plan.');
    assert.equal(await page.evaluate(() => window.titleHarness.getSubmitted().length), 1,
      'A review-required draft pauses before any GPU submission.');
    const review = page.getByTestId('title-plan-review');
    assert.equal(await review.evaluate(element => element.open), true, 'A review-required plan opens its review details.');
    assert.match(await review.innerText(), /ORBIT.*ARRIVE/s, 'The review shows the detected phrase order.');
    assert.match(await review.innerText(), /0\.000–/);
    assert.match(await page.getByLabel('Title planner warnings').innerText(), /timing is approximate/);
    const rawDraftDetails = page.locator('details').filter({has: page.getByText('Raw planner draft diagnostics', {exact: true})}).first();
    if (!await rawDraftDetails.evaluate(element => element.open)) await rawDraftDetails.locator('summary').click();
    assert.match(await rawDraftDetails.locator('pre').innerText(), /detected/);
    assert.equal(await rawDraftDetails.getByRole('link', {name: 'Open saved draft history'}).getAttribute('href'),
      '/api/v1/llm/drafts/def456-title-draft');
    await page.getByRole('button', {name: 'Generate reviewed plan'}).click();
    await page.waitForFunction(() => window.titleHarness.getSubmitted().length === 2);
    assert.equal(planRequests.length, 2, 'Approving the review uses the saved plan without another planner request.');
    const sequencePlan = await page.evaluate(() => window.titleHarness.getSubmitted()[1]);
    assert.equal(sequencePlan.mode, 'sequence');
    assert.deepEqual(sequencePlan.phrases.map(phrase => phrase.text), ['ORBIT', 'ARRIVE']);
    assert.equal(sequencePlan.sequenceFrames, 158);
    assert.equal(sequencePlan.phrases.reduce((sum, phrase) => sum + Math.round(phrase.seconds * 24), 0), 158,
      'Phrase timings remain aligned to the selected whole-frame clip length.');

    await brief.fill('A new description that is deliberately delayed.');
    const scopeGate = {started: deferred(), release: deferred(), response: {
      plan: {sequence: false, phrases: [{text: 'STALE', animation: 'Fade in and out.', weight: 1}], look: '', direction: 'Fade in and out.'},
      warnings: [], raw_draft: '', review_required: false,
    }};
    delayedPlan = scopeGate;
    await page.getByRole('button', {name: 'Enhance with AI'}).click();
    await scopeGate.started.promise;
    const beforeScopeChange = await page.evaluate(() => window.titleHarness.getSettings());
    await page.evaluate(() => window.titleHarness.setScope('project-b:workspace-b:open'));
    await page.waitForFunction(() => window.titleHarness.getScope() === 'project-b:workspace-b:open');
    scopeGate.release.resolve();
    await page.getByRole('button', {name: 'Enhance with AI'}).waitFor({state: 'visible'});
    await page.waitForFunction(() => !window.titleHarness.composer.busy);
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings()), beforeScopeChange,
      'A plan returned for an old project/workspace scope is ignored.');
    assert.equal(await page.evaluate(() => window.titleHarness.getSubmitted().length), 2,
      'An ignored stale plan cannot submit a title.');

    const malformed = {plan: null, warnings: ['The title wording needs a manual correction.'],
      raw_draft: '{not valid planner output}', review_required: true, draft_history_id: 'bad789-title-draft'};
    responseQueue.push(malformed);
    await page.getByRole('button', {name: 'Enhance with AI'}).click();
    await page.getByRole('alert').filter({hasText: 'manual correction'}).waitFor();
    assert.equal(await brief.inputValue(), 'A new description that is deliberately delayed.',
      'A malformed plan preserves the user description.');
    assert.equal(await page.getByLabel('Title planner warnings').innerText(), malformed.warnings[0]);
    if (!await rawDraftDetails.evaluate(element => element.open)) await rawDraftDetails.locator('summary').click();
    assert.match(await rawDraftDetails.locator('pre').innerText(), /not valid planner output/,
      'The raw malformed draft remains available for review.');
    assert.equal(await rawDraftDetails.getByRole('link', {name: 'Open saved draft history'}).getAttribute('href'),
      '/api/v1/llm/drafts/bad789-title-draft');

    await page.getByText('Advanced title controls', {exact: true}).click();
    const advanced = page.locator('details').filter({hasText: 'Advanced title controls'}).first();
    const beforeManualPlanCount = planRequests.length;
    await advanced.getByLabel('Phrase 1 text').fill('MANUAL SECOND');
    await advanced.getByRole('button', {name: 'Move phrase 1 down'}).click();
    assert.equal(await page.getByRole('button', {name: 'Generate title'}).isDisabled(), true,
      'Advanced edits require explicit manual acceptance or a fresh AI plan before Generate.');
    assert.equal(planRequests.length, beforeManualPlanCount, 'Pending manual edits do not trigger a planner call.');
    await page.getByRole('button', {name: 'Use manual title plan'}).click();
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings().phrases.map(phrase => phrase.text)), ['ARRIVE', 'MANUAL SECOND'],
      'Manual acceptance preserves the manually edited order and wording.');
    assert.equal(await page.evaluate(() => window.titleHarness.getSubmitted().length), 2,
      'Accepting a manual plan does not submit before Generate.');
    await page.getByRole('button', {name: 'Generate title'}).click();
    await page.waitForFunction(() => window.titleHarness.getSubmitted().length === 3);
    assert.equal(planRequests.length, beforeManualPlanCount, 'An accepted manual plan requires no LLM call.');
    assert.equal((await page.evaluate(() => window.titleHarness.getSubmitted().at(-1))).phrases[0].text, 'ARRIVE');

    const validManualSettings = await page.evaluate(() => window.titleHarness.getSettings());
    await page.evaluate(() => window.titleHarness.setSettings({mode: 'single', seconds: 0, sequenceFrames: undefined, plannedKey: undefined}));
    await page.getByRole('alert').filter({hasText: 'Title duration'}).first().waitFor();
    assert.deepEqual(pageErrors, [], 'An invalid Advanced duration shows a validation error without breaking render.');
    await page.evaluate(settings => window.titleHarness.setSettings(settings), validManualSettings);

    if (screenshots) {
      const manualDetails = page.locator('details').filter({has: page.getByText('Advanced title controls', {exact: true})}).first();
      if (await manualDetails.evaluate(element => element.open)) await manualDetails.locator(':scope > summary').click();
      if (await rawDraftDetails.evaluate(element => element.open)) await rawDraftDetails.locator(':scope > summary').click();
      const screenshotReview = page.getByTestId('title-plan-review');
      if (!await screenshotReview.evaluate(element => element.open)) await screenshotReview.locator('summary').click();
      await page.screenshot({path: path.join(screenshots, 'title-composer-desktop.png'), fullPage: true});
      await page.setViewportSize({width: 390, height: 844});
      await page.screenshot({path: path.join(screenshots, 'title-composer-mobile.png'), fullPage: true});
      const mobile = await page.evaluate(() => ({
        viewportWidth: innerWidth,
        rootWidth: document.documentElement.scrollWidth,
        viewportHeight: innerHeight,
        composerWidth: document.querySelector('[aria-label="Describe your title animation"]')?.getBoundingClientRect().width,
      }));
      assert.ok(mobile.rootWidth <= mobile.viewportWidth + 1, `Composer has no mobile horizontal overflow: ${JSON.stringify(mobile)}`);
    }

    const directBrief = 'Keep the exact wording “MAESTRO” in silver lettering, then dissolve into a blue glow.\n'
      + 'Keep the backdrop flat and the lettering legible throughout its changing layout.\n'.repeat(100);
    assert.ok(directBrief.length > 5000);
    assert.equal(await page.getByLabel('Describe your title animation').getAttribute('maxlength'), '16000');
    const plannerRequestsBeforeDirect = planRequests.length;
    await page.getByLabel('Describe your title animation').fill(directBrief);
    await page.evaluate(() => window.titleHarness.setEnhanceOnGenerate(false));
    await page.getByRole('button', {name: 'Generate title'}).click();
    await page.waitForFunction(() => window.titleHarness.getSubmitted().length === 4);
    const directSubmission = await page.evaluate(() => ({
      settings: window.titleHarness.getSubmitted().at(-1),
      prompt: window.titleHarness.getSubmittedPrompt(3),
    }));
    assert.equal(planRequests.length, plannerRequestsBeforeDirect,
      'AI-off title generation sends no title-planner request.');
    assert.equal(directSubmission.settings.directDescription, true,
      'AI-off generation keeps the authored description as its direct title plan.');
    assert.equal(directSubmission.settings.brief, directBrief,
      'The direct title plan preserves the raw description exactly.');
    assert.ok(directSubmission.prompt.includes(`User description:\n${directBrief}`),
      'The H3 title prompt contains the user description verbatim.');
    assert.equal(directSubmission.settings.mode, 'single',
      'AI-off generation does not invent a phrase sequence.');
    await page.evaluate(() => window.titleHarness.setEnhanceOnGenerate(true));

    const unmountGate = {started: deferred(), release: deferred(), response: {
      plan: {sequence: false, phrases: [{text: 'UNMOUNTED', animation: 'Fade.', weight: 1}], look: '', direction: 'Fade.'},
      warnings: [], raw_draft: '', review_required: false,
    }};
    await page.getByLabel('Describe your title animation').fill('Request will finish after unmount.');
    delayedPlan = unmountGate;
    await page.getByRole('button', {name: 'Enhance with AI'}).click();
    await unmountGate.started.promise;
    const beforeUnmount = await page.evaluate(() => window.titleHarness.getSettings());
    await page.evaluate(() => window.titleHarness.setMounted(false));
    await page.getByRole('button', {name: 'Composer unmounted'}).waitFor();
    unmountGate.release.resolve();
    await page.waitForTimeout(50);
    assert.deepEqual(await page.evaluate(() => window.titleHarness.getSettings()), beforeUnmount,
      'An in-flight response cannot apply after the composer unmounts.');
    assert.equal(await page.evaluate(() => window.titleHarness.getSubmitted().length), 4,
      'An in-flight response cannot submit after unmount.');
    assert.deepEqual(pageErrors, [], 'The isolated title composer has no browser runtime errors.');
    console.log(`Unified title composer passed: single/sequence plans, review approval, manual acceptance, stale scope and unmount guards, retained diagnostics, whole-frame duration, and ${planRequests.length} synthetic planner requests.`);
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
