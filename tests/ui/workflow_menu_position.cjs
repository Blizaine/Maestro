// Model-free browser regression for the shared Studio workflow popover.
// Run: node tests/ui/workflow_menu_position.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const fixtures = {
  audio: [
    {label: 'Create', options: [
      {value: 'a1', label: 'Audio 01', description: 'Create the first audio workflow'},
      {value: 'a2', label: 'Audio 02', description: 'Create the second audio workflow'},
      {value: 'a3', label: 'Audio 03', description: 'Create the third audio workflow'},
    ]},
    {label: 'Process', options: [
      {value: 'a4', label: 'Audio 04', description: 'Process the fourth audio workflow'},
      {value: 'a5', label: 'Audio 05', description: 'Process the fifth audio workflow'},
    ]},
  ],
  image: [
    {label: 'Create', options: [
      {value: 'i1', label: 'Image 01', description: 'Create an image'},
    ]},
    {label: 'Transform', options: [
      {value: 'i2', label: 'Image 02', description: 'Transform an image'},
      {value: 'i3', label: 'Image 03', description: 'Expand an image'},
    ]},
    {label: 'Finish', options: [
      {value: 'i4', label: 'Image 04', description: 'Finish an image'},
    ]},
  ],
  video: [
    {label: 'Create', options: Array.from({length: 6}, (_, index) => ({
      value: 'v' + (index + 1), label: 'Video ' + String(index + 1).padStart(2, '0'),
      description: 'Create video workflow ' + (index + 1),
    }))},
    {label: 'Transform', options: Array.from({length: 6}, (_, index) => ({
      value: 'v' + (index + 7), label: 'Video ' + String(index + 7).padStart(2, '0'),
      description: 'Transform video workflow ' + (index + 7),
    }))},
    {label: 'Finish', options: Array.from({length: 2}, (_, index) => ({
      value: 'v' + (index + 13), label: 'Video ' + String(index + 13).padStart(2, '0'),
      description: 'Finish video workflow ' + (index + 13),
    }))},
  ],
};

const html = '<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head><body><div id="root"></div></body></html>';
const fixtureCss = [
  '#sidebar { position: fixed; inset: 0; width: 100%; height: 100%; }',
  '#selected-value { display: none; }',
  '#workflow-host { position: absolute; top: var(--workflow-top, 100px); left: var(--workflow-left, 24px); width: var(--workflow-width, 260px); }',
  '#workflow-host > div { width: 100%; }',
  '#workflow-host button[aria-expanded] { height: 46px; }',
  '#settings-anchor { position: absolute; top: var(--settings-top, 680px); left: 260px; width: 100px; height: 44px; }',
  '#settings-anchor button { width: 100%; height: 100%; }',
].join('\n');

function getCompiledCss() {
  const dist = path.join(root, 'ui/dist');
  const distHtml = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');
  const cssFile = distHtml.match(/<link[^>]+href="\/assets\/([^"]+\.css)"/)?.[1];
  assert.ok(cssFile, 'Built UI index references its compiled stylesheet');
  return fs.readFileSync(path.join(dist, 'assets', cssFile), 'utf8');
}

const compiledCss = getCompiledCss();

async function capturePhoneScreenshot(page, mode) {
  const directory = process.env.MAESTRO_WORKFLOW_SCREENSHOTS;
  if (!directory) return;
  fs.mkdirSync(directory, {recursive: true});
  await page.screenshot({path: path.join(path.resolve(directory), 'workflow-menu-' + mode + '-phone.png')});
}

async function buildBundle() {
  const source = [
    "import * as React from 'react';",
    "import {createRoot} from 'react-dom/client';",
    "import {Circle} from 'lucide-react';",
    "import {StudioWorkflowSelect} from './src/components/Sidebar/StudioWorkflowSelect';",
    "import {SidebarDialog, SidebarLayoutContext} from './src/components/Sidebar/SidebarPanels';",
    'const FIXTURES = ' + JSON.stringify(fixtures) + ';',
    'function App({mode}) {',
    '  const groups = React.useMemo(() => FIXTURES[mode].map(group => ({...group, options: group.options.map(option => ({...option, icon: Circle}))})), [mode]);',
    '  const [value, setValue] = React.useState(groups[0].options[0].value);',
    '  const [settingsOpen, setSettingsOpen] = React.useState(false);',
    '  const [sidebar, setSidebar] = React.useState(null);',
    '  const [settings, setSettings] = React.useState(null);',
    '  const activeOption = groups.flatMap(group => group.options).find(option => option.value === value);',
    '  return React.createElement(SidebarLayoutContext.Provider, {value: {sidebar, settings}},',
    '    React.createElement(React.Fragment, null,',
    '      React.createElement("div", {id: "sidebar", ref: setSidebar},',
    '        React.createElement("div", {id: "workflow-host"},',
    '          React.createElement(StudioWorkflowSelect, {value, activeOption, groups, hint: "Fixture workflows", onChange: setValue})),',
    '        React.createElement("div", {id: "settings-anchor", ref: setSettings},',
    '          React.createElement("button", {id: "settings-trigger", type: "button", onClick: () => setSettingsOpen(true)}, "Open settings")),',
    '        React.createElement(SidebarDialog, {open: settingsOpen, title: "Display settings", variant: "settings", anchor: settings, onClose: () => setSettingsOpen(false)},',
    '          React.createElement("div", null, "Settings panel body"))),',
    '      React.createElement("output", {id: "selected-value"}, value)));',
    '}',
    'window.mountFixture = mode => {',
    '  window.fixtureRoot = createRoot(document.getElementById("root"));',
    '  window.fixtureRoot.render(React.createElement(App, {mode}));',
    '};',
    'window.setTriggerRect = ({left, top, width}) => {',
    '  document.documentElement.style.setProperty("--workflow-left", String(left) + "px");',
    '  document.documentElement.style.setProperty("--workflow-top", String(top) + "px");',
    '  document.documentElement.style.setProperty("--workflow-width", String(width) + "px");',
    '};',
    'window.setFakeVisualViewport = (offsetTop, height) => {',
    '  let state = {offsetTop, height};',
    '  const viewport = new EventTarget();',
    '  Object.defineProperties(viewport, {',
    '    offsetTop: {get: () => state.offsetTop},',
    '    offsetLeft: {get: () => 0},',
    '    width: {get: () => window.innerWidth},',
    '    height: {get: () => state.height},',
    '  });',
    '  Object.defineProperty(window, "visualViewport", {configurable: true, value: viewport});',
    '  window.updateFakeVisualViewport = (nextOffsetTop, nextHeight) => {',
    '    state = {offsetTop: nextOffsetTop, height: nextHeight};',
    '    viewport.dispatchEvent(new Event("resize"));',
    '    viewport.dispatchEvent(new Event("scroll"));',
    '  };',
    '};',
  ].join('\n');

  const result = await esbuild.build({
    stdin: {contents: source, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true,
    write: false,
    format: 'iife',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  return result.outputFiles[0].text;
}

async function createFixturePage(browser, bundle, mode, contextOptions) {
  const context = await browser.newContext(contextOptions);
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setContent(html);
  await page.addStyleTag({content: compiledCss});
  await page.addStyleTag({content: fixtureCss});
  await page.addScriptTag({content: bundle});
  await page.evaluate(modeValue => window.mountFixture(modeValue), mode);
  await page.getByRole('button', {name: /^Workflow:/}).waitFor({state: 'visible'});
  await page.getByRole('button', {name: 'Open settings'}).waitFor({state: 'visible'});
  return {context, page, pageErrors};
}

async function inspectWorkflowDialog(page) {
  return page.evaluate(() => {
    const anchor = document.querySelector('#workflow-host > div').getBoundingClientRect();
    const panel = document.querySelector('[data-sidebar-overlay="workflow"]');
    const scroller = panel.children[1];
    const panelRect = panel.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const options = Array.from(scroller.querySelectorAll('button')).map(button => {
      const rect = button.getBoundingClientRect();
      return {top: rect.top, bottom: rect.bottom, visible: rect.top >= scrollerRect.top - 1 && rect.bottom <= scrollerRect.bottom + 1};
    });
    return {
      anchor: {left: anchor.left, right: anchor.right, top: anchor.top, bottom: anchor.bottom, width: anchor.width},
      panel: {left: panelRect.left, right: panelRect.right, top: panelRect.top, bottom: panelRect.bottom, width: panelRect.width, height: panelRect.height},
      style: {top: panel.style.top, bottom: panel.style.bottom, width: panel.style.width, maxHeight: panel.style.maxHeight},
      scroller: {top: scrollerRect.top, bottom: scrollerRect.bottom, clientHeight: scroller.clientHeight, scrollHeight: scroller.scrollHeight, scrollTop: scroller.scrollTop},
      options,
    };
  });
}

async function openWorkflow(page) {
  await page.getByRole('button', {name: /^Workflow:/}).click();
  await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'visible'});
}

function assertAnchored(info, expectedWidth, viewportHeight, label) {
  assert.ok(Math.abs(info.panel.top - (info.anchor.bottom + 6)) <= 1,
    label + ': workflow menu begins 6px below its trigger');
  assert.equal(info.panel.width, expectedWidth, label + ': workflow menu width follows its selector');
  assert.equal(info.style.bottom, '', label + ': workflow menu does not use bottom docking');
  assert.ok(info.panel.bottom <= viewportHeight - 8 + 1, label + ': workflow menu fits above the visible viewport edge');
}

function assertCompact(info, optionCount, label) {
  assert.equal(info.options.length, optionCount, label + ': all fixture workflow options are rendered');
  assert.ok(info.options.every(option => option.visible), label + ': every short-menu option is visible without scrolling');
  assert.ok(info.panel.height < 420, label + ': short menu stays content-sized instead of filling the viewport');
}

async function runPhoneShortMenus(browser, bundle) {
  for (const [mode, count] of [['audio', 5], ['image', 4]]) {
    const {context, page, pageErrors} = await createFixturePage(browser, bundle, mode, {
      viewport: {width: 393, height: 760},
    });
    try {
      assert.equal(await page.evaluate(() => matchMedia('(max-width: 767px)').matches), true,
        mode + ': phone viewport activates compact sidebar layout');
      const footerBefore = await page.locator('#settings-anchor').boundingBox();
      await openWorkflow(page);
      const info = await inspectWorkflowDialog(page);
      assertAnchored(info, 260, 760, mode);
      assertCompact(info, count, mode);
      const footerWhileOpen = await page.locator('#settings-anchor').boundingBox();
      assert.deepEqual(footerWhileOpen, footerBefore, mode + ': opening the menu does not move the footer control');
      await capturePhoneScreenshot(page, mode);

      if (mode === 'audio') {
        await page.getByRole('dialog', {name: 'Choose a workflow'}).getByRole('button', {name: /Audio 03/}).click();
        await page.waitForFunction(() => document.querySelector('#selected-value').textContent === 'a3');
        await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});
      } else {
        await page.keyboard.press('Escape');
        await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});
        await openWorkflow(page);
        await page.mouse.click(380, 730);
        await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});
      }
      assert.deepEqual(pageErrors, [], mode + ': the browser reports no runtime errors');
    } finally {
      await context.close();
    }
  }
}

async function runPhoneVideo(browser, bundle) {
  const {context, page, pageErrors} = await createFixturePage(browser, bundle, 'video', {
    viewport: {width: 393, height: 760},
  });
  try {
    await openWorkflow(page);
    const info = await inspectWorkflowDialog(page);
    assertAnchored(info, 260, 760, 'phone Video');
    assert.equal(info.options.length, 14, 'phone Video: the fixture has all 14 workflows');
    assert.ok(info.panel.bottom <= 752 + 1, 'phone Video: outer bounds remain inside the viewport');
    assert.ok(info.scroller.scrollHeight > info.scroller.clientHeight,
      'phone Video: long options scroll inside the dialog scroller');
    await capturePhoneScreenshot(page, 'video');

    const lastOption = page.getByRole('dialog', {name: 'Choose a workflow'}).getByRole('button', {name: /Video 14/});
    await lastOption.scrollIntoViewIfNeeded();
    const scrolled = await inspectWorkflowDialog(page);
    assert.ok(scrolled.scroller.scrollTop > 0, 'phone Video: reaching the last option scrolls the inner list');
    assert.ok(scrolled.options.at(-1).visible, 'phone Video: the last option is reachable inside the visible scroller');
    await lastOption.click();
    await page.waitForFunction(() => document.querySelector('#selected-value').textContent === 'v14');
    await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});
    assert.deepEqual(pageErrors, [], 'phone Video: the browser reports no runtime errors');
  } finally {
    await context.close();
  }
}

async function runDesktop(browser, bundle) {
  const {context, page, pageErrors} = await createFixturePage(browser, bundle, 'audio', {
    viewport: {width: 1280, height: 900},
  });
  try {
    await openWorkflow(page);
    let info = await inspectWorkflowDialog(page);
    assertAnchored(info, 260, 900, 'desktop Audio');
    assertCompact(info, 5, 'desktop Audio');
    await page.keyboard.press('Escape');
    await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});

    await page.evaluate(() => window.setTriggerRect({left: 1200, top: 100, width: 260}));
    await openWorkflow(page);
    info = await inspectWorkflowDialog(page);
    assert.equal(info.panel.width, 260, 'desktop horizontal clamp preserves selector width');
    assert.equal(info.panel.left, 1012, 'desktop horizontal clamp keeps the menu within the right viewport inset');
    assert.ok(info.panel.right <= 1272 + 1, 'desktop horizontal clamp leaves an 8px viewport margin');
    assert.deepEqual(pageErrors, [], 'desktop workflow menu has no browser runtime errors');
  } finally {
    await context.close();
  }
}

async function runTouchLandscape(browser, bundle) {
  const {context, page, pageErrors} = await createFixturePage(browser, bundle, 'audio', {
    viewport: {width: 900, height: 320},
    isMobile: true,
    hasTouch: true,
  });
  try {
    const mobileSidecar = await page.evaluate(() => matchMedia('(max-width: 767px), (pointer: coarse) and (hover: none) and (orientation: landscape)').matches);
    assert.equal(mobileSidecar, true, 'touch landscape viewport activates mobile sidecar behavior');
    await openWorkflow(page);
    const info = await inspectWorkflowDialog(page);
    assertAnchored(info, 260, 320, 'touch landscape Audio');
    assert.ok(info.panel.bottom <= 312 + 1, 'touch landscape Audio: menu stays inside the short viewport');
    assert.ok(info.scroller.scrollHeight > info.scroller.clientHeight,
      'touch landscape Audio: constrained choices scroll inside the menu');
    assert.deepEqual(pageErrors, [], 'touch landscape workflow menu has no browser runtime errors');
  } finally {
    await context.close();
  }
}

async function runViewportFallbackAndSettings(browser, bundle) {
  const {context, page, pageErrors} = await createFixturePage(browser, bundle, 'video', {
    viewport: {width: 393, height: 760},
  });
  try {
    await page.evaluate(() => window.setFakeVisualViewport(0, 760));
    await page.getByRole('button', {name: 'Open settings'}).click();
    const settings = page.getByRole('dialog', {name: 'Display settings'});
    await settings.waitFor({state: 'visible'});
    let settingsBounds = await settings.evaluate(panel => {
      const rect = panel.getBoundingClientRect();
      return {top: rect.top, bottom: rect.bottom, styleTop: panel.style.top, styleBottom: panel.style.bottom};
    });
    assert.equal(settingsBounds.styleTop, '', 'mobile settings remain bottom anchored');
    assert.notEqual(settingsBounds.styleBottom, '', 'mobile settings keep their footer bottom offset');
    assert.ok(settingsBounds.bottom <= 680 - 5, 'mobile settings open upward from the footer control');

    await page.evaluate(() => window.updateFakeVisualViewport(320, 300));
    await page.waitForFunction(() => {
      const panel = document.querySelector('[data-sidebar-overlay="settings"]');
      return panel && panel.style.bottom === '148px';
    });
    settingsBounds = await settings.evaluate(panel => {
      const rect = panel.getBoundingClientRect();
      return {top: rect.top, bottom: rect.bottom, styleTop: panel.style.top, styleBottom: panel.style.bottom};
    });
    assert.equal(settingsBounds.styleTop, '', 'keyboard viewport changes do not switch settings to top anchoring');
    assert.ok(settingsBounds.bottom <= 612 + 1, 'settings bottom remains inside the simulated visible viewport');
    assert.ok(settingsBounds.top >= 328 - 1, 'settings panel remains inside the simulated visible viewport');
    await page.keyboard.press('Escape');
    await settings.waitFor({state: 'hidden'});

    // The workflow trigger remains above the keyboard-panned visible viewport.
    await openWorkflow(page);
    let info = await inspectWorkflowDialog(page);
    assert.equal(info.style.top, '328px', 'offscreen workflow trigger falls back to the visible viewport top');
    assert.equal(info.style.maxHeight, '284px', 'offscreen workflow fallback uses the remaining visible viewport');
    assert.ok(info.panel.top >= 328 - 1 && info.panel.bottom <= 612 + 1,
      'offscreen workflow fallback stays inside the visible viewport');
    assert.ok(info.scroller.scrollHeight > info.scroller.clientHeight,
      'offscreen long workflow list remains scrollable inside its fallback bounds');
    await page.keyboard.press('Escape');
    await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});

    // A visible trigger with less than 160px remaining also falls back.
    await page.evaluate(() => window.setTriggerRect({left: 24, top: 450, width: 260}));
    await openWorkflow(page);
    info = await inspectWorkflowDialog(page);
    assert.equal(info.style.top, '328px', 'crowded workflow trigger falls back when less than 160px remains below it');
    assert.ok(info.panel.bottom <= 612 + 1, 'crowded workflow menu remains within the visible viewport');
    await page.keyboard.press('Escape');
    await page.getByRole('dialog', {name: 'Choose a workflow'}).waitFor({state: 'hidden'});
    assert.deepEqual(pageErrors, [], 'visual viewport keyboard scenarios have no browser runtime errors');
  } finally {
    await context.close();
  }
}

async function main() {
  const bundle = await buildBundle();
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}
      : {}),
  });
  try {
    await runPhoneShortMenus(browser, bundle);
    await runPhoneVideo(browser, bundle);
    await runDesktop(browser, bundle);
    await runTouchLandscape(browser, bundle);
    await runViewportFallbackAndSettings(browser, bundle);
  } finally {
    await browser.close();
  }
}

main()
  .then(() => console.log('Model-free workflow menu positioning checks passed.'))
  .catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
