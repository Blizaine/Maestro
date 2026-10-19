const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const { chromium } = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

function closeBox(actual, expected) {
  assert.equal(actual.length, 4);
  actual.forEach((value, index) => assert(Math.abs(value - expected[index]) <= 2, `${actual} != ${expected}`));
}

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {
      contents: `import React from 'react';import {createRoot} from 'react-dom/client';import {useStore} from './src/stores/useStore';import {Flux3PromptHelper} from './src/components/Sidebar/Flux3PromptHelper';window.store=useStore;useStore.setState({params:{...useStore.getState().params,prompt:'A quiet garden.',resolution:'1024x1024'},imageRefs:[]});createRoot(document.getElementById('root')).render(<Flux3PromptHelper/>);`,
      resolveDir: path.join(root, 'ui'), loader: 'tsx',
    },
    bundle: true, write: false, jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent',
  });
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' } : {}),
  });
  try {
    for (const width of [1280, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, hasTouch: width === 390 });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/api/**', route => route.fulfill({ json: {} }));
      await page.setContent('<div id="root"></div>');
      const assets = path.join(root, 'ui/dist/assets');
      await page.addStyleTag({ content: fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8') });
      await page.addScriptTag({ content: bundle.outputFiles[0].text });

      async function draw(box) {
        const canvas = page.getByTestId('flux3-layout-canvas');
        await canvas.scrollIntoViewIfNeeded();
        const rect = await canvas.boundingBox();
        assert(rect && rect.width > 0 && rect.height > 0);
        await page.mouse.move(rect.x + rect.width * box[1] / 1000, rect.y + rect.height * box[0] / 1000);
        await page.mouse.down();
        await page.mouse.move(rect.x + rect.width * box[3] / 1000, rect.y + rect.height * box[2] / 1000, { steps: 5 });
        await page.mouse.up();
      }
      async function savedElements() {
        const prompt = await page.evaluate(() => window.store.getState().params.prompt);
        return JSON.parse(prompt.slice(prompt.indexOf('[')));
      }
      async function assertReference(naturalWidth, naturalHeight) {
        const image = page.getByRole('img', { name: 'Selected reference' });
        await image.waitFor();
        await page.waitForFunction(([w, h]) => {
          const image = document.querySelector('img[alt="Selected reference"]');
          const canvas = document.querySelector('[data-testid="flux3-layout-canvas"]');
          if (!image || !canvas || image.naturalWidth !== w || image.naturalHeight !== h) return false;
          const box = canvas.getBoundingClientRect();
          return Math.abs(box.width / box.height - w / h) < 0.01;
        }, [naturalWidth, naturalHeight]);
      }

      // A new image keeps its blank output-layout workflow.
      await page.getByRole('button', { name: 'Prompt Helper' }).click();
      assert.equal(await page.getByLabel('Layout view').inputValue(), 'target');
      assert.equal(await page.getByRole('img', { name: 'Selected reference' }).count(), 0);
      await page.getByRole('button', { name: 'Add element' }).click();
      await page.getByLabel('Element description').fill('A flowering cherry tree.');
      await draw([100, 50, 600, 350]);
      await page.getByRole('button', { name: 'Save prompt' }).click();
      closeBox((await savedElements())[0].bbox, [100, 50, 600, 350]);

      // Use decoded PNGs: invalid image bytes hid the original preview gap.
      await page.evaluate(async () => {
        async function reference(width, height, color, name) {
          const canvas = document.createElement('canvas');
          canvas.width = width; canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = color; ctx.fillRect(0, 0, width, height);
          ctx.fillStyle = 'white'; ctx.fillRect(width / 4, height / 4, width / 2, height / 2);
          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
          return new File([blob], name, { type: 'image/png' });
        }
        const refs = [await reference(1000, 500, 'red', 'landscape.png'), await reference(300, 600, 'blue', 'portrait-' + 'long-upload-name-'.repeat(8) + '.png')];
        window.store.setState({ params: { ...window.store.getState().params, prompt: 'A suitcase beside a chair.' }, imageRefs: refs });
      });
      await page.getByRole('button', { name: 'Prompt Helper' }).click();
      assert.equal(await page.getByLabel('Layout view').inputValue(), 'source');
      await assertReference(1000, 500);
      await page.getByRole('button', { name: 'Add element' }).click();
      assert.equal(await page.getByLabel('Content source').inputValue(), 'ref_image_0');
      await page.getByLabel('Element description').fill('The same suitcase.');
      await draw([100, 100, 800, 600]);
      await page.getByLabel('Layout view').selectOption('target');
      await draw([50, 50, 300, 350]);
      await page.getByLabel('Remove this reference element').check();
      await page.getByRole('button', { name: 'Save prompt' }).click();
      let elements = await savedElements();
      assert.equal(elements[0].from, 'ref_image_0');
      assert.equal(elements[0].tgt_bbox, null);
      closeBox(elements[0].src_bbox, [100, 100, 800, 600]);

      // Reopening preserves removal; a portrait keeps its natural geometry.
      await page.getByRole('button', { name: 'Prompt Helper' }).click();
      await assertReference(1000, 500);
      assert.equal(await page.getByLabel('Remove this reference element').isChecked(), true);
      await page.getByLabel('Reference image').selectOption('1');
      await assertReference(300, 600);
      const horizontalOverflow = await page.getByRole('dialog').evaluate(dialog => dialog.scrollWidth - dialog.clientWidth);
      assert(horizontalOverflow <= 1, `Dialog overflows by ${horizontalOverflow}px at width ${width}`);
      await page.getByRole('button', { name: 'Add element' }).click();
      assert.equal(await page.getByLabel('Content source').inputValue(), 'ref_image_1');
      await page.getByLabel('Element description').fill('The same chair.');
      await draw([100, 100, 700, 800]);
      if (process.env.MAESTRO_UI_SCREENSHOT_DIR) {
        fs.mkdirSync(process.env.MAESTRO_UI_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: path.join(process.env.MAESTRO_UI_SCREENSHOT_DIR, `flux3-helper-reference-${width}.png`) });
      }
      await page.getByRole('button', { name: 'Save prompt' }).click();
      elements = await savedElements();
      assert.equal(elements.length, 2);
      assert.equal(elements[0].tgt_bbox, null);
      assert.equal(elements[1].from, 'ref_image_1');
      closeBox(elements[1].src_bbox, [100, 100, 700, 800]);

      // Reopen on the bound reference, and close without rewriting the prompt.
      await page.evaluate(() => {
        const prompt = window.store.getState().params.prompt;
        const elements = JSON.parse(prompt.slice(prompt.indexOf('[')));
        window.store.setState({ params: { ...window.store.getState().params, prompt: 'The same chair <element_2>.\n' + JSON.stringify([elements[1]]) } });
      });
      const before = await page.evaluate(() => window.store.getState().params.prompt);
      await page.getByRole('button', { name: 'Prompt Helper' }).click();
      await assertReference(300, 600);
      assert.equal(await page.getByLabel('Reference image').inputValue(), '1');
      await page.getByRole('button', { name: 'Close prompt helper' }).click();
      assert.equal(await page.evaluate(() => window.store.getState().params.prompt), before);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Flux3 helper desktop/mobile: generated layout, automatic reference preview, decoded aspect ratios, source drawing, multi-reference binding, removal and reopen passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
