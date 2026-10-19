const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const root = path.resolve(__dirname, '../..')
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'))
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright')
const { chromium } = playwright
const screenshotDirectory = path.join(root, '.codex-tmp/moss-image-20261019')

function near(actual, expected, tolerance, message) {
  assert(Math.abs(actual - expected) <= tolerance, `${message}: got ${actual}, expected ${expected} ± ${tolerance}`)
}

;(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {
      contents: `import React from 'react';import {createRoot} from 'react-dom/client';import {useStore,getModelMode,modelSupportsImageWorkflow} from './src/stores/useStore';import {MossImagePromptHelper} from './src/components/Sidebar/MossImagePromptHelper';window.store=useStore;window.catalog={getModelMode,modelSupportsImageWorkflow};useStore.setState({params:{...useStore.getState().params,prompt:'A cream travel poster.',resolution:'800x600'}});createRoot(document.getElementById('root')).render(<MossImagePromptHelper/>);`,
      resolveDir: path.join(root, 'ui'),
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' },
    logLevel: 'silent',
  })
  const browser = await chromium.launch({
    headless: true,
    ...(process.platform === 'win32' ? { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' } : {}),
  })
  try {
    for (const width of [1280, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, hasTouch: width === 390 })
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.route('**/api/**', route => route.fulfill({ json: {} }))
      await page.setContent('<div id="root"></div>')
      const assets = path.join(root, 'ui/dist/assets')
      const css = fs.readdirSync(assets).find(name => name.endsWith('.css'))
      assert(css, 'compiled UI CSS is available')
      await page.addStyleTag({ content: fs.readFileSync(path.join(assets, css), 'utf8') })
      await page.addScriptTag({ content: bundle.outputFiles[0].text })

      assert.equal(await page.evaluate(() => window.catalog.getModelMode('moss_image_sft', 'moss_image')), 'image')
      assert.equal(await page.evaluate(() => window.store.getState().enabledModels.has('moss_image_sft')), true)
      assert.equal(await page.evaluate(() => window.store.getState().enabledModels.has('moss_image')), false)
      assert.deepEqual(await page.evaluate(() => {
        const model = {model_type:'moss_image_sft',family:'moss_image',supports_image_edit:false,supports_image_inpaint:false,supports_image_outpaint:false}
        return [window.catalog.modelSupportsImageWorkflow(model,'generate'),
          window.catalog.modelSupportsImageWorkflow(model,'inpaint'),
          window.catalog.modelSupportsImageWorkflow(model,'outpaint'),
          window.catalog.modelSupportsImageWorkflow(model,'generate',true)]
      }), [true,false,false,false])

      const prompt = () => page.evaluate(() => window.store.getState().params.prompt)
      const dialog = page.getByRole('dialog', { name: 'MOSS Image Prompt Helper' })
      const canvas = page.getByTestId('moss-image-layout-canvas')
      const boxField = page.getByLabel('Bounding box coordinates')

      await page.getByRole('button', { name: 'Prompt Helper' }).click()
      assert.equal(await page.getByLabel('Summary caption').inputValue(), 'A cream travel poster.')
      const initialCanvas = await canvas.boundingBox()
      assert(initialCanvas && initialCanvas.width > 0 && initialCanvas.height > 0)
      near(initialCanvas.width / initialCanvas.height, 800 / 600, 0.02, `canvas aspect ratio at ${width}px`)
      const dialogOverflow = await dialog.evaluate(element => element.scrollWidth - element.clientWidth)
      assert(dialogOverflow <= 1, `dialog overflows horizontally by ${dialogOverflow}px at ${width}px`)

      async function canvasXY(y, x, bounds) {
        const rect = bounds || await canvas.boundingBox()
        return {
          x: rect.x + ((x + 400) / 800) * rect.width,
          y: rect.y + ((y + 300) / 600) * rect.height,
        }
      }
      async function drag(from, to, bounds) {
        const start = await canvasXY(from[0], from[1], bounds)
        const end = await canvasXY(to[0], to[1], bounds)
        await page.mouse.move(start.x, start.y)
        await page.mouse.down()
        await page.mouse.move(end.x, end.y, { steps: 5 })
        await page.mouse.up()
      }
      async function fieldBox() {
        return JSON.parse(await boxField.inputValue())
      }

      await drag([-120, -160], [120, 160])
      assert.equal(await dialog.count(), 1, 'drawing on the canvas must not close the dialog')
      await page.getByLabel('Label').selectOption('text')
      await page.getByLabel('Item caption').fill('Large navy uppercase title.')
      await page.getByLabel('Exact visible text').fill('EXPLORE')
      let box = await fieldBox()
      assert(box.every(Number.isInteger), `drawn coordinates should be integers: ${box}`)
      const beforeMove = box
      const bounds = await canvas.boundingBox()
      const center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
      await drag(center, [center[0] + 24, center[1] + 32], bounds)
      const afterMove = await fieldBox()
      near(afterMove[0] - beforeMove[0], 24, 4, 'move shifts ymin')
      near(afterMove[1] - beforeMove[1], 32, 4, 'move shifts xmin')
      near(afterMove[2] - beforeMove[2], 24, 4, 'move shifts ymax')
      near(afterMove[3] - beforeMove[3], 32, 4, 'move shifts xmax')

      const resizeStart = await canvasXY(afterMove[2], afterMove[3], bounds)
      const resizeEnd = await canvasXY(afterMove[2] + 30, afterMove[3] + 40, bounds)
      await page.mouse.move(resizeStart.x, resizeStart.y)
      await page.mouse.down()
      await page.mouse.move(resizeEnd.x, resizeEnd.y, { steps: 5 })
      await page.mouse.up()
      const afterResize = await fieldBox()
      assert(afterResize[2] > afterMove[2] + 20, `resize should extend ymax: ${afterMove} -> ${afterResize}`)
      assert(afterResize[3] > afterMove[3] + 30, `resize should extend xmax: ${afterMove} -> ${afterResize}`)

      await boxField.fill('[-100, -100, 200, 220]')
      assert.deepEqual(await fieldBox(), [-100, -100, 200, 220], 'numeric editing updates the working box')
      fs.mkdirSync(screenshotDirectory, { recursive: true })
      await page.screenshot({ path: path.join(screenshotDirectory, `ui-moss-image-${width}.png`) })
      await page.getByRole('button', { name: 'Save prompt' }).click()
      const saved = JSON.parse(await prompt())
      assert.equal(saved.summary.caption, 'A cream travel poster.')
      assert.equal(saved.items[0].label, 'text')
      assert.equal(saved.items[0].caption, 'Large navy uppercase title.')
      assert.equal(saved.items[0].text, 'EXPLORE')
      assert.deepEqual(saved.items[0].bbox, [-100, -100, 200, 220])
      assert.equal(saved.bbox_coord_space, 'abs_centralized')
      assert.equal(saved.bbox_format, 'yxyx')
      assert.equal(await dialog.count(), 0, 'saving closes the dialog')

      const fractional = JSON.stringify({
        summary: { meta: [], canvas: 'single', caption: 'Keep this scene.' },
        items: [{ label: 'object', caption: 'A tree.', bbox: [-1.5, -20, 100.5, 20] }],
        bbox_coord_space: 'abs_centralized',
        bbox_format: 'yxyx',
      })
      await page.evaluate(value => window.store.setState({ params: { ...window.store.getState().params, prompt: value } }), fractional)
      await page.getByRole('button', { name: 'Prompt Helper' }).click()
      await page.getByRole('button', { name: 'Save prompt' }).click()
      await page.getByRole('alert').filter({ hasText: 'integer pixel corners' }).waitFor()
      assert.equal(await prompt(), fractional, 'invalid fractional boxes must not overwrite the prompt')
      await boxField.fill('[-1, -20, 100, 20]')
      await page.getByRole('button', { name: 'Save prompt' }).click()
      assert.equal(JSON.parse(await prompt()).items[0].bbox[0], -1)

      const malformed = '{"summary":{"caption":"keep source"},"items":['
      await page.evaluate(value => window.store.setState({ params: { ...window.store.getState().params, prompt: value } }), malformed)
      await page.getByRole('button', { name: 'Prompt Helper' }).click()
      await page.getByRole('alert').filter({ hasText: 'Original prompt text is preserved' }).waitFor()
      assert.equal(await dialog.count(), 0)
      assert.equal(await prompt(), malformed)

      await page.evaluate(() => window.store.setState({ params: { ...window.store.getState().params, prompt: 'A quiet garden.' } }))
      await page.getByRole('button', { name: 'Prompt Helper' }).click()
      const overlay = dialog.locator('xpath=..')
      await overlay.click({ position: { x: 2, y: 2 } })
      assert.equal(await dialog.count(), 0, 'backdrop click closes the dialog')
      assert.equal(await prompt(), 'A quiet garden.', 'closing without saving preserves the prompt')

      await page.getByRole('button', { name: 'Prompt Helper' }).click()
      await page.keyboard.press('Escape')
      assert.equal(await dialog.count(), 0, 'Escape closes the dialog')
      assert.deepEqual(errors, [], `page errors at ${width}px`)
      await page.close()
    }
    console.log('MOSS Image helper desktop/mobile: draw, move, resize, numeric edit, text item, save, bounds errors, malformed prompt, backdrop and Escape passed.')
  } finally {
    await browser.close()
  }
})().catch(error => { console.error(error); process.exitCode = 1 })
