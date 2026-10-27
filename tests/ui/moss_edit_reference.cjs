// Real Studio store submissions; every HTTP request is mocked, no GPU jobs.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {contents: "import {useStore} from './src/stores/useStore'; window.store = useStore;",
      resolveDir: path.join(root, 'ui'), loader: 'ts'},
    bundle: true, write: false, logLevel: 'silent', define: {'process.env.NODE_ENV': '"test"'},
  });
  const browser = await chromium.launch({headless: true,
    ...(process.platform === 'win32' ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {}),
    args: ['--disable-gpu'],
  });
  try {
    for (const width of [1200, 390]) {
      const context = await browser.newContext({viewport: {width, height: 900}});
      const page = await context.newPage();
      const submissions = [], uploads = [], errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', async route => {
        const key = new URL(route.request().url()).pathname;
        if (key === '/') return route.fulfill({contentType: 'text/html', body: '<div>Studio contract test</div>'});
        if (key === '/api/v1/upload') {
          const file = `uploads/reference-${uploads.length + 1}.png`;
          uploads.push(file);
          return route.fulfill({json: {path: file, filename: path.basename(file), url: '/image.png'}});
        }
        if (key === '/api/v1/generate') {
          submissions.push(route.request().postDataJSON());
          return route.fulfill({json: {job_id: `mock-${submissions.length}`, status: 'queued'}});
        }
        if (key.includes('/status/') || key.includes('/jobs/mock-')) {
          return route.fulfill({json: {status: 'completed', output_files: []}});
        }
        return route.fulfill({json: {jobs: [], outputs: [], items: [], ok: true}});
      });
      await page.goto('http://moss-reference.test');
      await page.addScriptTag({content: bundle.outputFiles[0].text});
      async function configure(modelType, missingChoices = false, staleModelMode = 2, referenceCount = 2) {
        await page.evaluate(({modelType, missingChoices, staleModelMode, referenceCount}) => {
          const moss = modelType.startsWith('moss_');
          const models = ['fibo2_edit_turbo', 'moss_image_edit', 'moss_image_edit_hq', 'flux3_klein', 'flux3_klein_base'].map(id => ({
            model_type: id, architecture: id.startsWith('moss_') ? 'moss_image' : id,
            family: id.startsWith('moss_') ? 'moss_image' : id.startsWith('flux3_') ? 'flux3' : 'fibo2',
            supports_image_edit: true, supports_instruction_image_edit: true,
          }));
          window.store.setState(state => ({
            models, enabledModels: new Set(models.map(model => model.model_type)), modelVisibilityOverrides: {},
            generationMode: 'image', studioImageWorkflow: 'generate', studioVideoWorkflow: 'frames',
            isGenerating: false, jobs: [], enhanceOnGeneration: false, enhanceOnGenerationDefault: false,
            imageRefs: [new File(['first-image'], 'first.png', {type: 'image/png'}),
                        new File(['second-image'], 'second.png', {type: 'image/png'})].slice(0, referenceCount),
            imageRefType: 'KI',
            modelOptions: {model_type: modelType, architecture: moss ? 'moss_image' : modelType,
              image_outputs: true, max_image_refs: 4, no_background_removal: true,
              image_ref_choices: {choices: missingChoices ? [] : modelType.startsWith('flux3_')
                ? [['None', ''], ['Ordered references', 'I'], ['Main reference sets shape', 'KI']]
                : [['Ordered references', moss ? 'I' : 'KI']], default: moss ? 'I' : 'KI'}},
            params: {...state.params, model_type: modelType, image_mode: 1, model_mode: staleModelMode, prompt: 'Replace the hands using reference 2.',
              resolution: '720x1280', num_inference_steps: moss ? 50 : 4,
              video_prompt_type: 'KI', image_refs: undefined, image_start: undefined, image_end: undefined,
              activated_loras: [], loras_multipliers: '', spatial_upsampling: '', temporal_upsampling: ''},
          }));
        }, {modelType, missingChoices, staleModelMode, referenceCount});
      }
      await configure('fibo2_edit_turbo');
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.length, 1, 'Fibo control submits');
      assert.equal(submissions[0].video_prompt_type, 'KI', 'Keep native Fibo selector');
      assert.equal(submissions[0].model_mode, 2, 'Do not change unrelated Fibo submit fields');
      for (const modelType of ['moss_image_edit', 'moss_image_edit_hq']) {
        for (const staleModelMode of [0, 2]) {
          await configure(modelType, false, staleModelMode);
          const before = uploads.length;
          await page.evaluate(() => window.store.getState().startGeneration());
          const submitted = submissions.at(-1);
          assert.equal(submitted.model_type, modelType);
          assert.equal(submitted.video_prompt_type, 'I', 'Drop the previous model’s stale K flag');
          assert.deepEqual(submitted.image_refs, uploads.slice(before), 'Preserve all references in upload order');
          assert.equal(submitted.image_refs.length, 2);
          assert.equal(submitted.image_mode, 1);
          assert.equal(submitted.resolution, '720x1280');
          assert.equal(Object.hasOwn(submitted, 'model_mode'), false, 'Do not submit another family\'s model selector');
          assert.equal(await page.evaluate(() => window.store.getState().params.model_mode), staleModelMode,
            'Request cleanup must not overwrite the persisted model preferences');
        }
      }
      assert.equal(submissions.length, 5);
      for (const modelType of ['flux3_klein', 'flux3_klein_base']) {
        for (const referenceCount of [0, 2]) {
          await configure(modelType, false, 2, referenceCount);
          const before = uploads.length;
          await page.evaluate(() => window.store.getState().startGeneration());
          const submitted = submissions.at(-1);
          assert.equal(submitted.model_type, modelType);
          assert.equal(Object.hasOwn(submitted, 'model_mode'), false, 'Drop stale generic music mode for FLUX 3');
          assert.equal(submitted.image_mode, 1, 'Use the real image workflow selector');
          assert.equal(await page.evaluate(() => window.store.getState().params.model_mode), 2,
            'Keep saved model preferences intact');
          if (referenceCount) {
            assert.equal(submitted.video_prompt_type, 'I', 'Keep FLUX 3 native ordered-reference serialization');
            assert.deepEqual(submitted.image_refs, uploads.slice(before), 'Keep ordered reference uploads');
          } else {
            assert.equal(submitted.video_prompt_type, '');
            assert.equal(submitted.image_refs, undefined);
          }
        }
      }
      assert.equal(submissions.length, 9);
      await configure('moss_image_edit', true);
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.length, 9, 'Do not submit with missing native reference capabilities');
      assert.match(await page.evaluate(() => window.store.getState().promptEnhanceError), /MOSS Image Edit reference settings/);
      await configure('fibo2_edit_turbo');
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.at(-1).video_prompt_type, 'KI', 'Switching back preserves Fibo behavior');
      assert.equal(submissions.at(-1).model_mode, 2, 'Switching back preserves Fibo selector fields');
      assert.equal(errors.length, 0, errors.join('\n'));
      await context.close();
    }
    console.log('Desktop/phone MOSS and FLUX 3 submissions, stale model modes, ordered refs, text-only generation and Fibo isolation passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
