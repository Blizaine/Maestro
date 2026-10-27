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
      async function configure(modelType, missingChoices = false) {
        await page.evaluate(({modelType, missingChoices}) => {
          const moss = modelType.startsWith('moss_');
          const models = ['fibo2_edit_turbo', 'moss_image_edit', 'moss_image_edit_hq'].map(id => ({
            model_type: id, architecture: id.startsWith('moss_') ? 'moss_image' : id,
            family: id.startsWith('moss_') ? 'moss_image' : 'fibo2',
            supports_image_edit: true, supports_instruction_image_edit: true,
          }));
          window.store.setState(state => ({
            models, enabledModels: new Set(models.map(model => model.model_type)), modelVisibilityOverrides: {},
            generationMode: 'image', studioImageWorkflow: 'generate', studioVideoWorkflow: 'frames',
            isGenerating: false, jobs: [], enhanceOnGeneration: false, enhanceOnGenerationDefault: false,
            imageRefs: [new File(['first-image'], 'first.png', {type: 'image/png'}),
                        new File(['second-image'], 'second.png', {type: 'image/png'})],
            imageRefType: 'KI',
            modelOptions: {model_type: modelType, architecture: moss ? 'moss_image' : modelType,
              image_outputs: true, max_image_refs: 4, no_background_removal: true,
              image_ref_choices: {choices: missingChoices ? [] : [['Ordered references', moss ? 'I' : 'KI']], default: moss ? 'I' : 'KI'}},
            params: {...state.params, model_type: modelType, image_mode: 1, prompt: 'Replace the hands using reference 2.',
              resolution: '720x1280', num_inference_steps: moss ? 50 : 4,
              video_prompt_type: 'KI', image_refs: undefined, image_start: undefined, image_end: undefined,
              activated_loras: [], loras_multipliers: '', spatial_upsampling: '', temporal_upsampling: ''},
          }));
        }, {modelType, missingChoices});
      }
      await configure('fibo2_edit_turbo');
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.length, 1, 'Fibo control submits');
      assert.equal(submissions[0].video_prompt_type, 'KI', 'Keep native Fibo selector');
      for (const modelType of ['moss_image_edit', 'moss_image_edit_hq']) {
        await configure(modelType);
        const before = uploads.length;
        await page.evaluate(() => window.store.getState().startGeneration());
        const submitted = submissions.at(-1);
        assert.equal(submitted.model_type, modelType);
        assert.equal(submitted.video_prompt_type, 'I', 'Drop the previous model’s stale K flag');
        assert.deepEqual(submitted.image_refs, uploads.slice(before), 'Preserve all references in upload order');
        assert.equal(submitted.image_refs.length, 2);
        assert.equal(submitted.image_mode, 1);
        assert.equal(submitted.resolution, '720x1280');
      }
      assert.equal(submissions.length, 3);
      await configure('moss_image_edit', true);
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.length, 3, 'Do not submit with missing native reference capabilities');
      assert.match(await page.evaluate(() => window.store.getState().promptEnhanceError), /MOSS Image Edit reference settings/);
      await configure('fibo2_edit_turbo');
      await page.evaluate(() => window.store.getState().startGeneration());
      assert.equal(submissions.at(-1).video_prompt_type, 'KI', 'Switching back preserves Fibo behavior');
      assert.equal(errors.length, 0, errors.join('\n'));
      await context.close();
    }
    console.log('Desktop/phone Fibo→MOSS SFT/HQ→Fibo submissions, ordered refs and capability guard passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
