// Exercise the real Director UI, store actions, planner requests and project
// restore path. Every network request is intercepted; no settings or jobs are
// written to the running Maestro app.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const lyrics = [{start: 0, end: 2.4, text: 'Follow the light home', speaker: 'speaker_1'}];
const analysis = {
  duration: 12,
  sample_rate: 44100,
  bpm: 120,
  beats: [],
  downbeats: [],
  sections: [{start: 0, end: 12, label: 'verse', energy: 0.5}],
  onset_envelope: [],
  lyrics,
  vocals_path: null,
};
const clips = [{
  start: 0,
  end: 12,
  section_label: 'verse',
  energy: 0.5,
  suggested_prompt_hint: '',
  beat_count: 8,
  duration_frames: 288,
}];
const plan = {image_prompt: 'A person follows a lantern.', video_prompt: 'The camera follows.'};
const savedPipeline = (snapshot, params = {}) => ({
  version: 2,
  pipeline_id: 'saved-project',
  project_id: 'saved-project',
  created_at: 0,
  completed_at: 0,
  status: 'completed',
  pipeline_type: 'music_video',
  scene_description: 'A quiet night walk.',
  reference_image_path: null,
  auto_mode: false,
  seamless: false,
  image_model: '',
  video_model: '',
  director_ui_snapshot: snapshot,
  _params_snapshot: params,
  llm_log: null,
  clips: [],
  output_files: [],
  total_time_sec: null,
});

(async () => {
  const bundle = await esbuild.build({
    absWorkingDir: path.join(root, 'ui'),
    stdin: {
      contents: [
        "import React from 'react'; import {createRoot} from 'react-dom/client';",
        "import {useStore} from './src/stores/useStore';",
        "import {DirectorChat} from './src/components/Sidebar/DirectorChat';",
        'window.store = useStore;',
        "window.mount = () => {window.root = createRoot(document.getElementById('root')); window.root.render(<DirectorChat/>);};",
      ].join('\n'),
      resolveDir: path.join(root, 'ui'),
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    jsx: 'automatic',
    tsconfig: path.join(root, 'ui', 'tsconfig.app.json'),
    nodePaths: [path.join(root, 'ui', 'node_modules')],
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });
  const browser = await playwright.chromium.launch({
    headless: true,
    ...(process.platform === 'win32'
      ? {executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}
      : {}),
  });
  try {
    const page = await browser.newPage();
    const writes = [];
    const requests = [];
    let currentSavedPipeline = savedPipeline({});
    page.on('pageerror', error => requests.push({pageError: error.message}));
    await page.route('**/*', async route => {
      const request = route.request();
      const endpoint = new URL(request.url()).pathname;
      const json = body => route.fulfill({json: body});
      if (endpoint === '/') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (request.method() !== 'GET') {
        const body = request.postDataJSON();
        writes.push({endpoint, body});
        if (endpoint === '/api/v1/director/v2/plan'
          || endpoint === '/api/v1/director/plan-prompts-and-images') {
          requests.push({endpoint, body});
          return json({clip_plans: [plan], planned_clips: clips});
        }
        if (endpoint === '/api/v1/director/queue') {
          requests.push({endpoint, body});
          return json({entries: [], running: false});
        }
        return json({});
      }
      if (endpoint === '/api/v1/director/pipelines/saved-project') {
        return json(currentSavedPipeline);
      }
      if (endpoint.includes('/model-options/')) {
        return json({
          model_type: decodeURIComponent(endpoint.split('/').pop()),
          fps: 24,
          frames_minimum: 5,
          frames_steps: 4,
          frames_maximum: 1000,
          default_num_inference_steps: 8,
          resolutions: ['1280x720'],
        });
      }
      if (endpoint === '/api/v1/director/music-clip-limits') {
        return json({fps: 24, frames_minimum: 5, frame_step: 4, hard_max_frames: 1000,
          recommended_frames: 240, max_frames: 1000, max_seconds: 40,
          recommended_seconds: 10, auto: true});
      }
      return json({});
    });

    await page.goto('http://director-lyrics-context.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(({analysis, clips}) => {
      const state = window.store.getState();
      state.directorReset();
      window.store.setState({
        sidebarMode: 'director',
        directorSkill: 'music_video',
        directorStep: 'style',
        directorLoading: false,
        directorAnalysis: analysis,
        directorPlannedClips: clips,
        directorClipPlans: [],
        directorClipImages: [],
        directorSceneDescription: 'A person follows a lantern through the night.',
        directorAudioPath: '/fixture/song.wav',
        directorShotImageGuidance: 'prompt_only',
        directorAutoMode: false,
        directorUseLyricsContext: false,
        servicesConfig: {use_director_v2: true},
      });
      window.mount();
    }, {analysis, clips});

    const checkbox = page.getByRole('checkbox', {name: 'Use lyrics to guide visuals'});
    await checkbox.waitFor();
    assert.equal(await checkbox.isChecked(), false, 'new music-video projects start with lyric context off');
    assert.equal(await page.getByText(
      'Let the lyrics influence scenes and story. Beat timing and lip-sync remain enabled either way.',
      {exact: true},
    ).count(), 1, 'help explains that timing and lip-sync remain enabled');

    await checkbox.check();
    await page.evaluate(async () => { await window.store.getState().directorPlanPrompts(); });
    const v2Request = requests.find(request => request.endpoint === '/api/v1/director/v2/plan');
    assert.equal(v2Request.body.use_lyrics_context, true, 'v2 visual planning receives the enabled setting');
    assert.deepEqual(v2Request.body.lyrics, lyrics, 'lyric timing and speakers remain in the request');

    await checkbox.uncheck();
    assert.equal(await page.evaluate(() => window.store.getState().directorClipPlans.length), 0,
      'changing the setting invalidates reviewed visual prompts');
    assert.equal(await page.evaluate(() => window.store.getState().directorStep), 'style',
      'the user returns to the explicit visual planning setup');
    await page.evaluate(() => window.store.setState({servicesConfig: {use_director_v2: false}}));
    await page.evaluate(async () => { await window.store.getState().directorPlanPrompts(); });
    const legacyRequest = requests.find(request => request.endpoint === '/api/v1/director/plan-prompts-and-images');
    assert.equal(legacyRequest.body.use_lyrics_context, false, 'legacy visual planning receives the disabled setting');
    assert.deepEqual(legacyRequest.body.lyrics, lyrics, 'legacy planning retains the lyric payload');

    await checkbox.check();
    await page.evaluate(async () => { await window.store.getState().startDirectorPipeline('queue'); });
    const queueRequest = requests.find(request => request.endpoint === '/api/v1/director/queue');
    assert.equal(queueRequest.body.params.use_lyrics_context, true, 'queued music projects carry the transport field');
    assert.equal(queueRequest.body.params.director_ui_snapshot.directorUseLyricsContext, true,
      'the project snapshot remembers the setting');
    assert.deepEqual(queueRequest.body.params.lyrics, lyrics, 'pipeline submission retains analyzed lyrics');

    currentSavedPipeline = savedPipeline({
      directorSkill: 'music_video',
      directorStep: 'review_video',
      directorUseLyricsContext: true,
      directorAnalysis: analysis,
      directorPlannedClips: clips,
      directorClipPlans: [plan],
    }, {use_lyrics_context: false});
    await page.evaluate(async () => { await window.store.getState().loadDirectorFromPipeline('saved-project'); });
    assert.equal(await page.evaluate(() => window.store.getState().directorUseLyricsContext), true,
      'the saved UI snapshot takes precedence when restoring a project');
    assert.equal(await checkbox.isChecked(), true, 'the restored preference is reflected in the checkbox');
    const writesBeforeSavedToggle = writes.length;
    await checkbox.uncheck();
    assert.equal(await page.evaluate(() => window.store.getState().directorClipPlans.length), 0,
      'changing a restored project invalidates its local cached prompts');
    assert.equal(currentSavedPipeline.director_ui_snapshot.directorClipPlans.length, 1,
      'editing the next setup does not mutate the saved rendered revision');
    assert.equal(writes.length, writesBeforeSavedToggle, 'toggling does not submit or mutate a saved project');

    currentSavedPipeline = savedPipeline({}, {use_lyrics_context: true});
    await page.evaluate(async () => { await window.store.getState().loadDirectorFromPipeline('saved-project'); });
    assert.equal(await page.evaluate(() => window.store.getState().directorUseLyricsContext), true,
      'saved request params restore the setting when no UI snapshot value exists');
    currentSavedPipeline = savedPipeline({}, {
      director_ui_snapshot: {directorUseLyricsContext: true},
    });
    await page.evaluate(async () => { await window.store.getState().loadDirectorFromPipeline('saved-project'); });
    assert.equal(await page.evaluate(() => window.store.getState().directorUseLyricsContext), true,
      'a UI snapshot stored inside saved params is also restored');
    currentSavedPipeline = savedPipeline({});
    await page.evaluate(async () => { await window.store.getState().loadDirectorFromPipeline('saved-project'); });
    assert.equal(await page.evaluate(() => window.store.getState().directorUseLyricsContext), false,
      'legacy saved projects missing the field default to off');
    await page.evaluate(({analysis, clips}) => window.store.setState({
      directorSkill: 'short_film',
      shortFilmPath: 'audio',
      directorStep: 'style',
      directorAnalysis: analysis,
      directorPlannedClips: clips,
    }), {analysis, clips});
    assert.equal(await checkbox.count(), 0, 'short-film analysis does not show the music-video setting');
    assert.deepEqual(requests.filter(request => request.pageError), [], 'the Director UI reports no browser errors');
    console.log('Director lyric context: default, v2 and legacy requests, pipeline save, project restore, and plan invalidation passed');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
