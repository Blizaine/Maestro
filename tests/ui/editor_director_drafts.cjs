// Run after the UI dependencies are installed: node tests/ui/editor_director_drafts.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const esbuild = require(path.join(__dirname, '../../ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

function makeProject() {
  return {
    schema_version: 5,
    id: 'project-director-drafts',
    name: 'Director drafts',
    workspace: 'draft-workspace',
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {
      source: {
        id: 'source', name: 'source.mp4', type: 'video', origin: 'output',
        workspace: 'draft-workspace', path: '/outputs/source.mp4', url: '/source.mp4',
        duration: 8, width: 1280, height: 720, fps: 24, has_audio: true, size: 100,
      },
    },
    tracks: [
      {
        id: 'video-main', name: 'Main video', type: 'video', z_index: 0,
        muted: false, locked: false,
        items: [{
          id: 'director-shot', asset_id: 'source', name: 'Shot 1', start: 0,
          duration: 5, source_in: 0, speed: 1, volume: 1, opacity: 1,
          fit: 'cover', transform: {x: 0, y: 0, scale: 1, rotation: 0},
          take_asset_ids: ['source'], take_states: {source: {source_in: 0, speed: 1}},
          director: {
            pipeline_id: 'pipeline-draft-test', clip_index: 0, pipeline_type: 'music_video',
            workspace: 'draft-workspace', video_prompt: 'Original rendered prompt',
            window_prompts: ['Original rendered prompt'],
          },
        }, {
          id: 'director-shot-2', asset_id: 'source', name: 'Shot 2', start: 5,
          duration: 3, source_in: 0, speed: 1, volume: 1, opacity: 1,
          fit: 'cover', transform: {x: 0, y: 0, scale: 1, rotation: 0},
          take_asset_ids: ['source'], take_states: {source: {source_in: 0, speed: 1}},
          director: {
            pipeline_id: 'pipeline-draft-test', clip_index: 1, pipeline_type: 'music_video',
            workspace: 'draft-workspace', video_prompt: 'Second original prompt',
            window_prompts: ['Second original prompt'],
          },
        }],
      },
      {id: 'audio-main', name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false, items: []},
      {id: 'titles-main', name: 'Titles', type: 'text', z_index: 10, muted: false, locked: false, items: []},
    ],
    markers: [],
    export: {
      quality: 'high', codec: 'h264', encoder: 'auto', include_audio: true,
      resolution: 'canvas', frame_rate: 'project', filename: '',
      spatial_upsampling: '', film_grain_intensity: 0, film_grain_saturation: 0.5,
    },
    exports: [],
  };
}

(async () => {
  const bundle = await esbuild.build({
    stdin: {
      contents: `
        import React from 'react';
        import {createRoot} from 'react-dom/client';
        import {EditorInspector} from './src/editor/EditorInspector';
        import {useEditorStore} from './src/editor/useEditorStore';
        window.store = useEditorStore;
        function App() {
          const selectedItemId = useEditorStore(state => state.selectedItemId);
          return <div onPointerDown={event => {
            if (event.target.closest('[role="alert"]')) {
              useEditorStore.setState({selectedItemId: null, selectedItemIds: []});
            }
          }}>
            <span data-testid="selected-item">{selectedItemId || 'none'}</span>
            <EditorInspector compact />
          </div>;
        }
        const root = createRoot(document.getElementById('root'));
        window.mount = () => root.render(<App />);
      `,
      resolveDir: path.resolve(__dirname, '../../ui'),
      loader: 'tsx',
    },
    bundle: true,
    write: false,
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"development"'},
    logLevel: 'silent',
  });

  const launchOptions = {headless: true};
  if (process.env.MAESTRO_CHROME) launchOptions.executablePath = process.env.MAESTRO_CHROME;
  else if (process.platform === 'win32') launchOptions.channel = 'chrome';
  const browser = await playwright.chromium.launch(launchOptions);
  const page = await browser.newPage({viewport: {width: 1000, height: 800}});
  const projectId = 'project-director-drafts';
  let savedProject = makeProject();
  let rerunMode = 'fail';
  let nextSaveGate = null;
  let nextFinalSaveGate = null;
  let nextRerunGate = null;
  const saves = [];
  const reruns = [];

  try {
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body, status = 200) => route.fulfill({
        status, contentType: 'application/json', body: JSON.stringify(body),
      });

      if (url.pathname === `/api/v1/editor/projects/${projectId}` && request.method() === 'GET') {
        return json(savedProject);
      }
      if (url.pathname === `/api/v1/editor/projects/${projectId}` && request.method() === 'PUT') {
        savedProject = JSON.parse(JSON.stringify(request.postDataJSON().project));
        saves.push(savedProject);
        if (nextSaveGate) {
          const gate = nextSaveGate;
          nextSaveGate = null;
          gate.entered();
          await gate.wait;
        }
        if (nextFinalSaveGate) {
          const gate = nextFinalSaveGate;
          nextFinalSaveGate = null;
          gate.entered();
          await gate.wait;
        }
        return json(savedProject);
      }
      if (url.pathname === '/api/v1/editor/projects' && request.method() === 'GET') {
        return json({projects: [{
          id: projectId, name: savedProject.name, workspace: savedProject.workspace,
          created_at: savedProject.created_at, updated_at: savedProject.updated_at,
          duration: 5, asset_count: Object.keys(savedProject.assets).length,
        }]});
      }
      if (url.pathname === '/api/v1/editor/media/status') {
        return json({assets: Object.keys(savedProject.assets).map(asset_id => ({
          asset_id, available: true, path: savedProject.assets[asset_id].path,
        }))});
      }
      if (url.pathname === '/api/v1/director/pipelines/pipeline-draft-test/clips/0/rerun-video') {
        reruns.push(request.postDataJSON());
        if (rerunMode === 'fail') return json({error: 'Director test failure'}, 500);
        if (nextRerunGate) {
          const gate = nextRerunGate;
          nextRerunGate = null;
          gate.entered();
          await gate.wait;
        }
        return json({filename: 'rerun.mp4', clip_index: 0});
      }
      if (url.pathname === '/api/v1/outputs') {
        return json({outputs: [{
          id: 'rerun-output', path: '/outputs/rerun.mp4', name: 'rerun.mp4', type: 'video',
          mode: 'video', size: 200, created_at: 2, url: '/rerun.mp4', workspace: 'draft-workspace',
        }], total: 1});
      }
      if (url.pathname === '/api/v1/editor/media/probe') {
        return json({
          name: 'rerun.mp4', type: 'video', duration: 5, width: 1280, height: 720,
          fps: 24, has_audio: true, audio_channels: 2, audio_sample_rate: 48000,
          size: 200, path: '/outputs/rerun.mp4',
        });
      }
      if (url.pathname.startsWith('/api/')) return json({});
      return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
    });

    await page.goto('http://editor-director-drafts.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});
    await page.evaluate(project => {
      window.store.setState({
        workspace: project.workspace,
        project,
        selectedItemId: 'director-shot',
        selectedItemIds: ['director-shot'],
        selectedTrackId: 'video-main',
        dirty: false,
        error: null,
      });
      window.mount();
    }, makeProject());

    const prompt = '  Revised prompt with exact spacing.\nKeep both lines.  ';
    const promptDraft = page.getByLabel('Video prompt draft');
    await promptDraft.fill(prompt);
    assert.equal(await promptDraft.inputValue(), prompt);
    assert.deepEqual(await page.evaluate(() => {
      const item = window.store.getState().project.tracks[0].items[0];
      return [item.director.video_prompt, item.director.draft_video_prompt];
    }), ['Original rendered prompt', prompt], 'Editing the draft leaves the last rendered prompt intact');
    assert.equal(await page.evaluate(() => window.store.getState().dirty), true,
      'Draft edits participate in normal project saving');

    await page.evaluate(() => window.store.getState().saveProject());
    await page.waitForFunction(() => window.store.getState().dirty === false);
    assert.equal(savedProject.tracks[0].items[0].director.video_prompt, 'Original rendered prompt');
    assert.equal(savedProject.tracks[0].items[0].director.draft_video_prompt, prompt);

    await page.evaluate(() => window.store.setState({selectedItemId: null, selectedItemIds: []}));
    await page.evaluate(() => window.store.setState({selectedItemId: 'director-shot', selectedItemIds: ['director-shot']}));
    assert.equal(await page.getByLabel('Video prompt draft').inputValue(), prompt,
      'Deselecting and reselecting restores the stored draft');

    let releaseSave;
    let signalSaveEntered;
    const saveEntered = new Promise(resolve => { signalSaveEntered = resolve; });
    nextSaveGate = {
      entered: signalSaveEntered,
      wait: new Promise(resolve => { releaseSave = resolve; }),
    };
    await page.getByRole('button', {name: 'Re-run Director shot'}).click();
    await saveEntered;
    assert.equal(await page.evaluate(() => window.store.getState().directorRerunItemId), 'director-shot',
      'The shot is locked before its draft save starts');
    await page.evaluate(() => window.store.getState().rerunDirectorClip('director-shot', 'duplicate request'));
    assert.equal(reruns.length, 0, 'A second submit during the draft save is ignored');
    releaseSave();
    const alert = page.getByRole('alert');
    await alert.waitFor();
    if (process.env.MAESTRO_EDITOR_DRAFT_SCREENSHOT) {
      fs.mkdirSync(path.dirname(process.env.MAESTRO_EDITOR_DRAFT_SCREENSHOT), {recursive: true});
      await page.getByLabel('Video prompt draft').locator('xpath=../../..').screenshot({
        path: process.env.MAESTRO_EDITOR_DRAFT_SCREENSHOT,
      });
    }
    await page.waitForFunction(() => window.store.getState().directorRerunItemId === null);
    assert.equal(reruns[0].prompt, prompt, 'Rerun sends the exact draft text');
    assert.equal(await page.evaluate(() => window.store.getState().error), null,
      'A rerun failure is kept out of the global error popup');
    assert.equal(await page.getByTestId('selected-item').textContent(), 'director-shot');

    await page.evaluate(() => window.store.setState({selectedItemId: null, selectedItemIds: []}));
    await page.evaluate(() => window.store.setState({selectedItemId: 'director-shot', selectedItemIds: ['director-shot']}));
    assert.equal(await page.getByRole('alert').count(), 1, 'The shot-local failure survives reselection');
    assert.equal(await page.getByLabel('Video prompt draft').inputValue(), prompt,
      'A failed rerun leaves the draft intact');

    await page.evaluate(async () => window.store.getState().loadProject('project-director-drafts'));
    await page.evaluate(() => window.store.setState({selectedItemId: 'director-shot', selectedItemIds: ['director-shot']}));
    assert.equal(await page.getByLabel('Video prompt draft').inputValue(), prompt,
      'The saved draft survives reloading the project');
    assert.equal(await page.getByRole('alert').count(), 1,
      'The local failure remains available after reloading the same project');
    await page.getByRole('alert').getByRole('button', {name: 'Dismiss'}).click();
    assert.equal(await page.getByRole('alert').count(), 0);
    assert.equal(await page.getByTestId('selected-item').textContent(), 'director-shot',
      'Using the local error control does not deselect the shot');

    const blankDraft = ' \n  ';
    await page.getByLabel('Video prompt draft').fill(blankDraft);
    await page.getByRole('button', {name: 'Re-run Director shot'}).click();
    await page.getByRole('alert').waitFor();
    assert.match(await page.getByRole('alert').textContent(), /Enter a video prompt/);
    assert.equal(reruns.length, 1, 'An empty prompt is rejected before contacting Director');
    await page.evaluate(() => window.store.getState().saveProject());
    await page.waitForFunction(() => window.store.getState().dirty === false);
    assert.equal(savedProject.tracks[0].items[0].director.draft_video_prompt, blankDraft,
      'A blank draft still participates in normal project saves');
    assert.equal(savedProject.tracks[0].items[0].director.video_prompt, 'Original rendered prompt',
      'Saving a blank draft does not relabel the last rendered prompt');

    const newerDraft = 'Typed while generation is pending';
    const newerSecondDraft = 'Second shot draft typed during the final save';
    let releaseRerun;
    let signalRerunEntered;
    const rerunEntered = new Promise(resolve => { signalRerunEntered = resolve; });
    nextRerunGate = {
      entered: signalRerunEntered,
      wait: new Promise(resolve => { releaseRerun = resolve; }),
    };
    rerunMode = 'success';
    await page.getByLabel('Video prompt draft').fill(prompt);
    await page.getByRole('button', {name: 'Re-run Director shot'}).click();
    await rerunEntered;
    await page.getByLabel('Video prompt draft').fill(newerDraft);
    let releaseFinalSave;
    let signalFinalSaveEntered;
    const finalSaveEntered = new Promise(resolve => { signalFinalSaveEntered = resolve; });
    nextFinalSaveGate = {
      entered: signalFinalSaveEntered,
      wait: new Promise(resolve => { releaseFinalSave = resolve; }),
    };
    releaseRerun();
    await finalSaveEntered;
    await page.evaluate(() => window.store.setState({
      selectedItemId: 'director-shot-2', selectedItemIds: ['director-shot-2'],
    }));
    await page.getByLabel('Video prompt draft').fill(newerSecondDraft);
    releaseFinalSave();
    await page.waitForFunction(() => window.store.getState().directorRerunItemId === null
      && window.store.getState().project?.tracks[0]?.items[0]?.asset_id !== 'source');
    const rendered = await page.evaluate(() => {
      const items = window.store.getState().project.tracks[0].items;
      const item = items[0];
      return {
        assetId: item.asset_id,
        takes: item.take_asset_ids,
        actualPrompt: item.director.video_prompt,
        draftPrompt: item.director.draft_video_prompt,
        secondDraftPrompt: items[1].director.draft_video_prompt,
        dirty: window.store.getState().dirty,
      };
    });
    assert.ok(rendered.takes.includes('source'), 'The original take remains available');
    assert.equal(rendered.actualPrompt, prompt, 'The successful render records the exact prompt used');
    assert.equal(rendered.draftPrompt, newerDraft, 'A newer draft typed during generation survives success');
    assert.equal(rendered.secondDraftPrompt, newerSecondDraft,
      'A different shot draft typed during the final save survives success');
    assert.equal(rendered.dirty, true, 'Drafts merged after the final save remain dirty for autosave');
    await page.evaluate(() => window.store.getState().saveProject());
    await page.waitForFunction(() => window.store.getState().dirty === false);
    assert.equal(savedProject.tracks[0].items[1].director.draft_video_prompt, newerSecondDraft,
      'The merged draft is persisted by the following normal save');
    assert.equal(await page.getByRole('alert').count(), 0, 'A successful retry clears its previous error');
    assert.ok(saves.length >= 3, 'Normal save and rerun persistence both reached the project API');
    console.log('Editor Director drafts survive edits, rerun failures, saves, reloads, and successful reruns.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
