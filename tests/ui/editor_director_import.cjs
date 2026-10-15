// Run after UI dependencies are installed: node tests/ui/editor_director_import.cjs
// This mounts the real Editor Zustand store and intercepts every request. It
// never contacts the running Maestro app or reads/writes a saved Editor project.
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const workspace = 'Kitten-Music-Video';
const clipNames = Array.from({length: 9}, (_, index) => `director-shot-${index + 1}.mp4`);
const joinedName = 'director-final_multiclip.mp4';
const masterAudioName = 'director-master.wav';

function makePipeline({
  id,
  seamless = false,
  completedClips = 9,
  includeJoin = true,
  joinFilename = joinedName,
}) {
  const clips = Array.from({length: 9}, (_, index) => ({
    index,
    planned_clip: {start: index * 5, end: (index + 1) * 5, duration_frames: 120},
    video_prompt: `Rendered shot prompt ${index + 1}`,
    window_prompts: [`Rendered shot prompt ${index + 1}`],
    video_filename: index < completedClips ? clipNames[index] : null,
  }));
  const outputFiles = clipNames.slice(0, completedClips);
  if (includeJoin) outputFiles.push(joinFilename);
  return {
    pipeline_id: id,
    status: completedClips === 9 ? 'completed' : 'cancelled',
    pipeline_type: 'music_video',
    workspace,
    scene_description: `Synthetic ${id}`,
    seamless,
    clips,
    output_files: outputFiles,
    asset_manifest: {
      audio_path: {path: `audio/${masterAudioName}`, serve_path: masterAudioName},
    },
  };
}

function makeProject(id) {
  return {
    schema_version: 5,
    id,
    name: 'Isolated Director import test',
    workspace: 'editor-test-workspace',
    created_at: 1,
    updated_at: 1,
    canvas: {width: 1920, height: 1080, fps: 24, background: '#000000'},
    assets: {},
    tracks: [
      {id: `${id}-video`, name: 'Main video', type: 'video', z_index: 0, muted: false, locked: false, volume: 1, items: []},
      {id: `${id}-audio`, name: 'Audio', type: 'audio', z_index: 0, muted: false, locked: false, volume: 1, items: []},
      {id: `${id}-titles`, name: 'Titles', type: 'text', z_index: 10, muted: false, locked: false, items: []},
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
    absWorkingDir: path.join(root, 'ui'),
    stdin: {
      contents: `
        import { useEditorStore } from './src/editor/useEditorStore';
        window.editorStore = useEditorStore;
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
  const page = await browser.newPage();
  let pipeline = null;
  const pipelineRequests = [];
  const probeRequests = [];
  const unexpectedRequests = [];
  const projectWrites = [];
  const unavailableMedia = new Set();

  try {
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body, status = 200) => route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });

      if (url.pathname === '/' || url.pathname === '/index.html') {
        return route.fulfill({contentType: 'text/html', body: '<div id="root"></div>'});
      }
      if (url.pathname === '/favicon.ico') return route.fulfill({status: 204, body: ''});
      if (url.pathname.startsWith('/api/')) {
        if (url.pathname === `/api/v1/director/pipelines/${pipeline?.pipeline_id}` && request.method() === 'GET') {
          pipelineRequests.push(pipeline.pipeline_id);
          return json(pipeline);
        }
        if (url.pathname === '/api/v1/editor/media/probe' && request.method() === 'POST') {
          const body = request.postDataJSON();
          const asset = body.asset || {};
          probeRequests.push({
            name: asset.name,
            origin: asset.origin,
            assetWorkspace: asset.workspace,
            requestWorkspace: body.workspace,
          });
          if (unavailableMedia.has(asset.name)) {
            return json({detail: 'Synthetic media unavailable'}, 404);
          }
          const isAudio = asset.name === masterAudioName;
          const isJoined = asset.name === joinedName || asset.name === 'seamless-final.mp4';
          return json({
            name: asset.name,
            type: isAudio ? 'audio' : 'video',
            duration: isAudio || isJoined ? 45 : 5,
            width: isAudio ? 0 : 1280,
            height: isAudio ? 0 : 720,
            fps: isAudio ? 0 : 24,
            has_audio: true,
            audio_channels: isAudio ? 2 : 1,
            audio_sample_rate: 48000,
            size: 1024,
            path: `/isolated/${asset.name}`,
          });
        }
        if (url.pathname === '/api/v1/outputs') {
          // The normal gallery is intentionally represented as final-only.
          return json({outputs: [{name: joinedName, type: 'video'}], total: 1});
        }
        if (url.pathname.startsWith('/api/v1/editor/projects')) {
          projectWrites.push(`${request.method()} ${url.pathname}`);
          return json({detail: 'Project persistence is forbidden in this test'}, 500);
        }
        unexpectedRequests.push(`${request.method()} ${url.pathname}`);
        return json({detail: 'Unexpected API request blocked by isolated test'}, 500);
      }
      unexpectedRequests.push(`${request.method()} ${url.origin}${url.pathname}`);
      return route.abort('blockedbyclient');
    });

    await page.goto('http://editor-director-import.test');
    await page.addScriptTag({content: bundle.outputFiles[0].text});

    async function runImport(nextPipeline, projectId, missingMedia = []) {
      pipeline = nextPipeline;
      unavailableMedia.clear();
      missingMedia.forEach(name => unavailableMedia.add(name));
      const project = makeProject(projectId);
      const galleryFilename = nextPipeline.output_files.find(filename => (
        filename.toLowerCase().includes('_multiclip')
      )) || nextPipeline.output_files.at(-1) || joinedName;
      const galleryAsset = {
        id: `gallery-${nextPipeline.pipeline_id}`,
        name: galleryFilename,
        type: 'video',
        origin: 'output',
        workspace,
        url: `/mock/${galleryFilename}`,
        duration: 45,
        width: 1280,
        height: 720,
        fps: 24,
        has_audio: true,
      };
      await page.evaluate(({project, run}) => {
        window.editorStore.setState({
          workspace: project.workspace,
          project,
          library: [run.galleryAsset],
          directorRuns: [{
            id: run.pipeline_id,
            workspace: run.workspace,
            scene_description: run.scene_description,
          }],
          directorImportingId: null,
          dirty: false,
          error: null,
          history: [],
          future: [],
          playhead: 0,
        });
      }, {project, run: {...nextPipeline, galleryAsset}});
      const galleryNamesBefore = await page.evaluate(() => (
        window.editorStore.getState().library.map(asset => asset.name)
      ));
      await page.evaluate(id => window.editorStore.getState().importDirectorRun(id), nextPipeline.pipeline_id);
      return page.evaluate(galleryNamesBefore => {
        const state = window.editorStore.getState();
        const project = state.project;
        const videoTrack = project.tracks.find(track => (
          track.type === 'video' && track.name.startsWith('Director · ')
        ));
        const audioTrack = project.tracks.find(track => (
          track.type === 'audio' && track.name.startsWith('Director music · ')
        ));
        return {
          galleryNamesBefore,
          assets: Object.values(project.assets).map(asset => ({
            id: asset.id,
            name: asset.name,
            type: asset.type,
            workspace: asset.workspace,
            duration: asset.duration,
          })),
          videoTrack: videoTrack && {
            name: videoTrack.name,
            items: videoTrack.items.map(item => ({
              id: item.id,
              asset_id: item.asset_id,
              name: item.name,
              start: item.start,
              duration: item.duration,
              source_in: item.source_in,
              director: item.director,
            })),
          },
          audioTrack: audioTrack && {
            name: audioTrack.name,
            items: audioTrack.items.map(item => ({
              asset_id: item.asset_id,
              start: item.start,
              duration: item.duration,
            })),
          },
          error: state.error,
          dirty: state.dirty,
        };
      }, galleryNamesBefore);
    }

    const individualRun = makePipeline({id: 'run-nine-clips'});
    const individual = await runImport(individualRun, 'editor-individual-test');
    assert.deepEqual(individual.galleryNamesBefore, [joinedName], 'The normal Editor library has only the joined output');
    assert.equal(individual.videoTrack.items.length, 9, 'All nine saved individual clips become timeline items');
    assert.equal(individual.assets.filter(asset => asset.type === 'video').length, 9, 'Every clip remains an individual Editor asset');
    assert.equal(new Set(individual.videoTrack.items.map(item => item.asset_id)).size, 9, 'Each shot has a distinct media source');
    assert.deepEqual(individual.videoTrack.items.map(item => item.director.clip_index), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert.deepEqual(individual.videoTrack.items.map(item => item.director.video_prompt), clipNames.map((_, index) => `Rendered shot prompt ${index + 1}`));
    assert.deepEqual(individual.videoTrack.items.map(item => item.source_in), Array(9).fill(0));
    assert.equal(individual.audioTrack.items.length, 1, 'The saved master soundtrack is added on its own track');
    assert.equal(individual.assets.find(asset => asset.type === 'audio')?.name, masterAudioName);
    assert.equal(individual.error, null);
    console.log('Final-only gallery state still imports nine individual Director clips and master audio');

    const seamlessRun = makePipeline({
      id: 'run-seamless-master',
      seamless: true,
      completedClips: 0,
      includeJoin: true,
      joinFilename: 'seamless-final.mp4',
    });
    const seamless = await runImport(seamlessRun, 'editor-seamless-test');
    assert.equal(seamless.videoTrack.items.length, 9, 'Every planned shot remains independently editable');
    assert.equal(seamless.assets.filter(asset => asset.type === 'video').length, 1, 'The cumulative render is stored once');
    assert.equal(new Set(seamless.videoTrack.items.map(item => item.asset_id)).size, 1, 'Shots share the seamless master asset');
    assert.deepEqual(seamless.videoTrack.items.map(item => item.source_in), [0, 5, 10, 15, 20, 25, 30, 35, 40]);
    assert.deepEqual(seamless.videoTrack.items.map(item => item.duration), Array(9).fill(5));
    assert.deepEqual(seamless.videoTrack.items.map(item => item.director.clip_index), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert.equal(seamless.audioTrack.items.length, 1);
    assert.equal(seamless.error, null);
    console.log('Seamless Director import maps each editable shot to its master source range');

    const cancelledRun = makePipeline({id: 'run-cancelled-prefix', completedClips: 4, includeJoin: false});
    const cancelled = await runImport(cancelledRun, 'editor-cancelled-test', [clipNames[1]]);
    assert.equal(cancelled.videoTrack.items.length, 3, 'Only surviving completed clips are imported when one saved file is missing');
    assert.equal(cancelled.assets.filter(asset => asset.type === 'video').length, 3, 'Every surviving clip remains a separate source');
    assert.deepEqual(cancelled.videoTrack.items.map(item => item.director.clip_index), [0, 2, 3]);
    assert.deepEqual(cancelled.videoTrack.items.map(item => item.director.video_prompt), [1, 3, 4].map(index => `Rendered shot prompt ${index}`));
    assert.deepEqual(cancelled.assets.filter(asset => asset.type === 'video').map(asset => asset.name), [clipNames[0], clipNames[2], clipNames[3]]);
    assert.equal(cancelled.audioTrack.items.length, 1);
    assert.equal(cancelled.error, 'Imported 3 of 9 completed Director shots.');
    console.log('Cancelled Director import keeps its completed per-clip prefix without a joined master');

    assert.equal(pipelineRequests.length, 3, 'Each import fetches only its saved pipeline state');
    assert.equal(projectWrites.length, 0, 'No Editor project endpoint is written');
    assert.equal(unexpectedRequests.length, 0, `No unexpected requests escape the mocks: ${unexpectedRequests.join(', ')}`);
    assert.equal(probeRequests.every(request => request.assetWorkspace === workspace), true, 'All probes preserve the source workspace');
    console.log('All requests stayed on isolated mocks; no live project or generation endpoint was touched');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
