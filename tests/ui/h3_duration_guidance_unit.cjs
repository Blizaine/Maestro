// Run with: node tests/ui/h3_duration_guidance_unit.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));

function options(overrides = {}) {
  return {
    model_type: 'minimax_h3_unit',
    architecture: 'minimax_h3',
    audio_only: false,
    frames_minimum: 124,
    frames_steps: 17,
    omni_reference: false,
    sliding_window_memory_policy: {
      auto_resolution_pixels: { '720p': 1280 * 704 },
      resolution_bands: [{
        min_pixels: 0,
        vram_tiers: [
          { max_vram_gb: 16, frames: 243 },
          { max_vram_gb: null, frames: 345 },
        ],
      }],
    },
    ...overrides,
  };
}

function input(overrides = {}) {
  return {
    options: options(),
    resolution: '1280x704',
    frames: 243,
    totalVramGb: 24,
    prompt: 'A person walks through rain.',
    ...overrides,
  };
}

(async () => {
  function loadTypeScript(filePath, dependencies = {}) {
    const transformed = esbuild.transformSync(fs.readFileSync(filePath, 'utf8'), {
      loader: 'ts',
      format: 'cjs',
      target: 'node22',
    });
    const loaded = { exports: {} };
    const localRequire = dependency => Object.hasOwn(dependencies, dependency)
      ? dependencies[dependency]
      : require(dependency);
    new Function('module', 'exports', 'require', transformed.code)(
      loaded, loaded.exports, localRequire,
    );
    return loaded.exports;
  }

  const h3Memory = loadTypeScript(path.join(root, 'ui/src/lib/h3Memory.ts'));
  const {
    estimateH3DurationGuidance,
    H3_RMS_NORM_NATIVE_MAX_ROWS,
  } = loadTypeScript(path.join(root, 'ui/src/lib/h3DurationGuidance.ts'), {
    './h3Memory': h3Memory,
  });
  const h3OutputMaxPixels = 768 * 1344

  const transformer = fs.readFileSync(
    path.join(root, 'app/models/minimax_h3/transformer.py'), 'utf8',
  );
  const cutoff = transformer.match(/MINIMAX_H3_RMS_NORM_NATIVE_MAX_TOKENS\s*=\s*(\d+)/);
  assert.ok(cutoff, 'Transformer source declares its native RMSNorm cutoff');
  assert.equal(H3_RMS_NORM_NATIVE_MAX_ROWS, Number(cutoff[1]));

  const shortPass = estimateH3DurationGuidance(input());
  assert.equal(shortPass.supported, true);
  assert.equal(shortPass.path, 'fast', '243 frames at 1280x704 stay under the 75k RMSNorm cutoff');
  assert.equal(shortPass.rowsLow, 64178, 'target video is 72 latents × 40 × 22 rows plus 810 stereo audio rows and text');
  assert.ok(shortPass.rowsHigh >= shortPass.rowsLow);
  assert.equal(shortPass.recommendationFrames, 345);
  assert.equal(shortPass.fastMaxFrames, 277);
  assert.equal(shortPass.beyondRecommended, false);
  assert.equal(shortPass.risk, 'within-profile');

  const staleOmniPolicy = estimateH3DurationGuidance(input({
    options: options({
      omni_reference: false,
      sliding_window_memory_policy: {
        auto_resolution_pixels: { '720p': 1280 * 704 },
        resolution_bands: [{ min_pixels: 0, vram_tiers: [{ max_vram_gb: null, frames: 345 }] }],
      },
      omni_sequence_memory_policy: {
        auto_resolution_pixels: { '720p': 1280 * 704 },
        resolution_bands: [{ min_pixels: 0, vram_tiers: [{ max_vram_gb: null, frames: 124 }] }],
      },
    }),
  }));
  assert.equal(staleOmniPolicy.recommendationFrames, 345, 'a disabled Omni mode must ignore its stale policy field');

  const nativeLimit = estimateH3DurationGuidance(input({ frames: 345 }));
  assert.equal(nativeLimit.path, 'chunked', '345 frames exceed the native RMSNorm row cutoff at 1280x704');
  assert.equal(nativeLimit.recommendationFrames, 345);
  assert.equal(nativeLimit.beyondRecommended, false, '345 frames are the native duration boundary');
  assert.equal(nativeLimit.risk, 'within-profile', 'the 24 GB profile explicitly supports this native pass');

  const lowVram = estimateH3DurationGuidance(input({ frames: 345, totalVramGb: 16 }));
  assert.equal(lowVram.recommendationFrames, 243);
  assert.equal(lowVram.beyondRecommended, false, 'memory recommendation exceedance is separate from the native duration boundary');
  assert.equal(lowVram.risk, 'high');

  const withinNativeButOverProfile = estimateH3DurationGuidance(input({ frames: 328, totalVramGb: 16 }));
  assert.equal(withinNativeButOverProfile.beyondRecommended, false);
  assert.equal(withinNativeButOverProfile.risk, 'high');

  const extended = estimateH3DurationGuidance(input({ frames: 346 }));
  assert.equal(extended.beyondRecommended, true, 'the first frame count above 345 crosses the native envelope');
  assert.equal(extended.risk, 'high');
  assert.ok(extended.notes.some(note => /experimental extended range/.test(note)));

  const unknownVram = estimateH3DurationGuidance(input({ totalVramGb: 0 }));
  assert.equal(unknownVram.recommendationFrames, null);
  assert.equal(unknownVram.risk, 'unknown');

  const unsupportedTierOptions = options({
    sliding_window_memory_policy: {
      auto_resolution_pixels: { 'auto': 864 * 480 },
      resolution_bands: [{ min_pixels: 0, vram_tiers: [{ max_vram_gb: null, frames: null }] }],
    },
  });
  const unsupportedTier = estimateH3DurationGuidance(input({
    options: unsupportedTierOptions,
    resolution: 'auto',
  }));
  assert.equal(unsupportedTier.supported, true);
  assert.equal(unsupportedTier.path, 'uncertain', 'auto pixel budgets do not reveal the realized canvas aspect');
  assert.equal(unsupportedTier.rowsHigh, null);
  assert.equal(unsupportedTier.recommendationFrames, null);
  assert.equal(unsupportedTier.risk, 'high', 'an explicit unsupported model/resolution/VRAM tier is high risk');

  const unsupportedModel = estimateH3DurationGuidance(input({
    options: options({ audio_only: true }),
  }));
  assert.equal(unsupportedModel.supported, false);
  assert.equal(unsupportedModel.risk, 'unknown');
  assert.equal(unsupportedModel.rowsHigh, null);

  const videoReference = estimateH3DurationGuidance(input({
    references: [{
      id: 'video', type: 'video', path: 'video.mp4', filename: 'video.mp4',
      duration_seconds: 10, has_audio: false, include_audio: false,
    }],
  }));
  assert.notEqual(videoReference.path, 'fast', 'a 10-second video reference cannot retain a confident fast path');
  assert.equal(videoReference.risk, 'caution');
  assert.ok(videoReference.rowsHigh > videoReference.rowsLow);

  const matchImage = estimateH3DurationGuidance(input({
    references: [{ id: 'image', type: 'image', path: 'image.png', filename: 'image.png' }],
  }));
  const maxImage = estimateH3DurationGuidance(input({
    referenceDetail: 'max',
    references: [{ id: 'image', type: 'image', path: 'image.png', filename: 'image.png' }],
  }));
  assert.ok(maxImage.rowsHigh > matchImage.rowsHigh, 'max detail bounds more visual rows than match detail');
  assert.equal(matchImage.risk, 'caution');
  assert.ok(!matchImage.notes.some(note => /reference duration/i.test(note)), 'image references do not need duration metadata');

  const tinyCanvasImage = estimateH3DurationGuidance(input({
    resolution: '32x32',
    references: [{ id: 'tiny-image', type: 'image', path: 'tiny.png', filename: 'tiny.png' }],
  }));
  assert.ok(tinyCanvasImage.rowsHigh >= tinyCanvasImage.rowsLow, 'Qwen image resize minimum keeps tiny-canvas bounds ordered');

  const noReference = estimateH3DurationGuidance(input());
  const maxVideo = estimateH3DurationGuidance(input({
    referenceDetail: 'max',
    references: [{
      id: 'max-video', type: 'video', path: 'video.mp4', filename: 'video.mp4',
      duration_seconds: 2, has_audio: false, include_audio: false,
    }],
  }));
  const roundedVideoArea = h3OutputMaxPixels + 40 * Math.sqrt(h3OutputMaxPixels) + 256;
  const roundedVideoRowsPerFrame = Math.ceil(roundedVideoArea / (32 * 32));
  assert.ok(
    maxVideo.rowsHigh - noReference.rowsHigh >= 12 * roundedVideoRowsPerFrame,
    'max video references include a conservative 32px rounded 4:1 canvas bound',
  );

  const largeCanvasBase = estimateH3DurationGuidance(input({resolution: '1920x1088'}));
  const largeCanvasVideo = estimateH3DurationGuidance(input({
    resolution: '1920x1088', referenceDetail: 'match',
    references: [{id:'large-video',type:'video',path:'large.mp4',filename:'large.mp4',
      duration_seconds:2,has_audio:false,include_audio:false}],
  }));
  const largeArea = 1920 * 1088;
  const largeRowsPerFrame = Math.ceil((largeArea + 40 * Math.sqrt(largeArea) + 256) / 1024);
  assert.ok(largeCanvasVideo.rowsHigh - largeCanvasBase.rowsHigh >= 17 * largeRowsPerFrame,
    'Matched video vision bounds follow a 1080p output rather than the smaller max-detail native cap');

  const shortSound = estimateH3DurationGuidance(input({
    references: [{
      id: 'sound', type: 'audio', path: 'sound.wav', filename: 'sound.wav',
      audio_intent: 'sound', audio_duration_seconds: 0.2,
    }],
  }));
  assert.equal(shortSound.rowsLow - shortPass.rowsLow, 20, 'a known 0.2-second static sound adds 16 audio rows plus label rows, not a two-second minimum');

  const unknownVideoDuration = estimateH3DurationGuidance(input({
    references: [{
      id: 'unknown-video', type: 'video', path: 'video.mp4', filename: 'video.mp4',
      has_audio: false, include_audio: false,
    }],
  }));
  assert.equal(unknownVideoDuration.rowsHigh, null, 'unknown video duration prevents a confident upper bound');
  assert.equal(unknownVideoDuration.path, 'uncertain');

  const unknownAudioDuration = estimateH3DurationGuidance(input({
    references: [{
      id: 'unknown-audio', type: 'audio', path: 'audio.wav', filename: 'audio.wav',
      audio_intent: 'voice',
    }],
  }));
  assert.equal(unknownAudioDuration.rowsHigh, null, 'unknown audio duration prevents a confident upper bound');

  const refmod = estimateH3DurationGuidance(input({
    references: [{
      id: 'refmod', type: 'image', path: 'image.png', filename: 'image.png',
      refmod_path: 'latent.refmod',
    }],
  }));
  assert.equal(refmod.rowsHigh, null);
  assert.equal(refmod.path, 'uncertain');
  assert.equal(refmod.fastMaxFrames, null);

  const videoGuide = estimateH3DurationGuidance(input({ videoGuide: true }));
  assert.equal(videoGuide.rowsHigh, null);
  assert.equal(videoGuide.path, 'uncertain');
  assert.equal(videoGuide.fastMaxFrames, null);
  assert.ok(videoGuide.notes.some(note => /no full extra stream is assumed/.test(note)));

  const autoCanvas = estimateH3DurationGuidance(input({ resolution: '720p' }));
  assert.equal(autoCanvas.path, 'uncertain');
  assert.equal(autoCanvas.rowsHigh, null);
  assert.equal(autoCanvas.recommendationFrames, 345, 'the pixel policy still supports the VRAM recommendation');

  const roundedCanvas = estimateH3DurationGuidance(input({ resolution: '1280x720' }));
  assert.equal(roundedCanvas.path, 'uncertain', 'non-32px canvas dimensions are not rounded with JavaScript semantics');
  assert.equal(roundedCanvas.rowsHigh, null);

  const continuation = estimateH3DurationGuidance(input({ continuationFrames: 17 }));
  assert.ok(continuation.rowsLow > shortPass.rowsLow, '17 history frames and their boundary add conditioning rows while reducing target frames');
  assert.ok(continuation.notes.some(note => /continuation|history/i.test(note)));

  console.log('H3 duration guidance: packing rows, RMSNorm cutoff, mode-specific VRAM policy, references, guides, unknown geometry and continuation passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
