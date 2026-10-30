// UI build staging/deployment safety regressions use temporary files only.
// Run: node tests/ui/ui_build_safety.cjs
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const helperPath = path.resolve(__dirname, '../../ui/scripts/ui-build.mjs');
const helperPromise = import(pathToFileURL(helperPath).href);
const hash = value => crypto.createHash('sha256').update(value).digest('hex');

async function write(root, relative, value) {
  const target = path.join(root, ...relative.split('/'));
  await fs.mkdir(path.dirname(target), {recursive: true});
  await fs.writeFile(target, value);
}

async function read(root, relative) {
  return fs.readFile(path.join(root, ...relative.split('/')), 'utf8');
}

async function makeUi(root, source = 'export const app = 1;\n') {
  await fs.mkdir(root, {recursive: true});
  await write(root, 'package.json', '{"name":"fixture-ui"}\n');
  await write(root, 'tsconfig.json', '{"compilerOptions":{}}\n');
  await write(root, '.env', 'VITE_FIXTURE=base\n');
  await write(root, '.env.production', 'VITE_FIXTURE=production\n');
  await write(root, 'vite.config.mjs', 'export default {};\n');
  await write(root, 'index.html', '<main>input</main>\n');
  await write(root, 'src/main.ts', source);
  await write(root, 'src/AppModeNavigation.tsx', 'export function AppModeNavigation(){ return <img />; }\n');
  await write(root, 'src/dist/kept.txt', 'legitimate source directory\n');
  await write(root, 'public/staging/kept.txt', 'legitimate public directory\n');
  await write(root, 'scripts/baseline/kept.txt', 'legitimate scripts directory\n');
  await write(root, 'node_modules/fixture/generated.js', 'excluded dependency\n');
  return root;
}

async function fakeCompile(_uiRoot, stageRoot) {
  await write(stageRoot, 'index.html', '<script type="module" src="/assets/app-abcdefgh.js"></script>\n');
  await write(stageRoot, 'assets/app-abcdefgh.js', 'new application bundle\n');
  await write(stageRoot, 'manifest.webmanifest', '{"name":"fixture"}\n');
}

async function makeTemp() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'maestro-ui-build-safety-'));
}

async function testInputInventoryAndNormalStageBuild() {
  const {collectUiInputs, buildUi, verifyStage} = await helperPromise;
  const temp = await makeTemp();
  try {
    const ui = await makeUi(path.join(temp, 'ui'));
    const snapshot = await collectUiInputs(ui);
    const inputPaths = snapshot.files.map(file => file.path);
    for (const expected of [
      '.env', '.env.production', 'vite.config.mjs', 'src/dist/kept.txt',
      'public/staging/kept.txt', 'scripts/baseline/kept.txt',
    ]) assert(inputPaths.includes(expected), `expected input inventory to include ${expected}`);
    assert(!inputPaths.some(file => file.startsWith('node_modules/')));

    const result = await buildUi({uiRoot: ui, stageOnly: true, compile: fakeCompile});
    assert.equal(result.activated, false);
    assert.equal((await verifyStage(result.stageRoot, ui)).input_manifest_sha256,
      result.receipt.input_manifest_sha256);
    assert(!await fs.stat(path.join(ui, 'dist')).then(() => true, () => false));
    assert(!JSON.stringify(result.receipt).includes(temp));
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testBaselineRequiresReviewedParity() {
  const {approveBaselineFromReviewedTree, assertBaselineMatchesCurrent, buildUi} = await helperPromise;
  const temp = await makeTemp();
  try {
    const ui = await makeUi(path.join(temp, 'ui'));
    const reviewed = await makeUi(path.join(temp, 'reviewed'));
    const baseline = await approveBaselineFromReviewedTree(ui, reviewed);
    assert.equal(baseline.files.length, (await assertBaselineMatchesCurrent(ui)).current.files.length);

    await write(ui, 'src/main.ts', 'export const app = 2;\n');
    await assert.rejects(assertBaselineMatchesCurrent(ui), error => {
      assert.match(error.message, /changed: src\/main\.ts/);
      assert.match(error.message, /approve-baseline --from/);
      return true;
    });
    let compileCalls = 0;
    await assert.rejects(buildUi({
      uiRoot: ui,
      compile: async () => { compileCalls += 1; },
    }), /differs from approved baseline/);
    assert.equal(compileCalls, 0, 'a baseline mismatch must stop before the compiler runs');
    await assert.rejects(approveBaselineFromReviewedTree(ui, reviewed), /does not match reviewed tree/);
    const stillApproved = JSON.parse(await read(ui, '.maestro-source-baseline.json'));
    assert.equal(stillApproved.manifest_sha256, baseline.manifest_sha256,
      'a mismatched tree must not replace the saved baseline');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testDifferentSourceTreeCannotDeployForeignNavigationBundle() {
  const {buildUi, deployStage} = await helperPromise;
  const temp = await makeTemp();
  try {
    const reviewedUi = await makeUi(path.join(temp, 'reviewed-ui'));
    const destinationUi = await makeUi(path.join(temp, 'destination-ui'));
    await write(reviewedUi, 'src/AppModeNavigation.tsx',
      'export function AppModeNavigation(){ return <nav className="theme-m">M</nav>; }\n');
    await write(destinationUi, 'dist/index.html', '<main>old served image UI</main>\n');
    await write(destinationUi, 'dist/assets/old-abcdefgh.js', 'old image navigation bundle\n');
    const staged = await buildUi({uiRoot: reviewedUi, stageOnly: true, compile: fakeCompile});

    // No approved baseline is installed here: the receipt alone must prevent
    // a themed AppModeNavigation bundle from replacing the old image function.
    await assert.rejects(deployStage({
      uiRoot: destinationUi,
      stageRoot: staged.stageRoot,
    }), error => {
      assert.match(error.message, /Build receipt source does not match current UI inputs/);
      assert.match(error.message, /changed: src\/AppModeNavigation\.tsx/);
      return true;
    });
    assert.equal(await read(destinationUi, 'dist/index.html'), '<main>old served image UI</main>\n');
    assert.equal(await read(destinationUi, 'dist/assets/old-abcdefgh.js'),
      'old image navigation bundle\n');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testMissingOrTamperedStageEvidenceCannotInstall() {
  const {buildUi, deployStage} = await helperPromise;
  const temp = await makeTemp();
  try {
    const sourceUi = await makeUi(path.join(temp, 'source-ui'));
    const targetUi = await makeUi(path.join(temp, 'target-ui'));
    await write(targetUi, 'dist/index.html', '<main>old served UI</main>\n');
    await write(targetUi, 'dist/manifest.webmanifest', '{"name":"old"}\n');
    const staged = await buildUi({uiRoot: sourceUi, stageOnly: true, compile: fakeCompile});
    const receiptPath = path.join(staged.stageRoot, '.maestro-build-receipt.json');
    const receiptBytes = await fs.readFile(receiptPath);

    await fs.unlink(receiptPath);
    await assert.rejects(deployStage({uiRoot: targetUi, stageRoot: staged.stageRoot}), /ENOENT/);
    assert.equal(await read(targetUi, 'dist/index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(targetUi, 'dist/manifest.webmanifest'), '{"name":"old"}\n');
    await fs.writeFile(receiptPath, receiptBytes);

    const receipt = JSON.parse(receiptBytes.toString('utf8'));
    receipt.outputs.find(file => file.path === 'manifest.webmanifest').sha256 = hash('forged');
    await fs.writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
    await assert.rejects(deployStage({uiRoot: targetUi, stageRoot: staged.stageRoot}),
      /receipt output manifest checksum is invalid/);
    assert.equal(await read(targetUi, 'dist/index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(targetUi, 'dist/manifest.webmanifest'), '{"name":"old"}\n');

    await fs.writeFile(receiptPath, receiptBytes);
    await write(staged.stageRoot, 'assets/app-abcdefgh.js', 'tampered before deploy\n');
    await assert.rejects(deployStage({uiRoot: targetUi, stageRoot: staged.stageRoot}),
      /Build output checksum mismatch: assets\/app-abcdefgh\.js/);
    assert.equal(await read(targetUi, 'dist/index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(targetUi, 'dist/manifest.webmanifest'), '{"name":"old"}\n');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testSourceDriftDuringBuildKeepsServedIndex() {
  const {buildUi} = await helperPromise;
  const temp = await makeTemp();
  try {
    const ui = await makeUi(path.join(temp, 'ui'));
    await write(ui, 'dist/index.html', '<main>old served UI</main>\n');
    await write(ui, 'dist/assets/old-abcdefgh.js', 'old hashed bundle\n');
    await assert.rejects(buildUi({
      uiRoot: ui,
      compile: async (root, stage) => {
        await fakeCompile(root, stage);
        await write(root, 'src/main.ts', 'export const app = 9;\n');
      },
    }), /source changed while the build was running/);
    assert.equal(await read(ui, 'dist/index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(ui, 'dist/assets/old-abcdefgh.js'), 'old hashed bundle\n');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testCrossTreeDeployIsIndexLastAndPreservesOldAssets() {
  const {buildUi, deployStage, verifyDeployment} = await helperPromise;
  const temp = await makeTemp();
  try {
    const sourceUi = await makeUi(path.join(temp, 'source-ui'));
    const targetUi = await makeUi(path.join(temp, 'target-ui'));
    await write(targetUi, 'dist/index.html', '<main>old served UI</main>\n');
    await write(targetUi, 'dist/manifest.webmanifest', '{"name":"old"}\n');
    await write(targetUi, 'dist/assets/app-oldoldold.js', 'old hashed bundle\n');
    const staged = await buildUi({uiRoot: sourceUi, stageOnly: true, compile: fakeCompile});
    let observedOldIndex = false;
    const result = await deployStage({
      uiRoot: targetUi,
      stageRoot: staged.stageRoot,
      hooks: {
        beforeIndexSwitch: async ({distRoot}) => {
          observedOldIndex = await fs.readFile(path.join(distRoot, 'index.html'), 'utf8')
            === '<main>old served UI</main>\n';
          assert.equal(await fs.readFile(path.join(distRoot, 'manifest.webmanifest'), 'utf8'),
            '{"name":"fixture"}\n');
        },
      },
    });
    assert(observedOldIndex, 'stable files are installed while the old index still serves');
    assert.equal(await read(targetUi, 'dist/index.html'),
      '<script type="module" src="/assets/app-abcdefgh.js"></script>\n');
    assert.equal(await read(targetUi, 'dist/assets/app-oldoldold.js'), 'old hashed bundle\n');
    assert.equal((await verifyDeployment(targetUi)).output_manifest_sha256,
      result.receipt.output_manifest_sha256);
    const backupRoot = path.join(targetUi, result.backupDirectory);
    assert.equal(await read(backupRoot, 'index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(backupRoot, 'manifest.webmanifest'), '{"name":"old"}\n');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testMidDeployTamperingAndConcurrentIndexAreRejected() {
  const {buildUi, deployStage} = await helperPromise;
  const temp = await makeTemp();
  try {
    const sourceUi = await makeUi(path.join(temp, 'source-ui'));
    const targetUi = await makeUi(path.join(temp, 'target-ui'));
    await write(targetUi, 'dist/index.html', '<main>old served UI</main>\n');
    await write(targetUi, 'dist/manifest.webmanifest', '{"name":"old"}\n');
    const staged = await buildUi({uiRoot: sourceUi, stageOnly: true, compile: fakeCompile});

    await assert.rejects(deployStage({
      uiRoot: targetUi,
      stageRoot: staged.stageRoot,
      hooks: {
        beforeIndexSwitch: async ({stageRoot}) => {
          await write(stageRoot, 'assets/app-abcdefgh.js', 'tampered staged bundle\n');
        },
      },
    }), /Build output checksum mismatch: assets\/app-abcdefgh\.js/);
    assert.equal(await read(targetUi, 'dist/index.html'), '<main>old served UI</main>\n');
    assert.equal(await read(targetUi, 'dist/manifest.webmanifest'), '{"name":"old"}\n');

    // Rebuild the stage after the deliberate mutation, then simulate another
    // writer changing index.html while stable assets are being installed.
    const nextStage = await buildUi({uiRoot: sourceUi, stageOnly: true, compile: fakeCompile});
    await assert.rejects(deployStage({
      uiRoot: targetUi,
      stageRoot: nextStage.stageRoot,
      hooks: {
        beforeIndexSwitch: async ({distRoot}) => {
          await write(distRoot, 'index.html', '<main>intervening writer</main>\n');
        },
      },
    }), /index\.html changed during build activation/);
    assert.equal(await read(targetUi, 'dist/index.html'), '<main>intervening writer</main>\n');
    assert.equal(await read(targetUi, 'dist/manifest.webmanifest'), '{"name":"old"}\n');
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testUnsafePathsAndNestedStageOutputAreRejected() {
  const {buildUi, createBuildReceipt} = await helperPromise;
  assert.throws(() => createBuildReceipt(
    {hash_basis: 'basis', files: []},
    [{path: 'assets/bad:stream.js', size: 0, sha256: hash('')}],
  ), /Unsafe relative path/);
  assert.throws(() => createBuildReceipt(
    {hash_basis: 'basis', files: []},
    [{path: '../outside.js', size: 0, sha256: hash('')}],
  ), /Unsafe relative path/);

  const temp = await makeTemp();
  try {
    const ui = await makeUi(path.join(temp, 'ui'));
    await assert.rejects(buildUi({
      uiRoot: ui,
      stageOnly: true,
      outDir: 'src/.maestro-staging-hidden',
      compile: fakeCompile,
    }), /direct \.maestro-staging-\* child/);
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

async function testSymlinkedConfigAndOutputParentAreRejectedWhenSupported() {
  const {collectUiInputs, buildUi, verifyStage} = await helperPromise;
  const temp = await makeTemp();
  try {
    const ui = await makeUi(path.join(temp, 'ui'));
    const configLink = path.join(ui, 'vite.config.js');
    try {
      await fs.symlink(path.join(ui, 'vite.config.mjs'), configLink, 'file');
      await assert.rejects(collectUiInputs(ui), /Symlinked UI config/);
    } catch (error) {
      if (error.code !== 'EPERM' && error.code !== 'EACCES' && error.code !== 'UNKNOWN') throw error;
    }
    await fs.rm(configLink, {force: true});

    const stage = await buildUi({uiRoot: ui, stageOnly: true, compile: fakeCompile});
    const linkedAssets = path.join(temp, 'linked-assets');
    await fs.mkdir(linkedAssets);
    await fs.rm(path.join(stage.stageRoot, 'assets'), {recursive: true});
    try {
      await fs.symlink(linkedAssets, path.join(stage.stageRoot, 'assets'), 'junction');
      await assert.rejects(verifyStage(stage.stageRoot, ui), /Symlinked build output path/);
    } catch (error) {
      if (error.code !== 'EPERM' && error.code !== 'EACCES' && error.code !== 'UNKNOWN') throw error;
    }
  } finally {
    await fs.rm(temp, {recursive: true, force: true});
  }
}

(async () => {
  const tests = [
    testInputInventoryAndNormalStageBuild,
    testBaselineRequiresReviewedParity,
    testSourceDriftDuringBuildKeepsServedIndex,
    testDifferentSourceTreeCannotDeployForeignNavigationBundle,
    testMissingOrTamperedStageEvidenceCannotInstall,
    testCrossTreeDeployIsIndexLastAndPreservesOldAssets,
    testMidDeployTamperingAndConcurrentIndexAreRejected,
    testUnsafePathsAndNestedStageOutputAreRejected,
    testSymlinkedConfigAndOutputParentAreRejectedWhenSupported,
  ];
  for (const test of tests) {
    await test();
    console.log(`ok - ${test.name}`);
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
