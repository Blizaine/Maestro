#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { constants as fsConstants } from 'node:fs'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const DEFAULT_UI_ROOT = path.resolve(SCRIPT_DIR, '..')
const RECEIPT_NAME = '.maestro-build-receipt.json'
const BASELINE_NAME = '.maestro-source-baseline.json'
const FORMAT_VERSION = 1
const INPUT_HASH_BASIS = 'sha256 of raw file bytes; UI-root-relative POSIX paths; no text or line-ending normalization'
const OUTPUT_HASH_BASIS = 'sha256 of raw build-output file bytes; output-root-relative POSIX paths'
const SOURCE_DIRECTORIES = ['src', 'public', 'scripts']
const ROOT_INPUT_NAMES = new Set([
  'index.html', 'package.json', 'package-lock.json', 'npm-shrinkwrap.json',
  'yarn.lock', 'pnpm-lock.yaml', 'bun.lock', 'bun.lockb', 'jsconfig.json',
])
const ROOT_CONFIG_PATTERNS = [
  /^tsconfig(?:\.[\w.-]+)?\.json$/i,
  /^(?:vite|eslint|postcss|tailwind|babel|vitest|webpack|rollup)\.config\.[cm]?[jt]sx?$/i,
  /^\.eslintrc(?:\.[\w.-]+)?$/i,
  /^eslint\.config\.[cm]?[jt]sx?$/i,
  /^\.env(?:\..*)?$/,
  /^[\w.-]+\.d\.ts$/i,
]
const EXCLUDED_DIRECTORY_NAMES = new Set(['node_modules', '.git'])

const compareText = (left, right) => left < right ? -1 : left > right ? 1 : 0
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const canonicalJson = value => JSON.stringify(value)
const prettyJson = value => `${JSON.stringify(value, null, 2)}\n`
const isMissing = error => error?.code === 'ENOENT'
const isWithin = (root, target) => {
  const relative = path.relative(root, target)
  return relative === '' || (!path.isAbsolute(relative)
    && relative !== '..' && !relative.startsWith(`..${path.sep}`))
}

function safeRelativePath(relativePath) {
  if (typeof relativePath !== 'string' || !relativePath || relativePath.includes('\\')
    || relativePath.includes(':')
    || path.posix.isAbsolute(relativePath) || /^[A-Za-z]:/.test(relativePath)
    || relativePath.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error(`Unsafe relative path: ${String(relativePath)}`)
  }
  return relativePath
}

function resolveChild(root, relativePath) {
  const safePath = safeRelativePath(relativePath)
  const resolved = path.resolve(root, ...safePath.split('/'))
  if (!isWithin(path.resolve(root), resolved) || resolved === path.resolve(root)) {
    throw new Error(`Path escapes its root: ${relativePath}`)
  }
  return resolved
}

function sortedUniqueFiles(files, label) {
  const seen = new Set()
  const normalized = files.map(file => {
    const relativePath = safeRelativePath(file.path)
    if (seen.has(relativePath)) throw new Error(`Duplicate ${label} path: ${relativePath}`)
    seen.add(relativePath)
    if (!/^[0-9a-f]{64}$/.test(file.sha256) || !Number.isSafeInteger(file.size) || file.size < 0) {
      throw new Error(`Invalid ${label} hash entry: ${relativePath}`)
    }
    return {path: relativePath, sha256: file.sha256, size: file.size}
  })
  normalized.sort((a, b) => compareText(a.path, b.path))
  return normalized
}

function manifestDigest(files, hashBasis) {
  return sha256(Buffer.from(canonicalJson({format_version: FORMAT_VERSION, hash_basis: hashBasis, files})))
}

function normalizeSnapshot(snapshot, label = 'input') {
  if (!snapshot || !Array.isArray(snapshot.files)) throw new Error(`Invalid ${label} snapshot`)
  return {
    hash_basis: snapshot.hash_basis,
    files: sortedUniqueFiles(snapshot.files, label),
  }
}

function changesBetween(expectedFiles, actualFiles) {
  const expected = new Map(expectedFiles.map(file => [file.path, file]))
  const actual = new Map(actualFiles.map(file => [file.path, file]))
  const paths = [...new Set([...expected.keys(), ...actual.keys()])].sort(compareText)
  return paths.flatMap(relativePath => {
    const before = expected.get(relativePath)
    const after = actual.get(relativePath)
    if (!before) return [`added: ${relativePath}`]
    if (!after) return [`removed: ${relativePath}`]
    if (before.sha256 !== after.sha256 || before.size !== after.size) {
      return [`changed: ${relativePath}`]
    }
    return []
  })
}

function assertSameFiles(expectedFiles, actualFiles, context) {
  const changes = changesBetween(expectedFiles, actualFiles)
  if (changes.length) {
    throw new Error(`${context}:\n${changes.map(item => `  - ${item}`).join('\n')}`)
  }
}

async function assertNoSymlinkAncestors(targetPath, {
  allowMissing = false,
  finalType = null,
} = {}) {
  const absolute = path.resolve(targetPath)
  const parsed = path.parse(absolute)
  const parts = path.relative(parsed.root, absolute).split(path.sep).filter(Boolean)
  let current = parsed.root
  const paths = parts.length ? parts : ['']
  for (let index = 0; index < paths.length; index += 1) {
    if (paths[index]) current = path.join(current, paths[index])
    let stat
    try {
      stat = await fs.lstat(current)
    } catch (error) {
      if (allowMissing && isMissing(error)) return false
      throw error
    }
    if (stat.isSymbolicLink()) throw new Error(`Symlinked path is not allowed: ${current}`)
    const final = index === paths.length - 1
    if (!final && !stat.isDirectory()) throw new Error(`Path parent is not a directory: ${current}`)
    if (finalType === 'directory' && final && !stat.isDirectory()) {
      throw new Error(`Expected a real directory: ${current}`)
    }
    if (finalType === 'file' && final && !stat.isFile()) {
      throw new Error(`Expected a regular file: ${current}`)
    }
  }
  return true
}

async function assertRealDirectory(directoryPath, {create = false} = {}) {
  try {
    await assertNoSymlinkAncestors(directoryPath, {finalType: 'directory'})
  } catch (error) {
    if (!isMissing(error) || !create) throw error
    await fs.mkdir(directoryPath, {recursive: true})
    await assertNoSymlinkAncestors(directoryPath, {finalType: 'directory'})
  }
}

async function walkRegularFiles(root, relativeDirectory, results, {
  excludeReceipt = false,
  excludeDependencies = false,
} = {}) {
  const absoluteDirectory = relativeDirectory
    ? resolveChild(root, relativeDirectory)
    : root
  const entries = await fs.readdir(absoluteDirectory, {withFileTypes: true})
  entries.sort((a, b) => compareText(a.name, b.name))
  for (const entry of entries) {
    if (relativeDirectory === '' && excludeReceipt && entry.name === RECEIPT_NAME) continue
    if (excludeDependencies && EXCLUDED_DIRECTORY_NAMES.has(entry.name)) continue
    const relativePath = relativeDirectory
      ? `${relativeDirectory}/${entry.name}`
      : entry.name
    const absolutePath = resolveChild(root, relativePath)
    const stat = await fs.lstat(absolutePath)
    if (stat.isSymbolicLink()) {
      throw new Error(`Symlinked build input/output is not allowed: ${relativePath}`)
    }
    if (stat.isDirectory()) {
      await walkRegularFiles(root, relativePath, results, {excludeReceipt, excludeDependencies})
    } else if (stat.isFile()) {
      results.push(relativePath)
    } else {
      throw new Error(`Unsupported build input/output type: ${relativePath}`)
    }
  }
}

function isTopLevelBuildInput(name) {
  return ROOT_INPUT_NAMES.has(name) || ROOT_CONFIG_PATTERNS.some(pattern => pattern.test(name))
}

export async function collectUiInputs(uiRoot = DEFAULT_UI_ROOT) {
  uiRoot = path.resolve(uiRoot)
  await assertRealDirectory(uiRoot)
  const relativePaths = []
  for (const directory of SOURCE_DIRECTORIES) {
    try {
      const stat = await fs.lstat(path.join(uiRoot, directory))
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        throw new Error(`Expected a real UI input directory: ${directory}`)
      }
      await walkRegularFiles(uiRoot, directory, relativePaths, {excludeDependencies: true})
    } catch (error) {
      if (!isMissing(error)) throw error
    }
  }
  const rootEntries = await fs.readdir(uiRoot, {withFileTypes: true})
  for (const entry of rootEntries) {
    if (!isTopLevelBuildInput(entry.name)) continue
    if (entry.isSymbolicLink()) throw new Error(`Symlinked UI config is not allowed: ${entry.name}`)
    if (!entry.isFile()) continue
    relativePaths.push(entry.name)
  }
  relativePaths.sort(compareText)
  const files = []
  for (const relativePath of relativePaths) {
    const bytes = await fs.readFile(resolveChild(uiRoot, relativePath))
    files.push({path: relativePath, sha256: sha256(bytes), size: bytes.byteLength})
  }
  return {hash_basis: INPUT_HASH_BASIS, files}
}

async function collectOutputFiles(outputRoot) {
  outputRoot = path.resolve(outputRoot)
  await assertRealDirectory(outputRoot)
  const relativePaths = []
  await walkRegularFiles(outputRoot, '', relativePaths, {excludeReceipt: true})
  relativePaths.sort(compareText)
  const files = []
  for (const relativePath of relativePaths) {
    const bytes = await fs.readFile(resolveChild(outputRoot, relativePath))
    files.push({path: relativePath, sha256: sha256(bytes), size: bytes.byteLength})
  }
  if (!files.some(file => file.path === 'index.html')) {
    throw new Error('Build output is missing index.html')
  }
  return files
}

export function createBuildReceipt(inputSnapshot, outputFiles) {
  const inputs = normalizeSnapshot(inputSnapshot)
  const outputs = sortedUniqueFiles(outputFiles, 'output')
  return {
    format_version: FORMAT_VERSION,
    input_hash_basis: inputs.hash_basis,
    inputs: inputs.files,
    input_manifest_sha256: manifestDigest(inputs.files, inputs.hash_basis),
    output_hash_basis: OUTPUT_HASH_BASIS,
    outputs,
    output_manifest_sha256: manifestDigest(outputs, OUTPUT_HASH_BASIS),
  }
}

function validateBuildReceipt(receipt) {
  if (!receipt || receipt.format_version !== FORMAT_VERSION
    || receipt.input_hash_basis !== INPUT_HASH_BASIS
    || receipt.output_hash_basis !== OUTPUT_HASH_BASIS) {
    throw new Error('Missing or unsupported UI build receipt')
  }
  const inputs = sortedUniqueFiles(receipt.inputs, 'input')
  const outputs = sortedUniqueFiles(receipt.outputs, 'output')
  if (!outputs.some(file => file.path === 'index.html')) {
    throw new Error('Build receipt does not include index.html')
  }
  if (manifestDigest(inputs, receipt.input_hash_basis) !== receipt.input_manifest_sha256) {
    throw new Error('UI build receipt input manifest checksum is invalid')
  }
  if (manifestDigest(outputs, receipt.output_hash_basis) !== receipt.output_manifest_sha256) {
    throw new Error('UI build receipt output manifest checksum is invalid')
  }
  return {...receipt, inputs, outputs}
}

async function readJsonFile(filePath, description) {
  await assertNoSymlinkAncestors(path.dirname(filePath), {finalType: 'directory'})
  const stat = await fs.lstat(filePath)
  if (stat.isSymbolicLink() || !stat.isFile()) throw new Error(`${description} must be a regular file`)
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'))
  } catch (error) {
    throw new Error(`${description} is invalid JSON: ${error.message}`)
  }
}

async function readOptionalJson(filePath, description) {
  try {
    return await readJsonFile(filePath, description)
  } catch (error) {
    if (isMissing(error)) return null
    throw error
  }
}

async function replacePath(tempPath, targetPath) {
  try {
    await fs.rename(tempPath, targetPath)
    return
  } catch (error) {
    if (!['EEXIST', 'EPERM', 'ENOTEMPTY'].includes(error.code)) throw error
  }
  const holdPath = `${targetPath}.maestro-hold-${randomUUID()}`
  let movedOld = false
  try {
    await fs.rename(targetPath, holdPath)
    movedOld = true
  } catch (error) {
    if (!isMissing(error)) throw error
  }
  try {
    await fs.rename(tempPath, targetPath)
  } catch (error) {
    if (movedOld) await fs.rename(holdPath, targetPath).catch(() => {})
    throw error
  }
  // The replacement has committed. Failure to clean up the old hold file
  // must not be reported as an activation failure after the index switched.
  if (movedOld) await fs.unlink(holdPath).catch(() => {})
}

async function writeAtomic(filePath, bytes) {
  const parent = path.dirname(filePath)
  await assertRealDirectory(parent, {create: true})
  try {
    const existing = await fs.lstat(filePath)
    if (existing.isSymbolicLink() || !existing.isFile()) {
      throw new Error(`Refusing to replace non-regular file: ${filePath}`)
    }
  } catch (error) {
    if (!isMissing(error)) throw error
  }
  const tempPath = `${filePath}.maestro-tmp-${randomUUID()}`
  await fs.writeFile(tempPath, bytes, {flag: 'wx'})
  try {
    await replacePath(tempPath, filePath)
  } catch (error) {
    await fs.unlink(tempPath).catch(() => {})
    throw error
  }
}

async function readBaseline(uiRoot) {
  return readOptionalJson(path.join(uiRoot, BASELINE_NAME), 'UI source baseline')
}

function validateBaseline(baseline) {
  if (!baseline || baseline.format_version !== FORMAT_VERSION
    || baseline.hash_basis !== INPUT_HASH_BASIS) {
    throw new Error('UI source baseline is missing or uses an unsupported format')
  }
  const files = sortedUniqueFiles(baseline.files, 'baseline')
  if (manifestDigest(files, baseline.hash_basis) !== baseline.manifest_sha256) {
    throw new Error('UI source baseline checksum is invalid')
  }
  return {...baseline, files}
}

export async function assertBaselineMatchesCurrent(uiRoot = DEFAULT_UI_ROOT, current = null) {
  const baseline = await readBaseline(uiRoot)
  if (!baseline) return {baseline: null, current: current || await collectUiInputs(uiRoot)}
  const approved = validateBaseline(baseline)
  const snapshot = current || await collectUiInputs(uiRoot)
  assertSameFiles(approved.files, snapshot.files,
    'UI source differs from approved baseline. Review the changed paths, then explicitly update from a reviewed tree with `node scripts/ui-build.mjs approve-baseline --from <reviewed-ui-tree>`')
  return {baseline: approved, current: snapshot}
}

export async function approveBaselineFromReviewedTree(uiRoot, reviewedUiRoot) {
  uiRoot = path.resolve(uiRoot)
  reviewedUiRoot = path.resolve(reviewedUiRoot)
  const current = await collectUiInputs(uiRoot)
  const reviewed = await collectUiInputs(reviewedUiRoot)
  assertSameFiles(reviewed.files, current.files,
    'Cannot approve UI baseline because current source does not match reviewed tree')
  const baseline = {
    format_version: FORMAT_VERSION,
    hash_basis: INPUT_HASH_BASIS,
    files: current.files,
    manifest_sha256: manifestDigest(current.files, current.hash_basis),
  }
  await writeAtomic(path.join(uiRoot, BASELINE_NAME), prettyJson(baseline))
  return baseline
}

async function assertInputsMatchReceipt(uiRoot, receipt) {
  const current = await collectUiInputs(uiRoot)
  await assertBaselineMatchesCurrent(uiRoot, current)
  assertSameFiles(receipt.inputs, current.files,
    'Build receipt source does not match current UI inputs')
  return current
}

async function verifyOutputManifest(outputRoot, files) {
  await assertRealDirectory(outputRoot)
  for (const file of files) {
    const target = await assertRegularFileUnder(outputRoot, file.path)
    const bytes = await fs.readFile(target)
    if (bytes.byteLength !== file.size || sha256(bytes) !== file.sha256) {
      throw new Error(`Build output checksum mismatch: ${file.path}`)
    }
  }
}

async function readStageReceipt(stageRoot) {
  await assertRealDirectory(stageRoot)
  const receipt = validateBuildReceipt(await readJsonFile(
    path.join(stageRoot, RECEIPT_NAME), 'Staged UI build receipt',
  ))
  await verifyOutputManifest(stageRoot, receipt.outputs)
  return receipt
}

export async function verifyStage(stageRoot, uiRoot) {
  const receipt = await readStageReceipt(stageRoot)
  await assertInputsMatchReceipt(uiRoot, receipt)
  return receipt
}

async function fileSnapshot(filePath) {
  await assertNoSymlinkAncestors(path.dirname(filePath), {
    allowMissing: true,
    finalType: 'directory',
  })
  try {
    const stat = await fs.lstat(filePath)
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`Expected a regular file: ${filePath}`)
    }
    const bytes = await fs.readFile(filePath)
    return {exists: true, sha256: sha256(bytes), size: bytes.byteLength}
  } catch (error) {
    if (isMissing(error)) return {exists: false, sha256: null, size: null}
    throw error
  }
}

async function assertRegularFileUnder(root, relativePath) {
  await assertRealDirectory(root)
  const parts = safeRelativePath(relativePath).split('/')
  let current = path.resolve(root)
  for (let index = 0; index < parts.length; index += 1) {
    current = path.join(current, parts[index])
    const stat = await fs.lstat(current)
    if (stat.isSymbolicLink()) {
      throw new Error(`Symlinked build output path is not allowed: ${relativePath}`)
    }
    const final = index === parts.length - 1
    if (final ? !stat.isFile() : !stat.isDirectory()) {
      throw new Error(`Build output path has an invalid component: ${relativePath}`)
    }
  }
  return current
}

function sameSnapshot(left, right) {
  return left.exists === right.exists
    && (!left.exists || (left.sha256 === right.sha256 && left.size === right.size))
}

async function ensureDirectoryChain(root, relativeDirectory) {
  await assertRealDirectory(root, {create: true})
  let current = path.resolve(root)
  if (!relativeDirectory) return current
  for (const part of relativeDirectory.split('/')) {
    if (!part || part === '.' || part === '..') throw new Error(`Unsafe directory path: ${relativeDirectory}`)
    current = path.join(current, part)
    try {
      const stat = await fs.lstat(current)
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        throw new Error(`Unsafe build destination directory: ${current}`)
      }
    } catch (error) {
      if (!isMissing(error)) throw error
      await fs.mkdir(current)
    }
  }
  return current
}

async function ensureDestinationIsRegularOrMissing(distRoot, relativePath) {
  const parentRelative = path.posix.dirname(relativePath)
  await ensureDirectoryChain(distRoot, parentRelative === '.' ? '' : parentRelative)
  const target = resolveChild(distRoot, relativePath)
  const snapshot = await fileSnapshot(target)
  return {target, snapshot}
}

function isHashedAsset(relativePath) {
  if (!relativePath.startsWith('assets/')) return false
  const name = path.posix.basename(relativePath)
  return /-[A-Za-z0-9_-]{8,}\.[^./]+$/.test(name)
}

async function copyWithBackup(sourcePath, targetPath, backupPath = null) {
  const current = await fileSnapshot(targetPath)
  if (current.exists) {
    const sourceBytes = await fs.readFile(sourcePath)
    if (current.sha256 === sha256(sourceBytes) && current.size === sourceBytes.byteLength) {
      return {changed: false, existed: true, previous: current, backupPath: null}
    }
    if (backupPath) {
      await assertRealDirectory(path.dirname(backupPath), {create: true})
      await fs.copyFile(targetPath, backupPath, fsConstants.COPYFILE_EXCL)
    }
  }
  const tempPath = `${targetPath}.maestro-tmp-${randomUUID()}`
  await fs.copyFile(sourcePath, tempPath)
  try {
    await replacePath(tempPath, targetPath)
  } catch (error) {
    await fs.unlink(tempPath).catch(() => {})
    throw error
  }
  return {changed: true, existed: current.exists, previous: current, backupPath}
}

async function rollbackFileAction(action, {distRoot, uiRoot, receiptName = null} = {}) {
  const targetPath = action.kind === 'receipt'
    ? path.join(uiRoot, receiptName || RECEIPT_NAME)
    : resolveChild(distRoot, action.relativePath)
  const current = await fileSnapshot(targetPath)
  if (action.existed) {
    if (!action.backupPath) return
    // Preserve an intervening user change; only restore our backup when the
    // destination is still the file this deployment wrote.
    if (current.exists && current.sha256 !== action.writtenSha256) return
    const tempPath = `${targetPath}.maestro-rollback-${randomUUID()}`
    await fs.copyFile(action.backupPath, tempPath)
    await replacePath(tempPath, targetPath)
    return
  }
  if (current.exists && current.sha256 === action.writtenSha256) {
    await fs.unlink(targetPath)
  }
}

async function writeReceiptSidecar(uiRoot, receipt, backupRoot) {
  const targetPath = path.join(uiRoot, RECEIPT_NAME)
  const previous = await fileSnapshot(targetPath)
  let backupPath = null
  if (previous.exists) {
    backupPath = path.join(backupRoot, RECEIPT_NAME)
    await fs.copyFile(targetPath, backupPath)
  }
  const bytes = Buffer.from(prettyJson(receipt))
  await writeAtomic(targetPath, bytes)
  return {kind: 'receipt', existed: previous.exists, backupPath, writtenSha256: sha256(bytes)}
}

async function rollbackActions(actions, options) {
  for (const action of [...actions].reverse()) {
    try {
      await rollbackFileAction(action, options)
    } catch {
      // Rollback is best-effort so the original deployment error remains the
      // primary diagnosis. Durable backups are retained for manual recovery.
    }
  }
}

async function replaceIndexLast(stageIndexPath, expectedIndex, distRoot, backupRoot, initialIndex) {
  const targetPath = path.join(distRoot, 'index.html')
  let backupPath = null
  if (initialIndex.exists) {
    backupPath = path.join(backupRoot, 'index.html')
    await fs.copyFile(targetPath, backupPath)
  }
  const sourceBytes = await fs.readFile(stageIndexPath)
  if (sourceBytes.byteLength !== expectedIndex.size || sha256(sourceBytes) !== expectedIndex.sha256) {
    throw new Error('Staged index.html changed during build activation')
  }
  const action = {
    kind: 'dist', relativePath: 'index.html', existed: initialIndex.exists,
    backupPath, writtenSha256: sha256(sourceBytes),
  }
  const tempPath = `${targetPath}.maestro-index-${randomUUID()}`
  await fs.writeFile(tempPath, sourceBytes, {flag: 'wx'})
  try {
    const beforeSwitch = await fileSnapshot(targetPath)
    if (!sameSnapshot(initialIndex, beforeSwitch)) {
      throw new Error('Existing dist/index.html changed during build activation; refusing index switch')
    }
    await replacePath(tempPath, targetPath)
  } catch (error) {
    await fs.unlink(tempPath).catch(() => {})
    throw error
  }
  return action
}

async function createBackupRoot(uiRoot) {
  return fs.mkdtemp(path.join(uiRoot, '.maestro-build-backups-'))
}

export async function deployStage({uiRoot = DEFAULT_UI_ROOT, stageRoot, hooks = {}}) {
  uiRoot = path.resolve(uiRoot)
  stageRoot = path.resolve(stageRoot)
  const receipt = await readStageReceipt(stageRoot)
  await assertInputsMatchReceipt(uiRoot, receipt)
  const distRoot = path.join(uiRoot, 'dist')
  try {
    const stat = await fs.lstat(distRoot)
    if (stat.isSymbolicLink() || !stat.isDirectory()) throw new Error('UI dist must be a real directory')
  } catch (error) {
    if (!isMissing(error)) throw error
  }
  const initialIndex = await fileSnapshot(path.join(distRoot, 'index.html'))
  const nonIndexOutputs = receipt.outputs.filter(file => file.path !== 'index.html')

  // Complete destination preflight before modifying dist. In particular, a
  // same-named hashed asset may never be replaced by different bytes.
  for (const file of nonIndexOutputs) {
    const {snapshot} = await ensureDestinationIsRegularOrMissing(distRoot, file.path)
    if (isHashedAsset(file.path) && snapshot.exists && snapshot.sha256 !== file.sha256) {
      throw new Error(`Refusing to replace conflicting hashed asset: ${file.path}`)
    }
  }
  await ensureDestinationIsRegularOrMissing(distRoot, 'index.html')
  const backupRoot = await createBackupRoot(uiRoot)
  const actions = []
  let indexAction = null
  try {
    for (const file of nonIndexOutputs) {
      const sourcePath = resolveChild(stageRoot, file.path)
      const {target} = await ensureDestinationIsRegularOrMissing(distRoot, file.path)
      const hashed = isHashedAsset(file.path)
      const before = await fileSnapshot(target)
      if (hashed && before.exists) continue
      const backupPath = before.exists
        ? resolveChild(backupRoot, file.path)
        : null
      const action = {
        kind: 'dist', relativePath: file.path, existed: before.exists,
        backupPath, writtenSha256: file.sha256,
      }
      actions.push(action)
      await copyWithBackup(sourcePath, target, backupPath)
    }

    // Receipt is local metadata, not a served file. It is prepared before
    // index.html so the index remains the final serving switch.
    const receiptAction = await writeReceiptSidecar(uiRoot, receipt, backupRoot)
    actions.push(receiptAction)
    await hooks.beforeIndexSwitch?.({
      uiRoot, distRoot, stageRoot, receipt, backupRoot,
    })

    const finalReceipt = await readStageReceipt(stageRoot)
    if (finalReceipt.input_manifest_sha256 !== receipt.input_manifest_sha256
      || finalReceipt.output_manifest_sha256 !== receipt.output_manifest_sha256) {
      throw new Error('Staged build receipt changed during activation')
    }
    await assertInputsMatchReceipt(uiRoot, finalReceipt)
    await verifyOutputManifest(
      distRoot,
      finalReceipt.outputs.filter(file => file.path !== 'index.html'),
    )
    const currentIndex = await fileSnapshot(path.join(distRoot, 'index.html'))
    if (!sameSnapshot(initialIndex, currentIndex)) {
      throw new Error('Existing dist/index.html changed during build activation; refusing index switch')
    }
    // Index is switched only after receipt validation, output verification,
    // input recheck, and the existing-index race guard.
    indexAction = await replaceIndexLast(
      resolveChild(stageRoot, 'index.html'),
      finalReceipt.outputs.find(file => file.path === 'index.html'),
      distRoot, backupRoot, initialIndex,
    )
    return {
      receipt,
      backupDirectory: path.relative(uiRoot, backupRoot).split(path.sep).join('/'),
    }
  } catch (error) {
    if (indexAction) actions.push(indexAction)
    await rollbackActions(actions, {distRoot, uiRoot})
    throw error
  }
}

export async function verifyDeployment(uiRoot = DEFAULT_UI_ROOT) {
  uiRoot = path.resolve(uiRoot)
  const receipt = validateBuildReceipt(await readJsonFile(
    path.join(uiRoot, RECEIPT_NAME), 'UI build receipt',
  ))
  await assertInputsMatchReceipt(uiRoot, receipt)
  await verifyOutputManifest(path.join(uiRoot, 'dist'), receipt.outputs)
  return receipt
}

async function runCompiler(uiRoot, stageRoot) {
  const tsc = path.join(uiRoot, 'node_modules', 'typescript', 'bin', 'tsc')
  const vite = path.join(uiRoot, 'node_modules', 'vite', 'bin', 'vite.js')
  for (const executable of [tsc, vite]) {
    const stat = await fs.stat(executable).catch(() => null)
    if (!stat?.isFile()) throw new Error(`Missing UI build tool: ${path.relative(uiRoot, executable)}`)
  }
  const run = (scriptPath, args) => {
    const result = spawnSync(process.execPath, [scriptPath, ...args], {
      cwd: uiRoot,
      stdio: 'inherit',
      windowsHide: true,
    })
    if (result.error) throw result.error
    if (result.status !== 0) {
      throw new Error(`${path.basename(scriptPath)} exited with status ${result.status ?? result.signal}`)
    }
  }
  run(tsc, ['-b'])
  run(vite, ['build', '--outDir', stageRoot, '--emptyOutDir'])
}

async function prepareStageDirectory(uiRoot, outDir) {
  if (!outDir) {
    const stageRoot = await fs.mkdtemp(path.join(uiRoot, '.maestro-staging-'))
    return {stageRoot, owned: true}
  }
  const stageRoot = path.resolve(uiRoot, outDir)
  const insideUi = isWithin(uiRoot, stageRoot)
  const directLocalStage = insideUi
    && path.dirname(stageRoot) === uiRoot
    && path.basename(stageRoot).startsWith('.maestro-staging-')
  if ((insideUi && !directLocalStage) || (!insideUi && isWithin(stageRoot, uiRoot))) {
    throw new Error('Output directory must be a direct .maestro-staging-* child of the UI root or an unrelated external directory')
  }
  try {
    const stat = await fs.lstat(stageRoot)
    if (stat.isSymbolicLink() || !stat.isDirectory()) throw new Error('Output directory must be a real directory')
    if ((await fs.readdir(stageRoot)).length > 0) throw new Error('Output directory must be empty')
  } catch (error) {
    if (!isMissing(error)) throw error
    await fs.mkdir(stageRoot, {recursive: true})
  }
  return {stageRoot, owned: false}
}

function assertNode20() {
  const major = Number(process.versions.node.split('.')[0])
  if (!Number.isInteger(major) || major < 20) {
    throw new Error(`UI builds require Node.js 20 or newer; found ${process.versions.node}`)
  }
}

export async function buildUi({
  uiRoot = DEFAULT_UI_ROOT,
  stageOnly = false,
  outDir = null,
  compile = runCompiler,
  hooks = {},
} = {}) {
  assertNode20()
  uiRoot = path.resolve(uiRoot)
  const before = await collectUiInputs(uiRoot)
  await assertBaselineMatchesCurrent(uiRoot, before)
  const {stageRoot, owned} = await prepareStageDirectory(uiRoot, outDir)
  try {
    await compile(uiRoot, stageRoot)
    const after = await collectUiInputs(uiRoot)
    assertSameFiles(before.files, after.files,
      'UI source changed while the build was running; staged output was discarded')
    await assertBaselineMatchesCurrent(uiRoot, after)
    const receipt = createBuildReceipt(before, await collectOutputFiles(stageRoot))
    await fs.writeFile(path.join(stageRoot, RECEIPT_NAME), prettyJson(receipt), {flag: 'wx'})
    await verifyStage(stageRoot, uiRoot)
    if (stageOnly || outDir) {
      return {stageRoot, receipt, activated: false}
    }
    const deployment = await deployStage({uiRoot, stageRoot, hooks})
    return {stageRoot, receipt: deployment.receipt, activated: true, ...deployment}
  } catch (error) {
    if (owned) await fs.rm(stageRoot, {recursive: true, force: true}).catch(() => {})
    throw error
  } finally {
    if (owned && !stageOnly && !outDir) {
      await fs.rm(stageRoot, {recursive: true, force: true}).catch(() => {})
    }
  }
}

function parseBuildArgs(args) {
  const options = {stageOnly: false, outDir: null}
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]
    if (arg === '--stage-only') options.stageOnly = true
    else if (arg === '--outDir') {
      if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error('--outDir needs a directory')
      options.outDir = args[++index]
    } else throw new Error(`Unknown build option: ${arg}`)
  }
  return options
}

function printHelp() {
  console.log(`UI build safety tool (Node.js 20+)

  npm run build
      Verify source and optional baseline, build in private staging, then
      activate with index.html as the final served-file switch.
  node scripts/ui-build.mjs build --stage-only [--outDir DIR]
      Build and write a local receipt without activating the staged output.
  node scripts/ui-build.mjs deploy --from DIR
      Activate a staged build only when its receipt matches current inputs.
      Add --ui TARGET_UI_ROOT for an explicit cross-tree target.
  node scripts/ui-build.mjs verify
      Verify current source and deployed output against the local receipt.
  node scripts/ui-build.mjs approve-baseline --from REVIEWED_UI_TREE
      Write/replace the ignored source baseline only after current inputs
      exactly match the reviewed UI tree.`)
}

function parseNamedOptions(args, allowedNames) {
  const parsed = {}
  for (let index = 0; index < args.length; index += 1) {
    const name = args[index]
    if (!allowedNames.includes(name)) throw new Error(`Unknown option: ${name}`)
    if (Object.hasOwn(parsed, name)) throw new Error(`Repeated option: ${name}`)
    const value = args[++index]
    if (!value || value.startsWith('--')) throw new Error(`${name} needs a value`)
    parsed[name] = value
  }
  return parsed
}

async function main(args = process.argv.slice(2)) {
  const [command = 'build', ...rest] = args
  if (command === '--help' || command === '-h' || command === 'help') {
    printHelp()
    return
  }
  if (command === 'build') {
    const result = await buildUi(parseBuildArgs(rest))
    console.log(result.activated
      ? `UI build activated; previous stable files are backed up under ${result.backupDirectory}.`
      : `UI build staged at ${path.relative(DEFAULT_UI_ROOT, result.stageRoot).split(path.sep).join('/')}.`)
    return
  }
  if (command === 'deploy') {
    const options = parseNamedOptions(rest, ['--from', '--ui'])
    const source = options['--from']
    if (!source) throw new Error('deploy requires --from STAGE_DIRECTORY')
    const targetUiRoot = options['--ui']
      ? path.resolve(DEFAULT_UI_ROOT, options['--ui'])
      : DEFAULT_UI_ROOT
    const result = await deployStage({
      uiRoot: targetUiRoot,
      stageRoot: path.resolve(DEFAULT_UI_ROOT, source),
    })
    console.log(`UI build activated; previous stable files are backed up under ${result.backupDirectory}.`)
    return
  }
  if (command === 'verify') {
    const options = parseNamedOptions(rest, ['--ui'])
    const uiRoot = options['--ui']
      ? path.resolve(DEFAULT_UI_ROOT, options['--ui'])
      : DEFAULT_UI_ROOT
    await verifyDeployment(uiRoot)
    console.log('UI build receipt matches current inputs and deployed output.')
    return
  }
  if (command === 'approve-baseline') {
    const options = parseNamedOptions(rest, ['--from'])
    const source = options['--from']
    if (!source) {
      throw new Error('approve-baseline requires exactly --from REVIEWED_UI_TREE')
    }
    const baseline = await approveBaselineFromReviewedTree(
      DEFAULT_UI_ROOT, path.resolve(DEFAULT_UI_ROOT, source),
    )
    console.log(`Approved ${baseline.files.length} UI inputs from the matching reviewed tree.`)
    return
  }
  throw new Error(`Unknown command: ${command}`)
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : ''
if (invokedPath === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(`UI build safety failed: ${error.message}`)
    process.exitCode = 1
  })
}
