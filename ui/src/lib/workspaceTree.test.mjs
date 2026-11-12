import assert from 'node:assert/strict'
import test from 'node:test'
import { createJiti } from 'jiti'
const jiti = createJiti(import.meta.url)
const {
  buildWorkspaceTree, cleanWorkspaceLabel, isWithinWorkspace, rebaseWorkspaceName, rebaseWorkspaceUrl,
  workspaceDescendants, workspaceDropProblem, workspaceLabel, workspaceNameProblem, workspaceParent,
} = await jiti.import('./workspaceTree.ts')

// Older servers (and UI fixtures) send only names; the tree derives the rest.
const tree = buildWorkspaceTree([
  { name: 'default' }, { name: 'A', file_count: 2 }, { name: 'A/B', file_count: 3 }, { name: 'A/B/C' },
  { name: 'D' }, { name: 'D/B' }, { name: 'Worlds', reserved: true },
])

test('names parse into parents, labels and children', () => {
  assert.equal(workspaceParent('A/B/C'), 'A/B')
  assert.equal(workspaceParent('A'), 'default')
  assert.equal(workspaceParent('default'), null)
  assert.equal(workspaceLabel('A/B/C'), 'C')
  assert.deepEqual(tree.get('A').children, ['A/B'])
  assert.deepEqual(workspaceDescendants(tree, 'A'), ['A/B', 'A/B/C'])
  assert.equal(tree.get('A/B/C').depth, 3)
})

test('within and rebase respect folder boundaries', () => {
  assert.ok(isWithinWorkspace('A/B', 'A'))
  assert.ok(!isWithinWorkspace('AB', 'A'))
  assert.equal(rebaseWorkspaceName('A/B/C', 'A/B', 'X'), 'X/C')
  assert.equal(rebaseWorkspaceName('AB', 'A', 'X'), 'AB')
  assert.equal(rebaseWorkspaceName('A', 'default', 'X'), 'A')
  assert.equal(rebaseWorkspaceUrl('/api/v1/file/a%20b.mp4?workspace=A%2FB&size=480', 'A', 'Z'),
    '/api/v1/file/a%20b.mp4?workspace=Z%2FB&size=480')
  assert.equal(rebaseWorkspaceUrl('/api/v1/file/x.mp4?workspace=AB', 'A', 'Z'), '/api/v1/file/x.mp4?workspace=AB')
})

test('drops refuse loops, no-ops, collisions, reserved folders and over-deep trees', () => {
  assert.equal(workspaceDropProblem(tree, 'A/B', 'D'), '"B" already exists there.')
  assert.equal(workspaceDropProblem(tree, 'A', 'A/B'), 'A folder cannot go inside itself.')
  assert.equal(workspaceDropProblem(tree, 'A', 'A'), 'A folder cannot go inside itself.')
  assert.equal(workspaceDropProblem(tree, 'A/B', 'A'), 'It is already there.')
  assert.equal(workspaceDropProblem(tree, 'A', 'default'), 'It is already there.')
  assert.equal(workspaceDropProblem(tree, 'A', 'Worlds'), 'That folder is managed by Maestro.')
  assert.equal(workspaceDropProblem(tree, 'Worlds', 'A'), 'This folder cannot be moved.')
  assert.equal(workspaceDropProblem(tree, 'default', 'A'), 'This folder cannot be moved.')
  assert.equal(workspaceDropProblem(tree, 'A/B/C', 'default'), null)
  assert.equal(workspaceDropProblem(tree, 'D', 'A/B'), null)
  const deep = buildWorkspaceTree(['P', 'P/1', 'P/1/2', 'P/1/2/3', 'P/1/2/3/4', 'P/1/2/3/4/5', 'P/1/2/3/4/5/6', 'Q', 'Q/R']
    .map(name => ({ name })))
  assert.match(workspaceDropProblem(deep, 'Q', 'P/1/2/3/4/5/6'), /at most 8 deep/)
  assert.equal(workspaceDropProblem(deep, 'Q/R', 'P/1/2/3/4/5/6'), null)
})

test('new names follow the server rule', () => {
  assert.equal(cleanWorkspaceLabel('  My  Shoot '), 'My-Shoot')
  assert.equal(workspaceNameProblem('Shoot_2-b'), null)
  assert.ok(workspaceNameProblem('a/b'))
  assert.ok(workspaceNameProblem('_hidden'))
  assert.ok(workspaceNameProblem('default'))
  assert.ok(workspaceNameProblem(''))
})
