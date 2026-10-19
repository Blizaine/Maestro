const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const root = path.resolve(__dirname, '../..')
const typescript = require(path.join(root, 'ui/node_modules/typescript'))

;(async () => {
  const moduleSource = fs.readFileSync(path.join(root, 'ui/src/lib/mossImageLayout.ts'), 'utf8')
  const output = typescript.transpileModule(moduleSource, {
    compilerOptions: { module: typescript.ModuleKind.CommonJS, target: typescript.ScriptTarget.ES2020 },
    reportDiagnostics: true,
  })
  const syntaxErrors = (output.diagnostics || []).filter(diagnostic => diagnostic.category === typescript.DiagnosticCategory.Error)
  assert.deepEqual(syntaxErrors, [], 'MOSS layout utility should transpile without syntax errors')
  const module = { exports: {} }
  new Function('module', 'exports', output.outputText)(module, module.exports)
  const {
    emptyMossImageDocument,
    fitMossImageBoxToCanvas,
    mossCanvasPoint,
    mossImageBoxProblem,
    parseMossImageLayout,
    parseMossImageSize,
    serializeMossImageLayout,
    validMossImageBox,
  } = module.exports

  const size = parseMossImageSize('800x600')
  assert.deepEqual(size, { width: 800, height: 600 })
  const rect = { left: 10, top: 20, width: 800, height: 600 }
  assert.deepEqual(mossCanvasPoint(10, 20, rect, size), [-300, -400])
  assert.deepEqual(mossCanvasPoint(810, 620, rect, size), [300, 400])
  assert.deepEqual(mossCanvasPoint(410, 320, rect, size), [0, 0])
  const oddSize = parseMossImageSize('5x3')
  assert.deepEqual(mossCanvasPoint(5, 10, { left: 5, top: 10, width: 5, height: 3 }, oddSize), [-1, -2])
  assert.deepEqual(mossCanvasPoint(10, 13, { left: 5, top: 10, width: 5, height: 3 }, oddSize), [1, 2])

  const source = [
    '# keep this note',
    '# keep this second note too',
    '{',
    '  "summary": {"meta": ["poster"], "canvas": "single", "caption": "A poster.", "extra_summary": 7},',
    '  "items": [{"label": "text", "caption": "Title", "text": "EXPLORE", "bbox": [-200, -300, 0, 300], "extra_item": {"keep": true}}],',
    '  "bbox_coord_space": "abs_centralized",',
    '  "bbox_format": "yxyx",',
    '  "extra_root": [1, 2, 3]',
    '}',
  ].join('\n')
  const parsed = parseMossImageLayout(source)
  assert.equal(parsed.plain, false)
  assert.deepEqual(parsed.comments, ['# keep this note', '# keep this second note too'])
  const saved = serializeMossImageLayout(parsed.document, parsed.comments)
  assert.ok(saved.startsWith('# keep this note\n# keep this second note too\n'))
  const roundTrip = parseMossImageLayout(saved)
  assert.deepEqual(roundTrip.document, parsed.document)
  assert.deepEqual(roundTrip.comments, parsed.comments)
  assert.deepEqual(roundTrip.document.extra_root, [1, 2, 3])
  assert.equal(roundTrip.document.summary.extra_summary, 7)
  assert.deepEqual(roundTrip.document.items[0].extra_item, { keep: true })

  const plainCaption = 'A red bicycle on a quiet street.\nKeep this wording.'
  const plain = parseMossImageLayout(plainCaption)
  assert.equal(plain.plain, true)
  assert.equal(plain.document.summary.caption, plainCaption)
  assert.deepEqual(plain.document, emptyMossImageDocument(plainCaption))
  const commentedPlain = parseMossImageLayout('# keep this note\n# and this one\nA quiet garden.')
  assert.equal(commentedPlain.document.summary.caption, 'A quiet garden.')
  assert.deepEqual(commentedPlain.comments, ['# keep this note', '# and this one'])
  const commentedPlainRoundTrip = parseMossImageLayout(serializeMossImageLayout(commentedPlain.document, commentedPlain.comments))
  assert.equal(commentedPlainRoundTrip.document.summary.caption, 'A quiet garden.')
  assert.deepEqual(commentedPlainRoundTrip.comments, commentedPlain.comments)

  assert.equal(validMossImageBox([-300, -400, 300, 400], size), true)
  assert.equal(validMossImageBox([-301, -10, 0, 10], size), false)
  assert.equal(validMossImageBox([0, 0, 0, 10], size), false)
  assert.equal(validMossImageBox([0, 10, 10, 0], size), false)
  assert.equal(validMossImageBox([0, 0, 10, Number.NaN], size), false)
  assert.equal(validMossImageBox([-1.5, 0, 1.5, 10], size), false)
  assert.match(mossImageBoxProblem([-1.5, 0, 1.5, 10], size), /integer pixel corners/)
  assert.deepEqual(fitMossImageBoxToCanvas([-350, -500, -250, -300], size), [-300, -400, -200, -200])
  assert.ok(fitMossImageBoxToCanvas([-350.5, -500.5, -250.5, -300.5], size).every(Number.isInteger))

  const malformed = '{"summary":{"caption":"keep me"},"items":[ '
  assert.throws(() => parseMossImageLayout(malformed), /Original prompt text is preserved/)
  assert.throws(() => parseMossImageLayout('{"summary":{},"items":[]} trailing'), /Original prompt text is preserved/)

  console.log('MOSS Image centered coordinates, rectangular canvas, comment and extra-field round-trip, plain prompts, box validation and malformed JSON preservation passed.')
})().catch(error => {
  console.error(error)
  process.exitCode = 1
})
