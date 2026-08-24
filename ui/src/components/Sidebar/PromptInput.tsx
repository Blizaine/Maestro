import { useState, useRef, useEffect, useLayoutEffect } from 'react'
import { Sparkles, Loader2, ChevronDown, ChevronUp, Brain, PenLine, RefreshCw } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  effectiveH3OmniSequenceFrames,
  h3OmniSequenceWindowCount,
  h3TimelineFrames,
} from '../../lib/h3Memory'

const placeholders: Record<string, string> = {
  image: 'Describe your image...',
  video: 'Describe your video...',
  audio: 'Enter text to speak or describe audio...',
  avatar: 'Describe your avatar animation...',
}

const H3_TEXT_TOKEN_LIMIT = 512
const H3_TEXT_TOKEN_TARGET = 480

/** Conservative browser-side estimate used while the user is typing. The
 * backend uses H3's exact Qwen tokenizer before generation. */
function estimateH3TextTokens(value: string): number {
  const lexical = value.match(/[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[^\w\s]/g)?.length ?? 0
  return Math.ceil(lexical * 1.25) + (value.trim() ? 8 : 0)
}

function useAutoGrowingTextarea(value: string) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const fitToContent = () => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    // scrollHeight includes padding but not the two one-pixel borders used
    // by these border-box textareas. Include them so the final line is visible.
    textarea.style.height = `${textarea.scrollHeight + 2}px`
  }

  useLayoutEffect(fitToContent, [value])
  useEffect(() => {
    window.addEventListener('resize', fitToContent)
    return () => window.removeEventListener('resize', fitToContent)
  }, [])

  return textareaRef
}

function H3WindowPromptTextarea({
  value,
  onChange,
  readOnly,
  title,
  active,
}: {
  value: string
  onChange: (value: string) => void
  readOnly: boolean
  title: string
  active: boolean
}) {
  const textareaRef = useAutoGrowingTextarea(value)

  return (
    <textarea
      ref={textareaRef}
      rows={1}
      value={value}
      onChange={event => onChange(event.target.value)}
      readOnly={readOnly}
      title={title}
      className={`w-full min-h-[92px] resize-none overflow-hidden bg-bg-secondary border rounded px-2 py-1.5 text-[10px] leading-relaxed text-text-secondary focus:outline-none focus:border-accent-blue ${
        active ? 'border-accent-blue/70 bg-accent-blue/5' : 'border-border'
      }`}
    />
  )
}

function useEnhanceStatus(isEnhancing: boolean) {
  const [status, setStatus] = useState<{ phase: 'loading' | 'thinking' | 'writing' | 'idle'; chars: number }>({ phase: 'idle', chars: 0 })

  useEffect(() => {
    if (!isEnhancing) {
      setStatus({ phase: 'idle', chars: 0 })
      return
    }
    setStatus({ phase: 'loading', chars: 0 })
    let active = true
    const poll = async () => {
      let streamStarted = false
      while (active) {
        try {
          // Check if LLM is still loading
          if (!streamStarted) {
            const llmRes = await fetch('/api/v1/llm/status')
            if (llmRes.ok && active) {
              const llmData = await llmRes.json()
              if (!llmData.loaded) {
                setStatus({ phase: 'loading', chars: 0 })
                await new Promise(r => setTimeout(r, 800))
                continue
              }
            }
          }
          const res = await fetch('/api/v1/llm/stream-status')
          if (res.ok && active) {
            const data = await res.json()
            const text = (data.text || '') as string
            if (text.length > 0) streamStarted = true
            const hasThinking = text.includes('<think>') || text.includes('<thinking>')
            const thinkingClosed = text.includes('</think>') || text.includes('</thinking>')
            if (hasThinking && !thinkingClosed) {
              setStatus({ phase: 'thinking', chars: text.length })
            } else if (text.length > 0) {
              setStatus({ phase: 'writing', chars: text.length })
            } else if (!streamStarted) {
              setStatus({ phase: 'loading', chars: 0 })
            }
            if (data.done) break
          }
        } catch { /* ignore */ }
        await new Promise(r => setTimeout(r, 800))
      }
    }
    poll()
    return () => { active = false }
  }, [isEnhancing])

  return status
}

export function PromptInput() {
  const prompt = useStore(s => s.params.prompt)
  const promptTextareaRef = useAutoGrowingTextarea(prompt)
  const setParam = useStore(s => s.setParam)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const enhancePrompt = useStore(s => s.enhancePrompt)
  const isEnhancing = useStore(s => s.isEnhancing)
  const promptEnhanceError = useStore(s => s.promptEnhanceError)
  const durationSeconds = useStore(s => s.durationSeconds)
  const slidingWindowSeconds = useStore(s => s.slidingWindowSeconds)
  const slidingWindowOverlap = useStore(s => s.slidingWindowOverlap)
  const slidingWindowLocked = useStore(s => s.slidingWindowLocked)
  const modelOptions = useStore(s => s.modelOptions)
  const resolution = useStore(s => s.params.resolution)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const imageMode = useStore(s => s.params.image_mode)
  const h3CameraCoverage = useStore(s => s.params.minimax_h3_camera_coverage || 'auto')
  const h3FirstLastMultiWindow = useStore(s => s.params.minimax_h3_multi_window === true)
  const h3WindowPlanningEnabled = useStore(s => s.params.minimax_h3_window_storyboard !== false)
  const h3ReferenceSequenceEnabled = useStore(s => s.params.minimax_h3_reference_sequence === true)
  const h3ManualSequencePrompts = useStore(s => s.params.minimax_h3_sequence_prompt_mode === 'manual')
  const h3NativeSequence = useStore(s => s.params.minimax_h3_sequence_continuity !== false)
  const ltxMultiWindow = useStore(s => s.params.ltx_multi_window === true)
  const ltxManualWindowPrompts = useStore(s => s.params.ltx_window_prompt_mode === 'manual')
  const h3WindowPlan = useStore(s => s.h3WindowPlan)
  const updateH3WindowPrompt = useStore(s => s.updateH3WindowPrompt)
  const activeH3JobPhase = useStore(s => {
    const job = s.jobs.find(item => (
      (item.status === 'queued' || item.status === 'running')
      && !!item.h3WindowPlan
    ))
    return job ? (job.phase || job.message || '') : ''
  })
  const activeH3JobPlanSignature = useStore(s => s.jobs.find(item => (
    (item.status === 'queued' || item.status === 'running')
    && !!item.h3WindowPlan
  ))?.h3WindowPlan?.signature || '')
  const [ttsMenuOpen, setTtsMenuOpen] = useState(false)
  const [windowPlanOpen, setWindowPlanOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  const isAudioOnly = modelOptions?.audio_only
  const voiceCount = useStore(s => s.ttsVoiceCount)
  const isMultiVoice = voiceCount >= 2
  // Does the active TTS model support multi-speaker output? Scenema, Kugel,
  // Qwen3-TTS, Index-TTS2 all do (max_voice_count >= 2 in their handlers).
  // Single-speaker-only engines leave it undefined; default 6 is the legacy
  // "any multi-speaker engine" assumption. Falling back to >1 keeps both
  // dialogue and monologue enhance available unless a model declares itself
  // single-speaker.
  const maxVoiceCount = ((modelOptions as { max_voice_count?: number } | null)?.max_voice_count) ?? 6
  const supportsDialogue = maxVoiceCount > 1
  // Main Sparkles button default: dialogue when the user has actually added
  // 2+ voice slots, monologue otherwise. The dropdown lets the user override
  // either way regardless of voice slot count.
  const defaultMode: 'dialogue' | 'monologue' = isMultiVoice ? 'dialogue' : 'monologue'
  const enhanceStatus = useEnhanceStatus(isEnhancing)
  const fps = modelOptions?.fps ?? 16
  const swDefaults = (modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
  const discardFrames = swDefaults?.discard_last_frames ?? 0
  const overlapSec = slidingWindowOverlap / fps
  const discardSec = discardFrames / fps
  const stride = slidingWindowSeconds - discardSec - overlapSec
  const supportsSlidingWindows = modelOptions?.sliding_window === true
  const isH3FirstLast = (
    String(modelOptions?.architecture || '').startsWith('minimax_h3')
    && modelOptions?.omni_reference !== true
  )
  const isLtxSequence = modelOptions?.multi_window_sequence_controls === true
  const windowCount = supportsSlidingWindows
    && (!isH3FirstLast || h3FirstLastMultiWindow)
    && (!isLtxSequence || ltxMultiWindow)
    && stride > 0
    && durationSeconds > slidingWindowSeconds
    ? 1 + Math.ceil((durationSeconds - slidingWindowSeconds + discardSec) / stride)
    : 1
  const usesWindows = generationMode === 'video' && supportsSlidingWindows && windowCount > 1 && imageMode !== 2
  const usesH3WindowPlanner = (
    usesWindows
    && modelOptions?.sliding_window_auto_prompt_pacing === true
    && h3WindowPlanningEnabled
  )
  const nativeMaximumFrames = modelOptions?.frames_maximum ?? null
  const sequenceClipFrames = nativeMaximumFrames != null
    ? effectiveH3OmniSequenceFrames({
        policy: modelOptions?.omni_sequence_memory_policy,
        resolution,
        totalVramGb,
        minimumFrames: modelOptions?.frames_minimum ?? 124,
        maximumFrames: nativeMaximumFrames,
        frameStep: modelOptions?.frames_steps ?? 17,
        selectedFrames: Math.round(slidingWindowSeconds * fps),
        manualOverride: slidingWindowLocked,
      }).frames
    : null
  const h3SequenceEnabled = (
    generationMode === 'video'
    && modelOptions?.omni_reference === true
    && h3ReferenceSequenceEnabled
    && sequenceClipFrames != null
  )
  const h3SequenceTotalFrames = h3TimelineFrames(
    durationSeconds,
    fps,
    modelOptions?.frames_maximum,
  )
  const h3SequenceNeedsMultiplePasses = (
    h3SequenceEnabled
    && sequenceClipFrames != null
    && h3SequenceTotalFrames > sequenceClipFrames
  )
  const usesH3ManualSequence = h3SequenceEnabled && h3ManualSequencePrompts
  const usesH3ManualFirstLast = (
    usesWindows
    && isH3FirstLast
    && h3FirstLastMultiWindow
    && !h3WindowPlanningEnabled
  )
  const usesH3ManualPrompts = usesH3ManualSequence || usesH3ManualFirstLast
  const usesLtxManualPrompts = (
    usesWindows
    && isLtxSequence
    && ltxMultiWindow
    && ltxManualWindowPrompts
  )
  const usesManualWindowPrompts = usesH3ManualPrompts || usesLtxManualPrompts
  const usesH3SequencePlanner = h3SequenceNeedsMultiplePasses && !h3ManualSequencePrompts
  const sequenceClipCount = h3SequenceEnabled && sequenceClipFrames
    ? h3OmniSequenceWindowCount({
        totalFrames: h3SequenceTotalFrames,
        windowFrames: sequenceClipFrames,
        overlapFrames: slidingWindowOverlap,
        nativeContinuation: h3NativeSequence,
    })
    : 1
  const manualPromptLineCount = prompt.split('\n').filter(line => line.trim()).length
  const manualPromptCount = (usesH3ManualFirstLast || usesLtxManualPrompts)
    ? windowCount
    : sequenceClipCount
  const manualPromptUnit = (usesH3ManualFirstLast || usesLtxManualPrompts)
    ? 'window'
    : (h3NativeSequence ? 'window' : 'clip')
  const usesH3Plan = usesH3WindowPlanner || usesH3SequencePlanner
  const h3FirstLastPromptUnits = isH3FirstLast && !usesH3Plan
    ? (usesH3ManualFirstLast
        ? prompt.split('\n').map(line => line.trim()).filter(Boolean)
        : [prompt.trim()].filter(Boolean))
    : []
  const h3FirstLastPromptCounts = h3FirstLastPromptUnits.map(estimateH3TextTokens)
  const largestH3PromptCount = h3FirstLastPromptCounts.length
    ? Math.max(...h3FirstLastPromptCounts)
    : 0
  const largestH3PromptIndex = h3FirstLastPromptCounts.indexOf(largestH3PromptCount)
  const expectedPlanCount = usesH3SequencePlanner ? sequenceClipCount : windowCount
  const expectedWindowFrames = usesH3SequencePlanner
    ? Math.max(1, Number(sequenceClipFrames || 1))
    : Math.max(1, Math.round(slidingWindowSeconds * fps))
  const h3PlanIsStale = !!h3WindowPlan && (
    h3WindowPlan.source_prompt.trim() !== prompt.trim()
    || h3WindowPlan.window_count !== expectedPlanCount
    || h3WindowPlan.total_frames !== Math.max(1, Math.round(durationSeconds * fps))
    || h3WindowPlan.window_frames !== expectedWindowFrames
    || (h3WindowPlan.camera_coverage || 'auto') !== h3CameraCoverage
    || (usesH3SequencePlanner
      ? h3WindowPlan.plan_kind !== 'reference_sequence'
      : h3WindowPlan.plan_kind === 'reference_sequence')
    || (usesH3SequencePlanner
      && !!h3WindowPlan.native_continuation !== h3NativeSequence)
    || (usesH3SequencePlanner
      && h3NativeSequence
      && Number(h3WindowPlan.overlap_frames || 0) !== slidingWindowOverlap)
  )
  const matchingActiveH3Phase = (
    h3WindowPlan?.signature === activeH3JobPlanSignature
      ? activeH3JobPhase
      : ''
  )
  const activeWindowMatch = matchingActiveH3Phase.match(/(?:Sliding Window|Clip)\s+(\d+)\/(\d+)/i)
  const activeH3Window = activeWindowMatch ? Number(activeWindowMatch[1]) : null
  const modePlaceholder = generationMode === 'avatar' && editSubMode === 'recast'
    ? 'Describe the finished video and replacement characters...'
    : generationMode === 'avatar' && editSubMode === 'restyle'
      ? 'Describe the finished video...'
      : (placeholders[generationMode] || 'Describe your content...')

  // Close TTS menu on outside click
  useEffect(() => {
    if (!ttsMenuOpen) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setTtsMenuOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [ttsMenuOpen])

  // A server-created plan used to arrive collapsed, making the exact prompts
  // effectively invisible once an expensive generation had started. Open a
  // newly planned storyboard once; the user can still collapse it afterward.
  useEffect(() => {
    if (usesH3Plan && h3WindowPlan?.signature) {
      setWindowPlanOpen(true)
    }
  }, [usesH3Plan, h3WindowPlan?.signature])

  // Keep the prompt area at the bottom when the sidebar has spare room. The
  // textarea itself grows to its complete content height, and the sidebar's
  // outer scroller handles prompts taller than the viewport.
  return (
    <div className="relative grow shrink-0 flex flex-col">
      {/* Enhance status indicator */}
      {isEnhancing && enhanceStatus.phase !== 'idle' && (
        <div className="flex items-center gap-1.5 px-2 py-1 text-[10px] text-text-muted bg-bg-tertiary/80 rounded-t-lg border border-b-0 border-border">
          {enhanceStatus.phase === 'loading' ? (
            <>
              <Loader2 size={10} className="text-text-muted animate-spin" />
              <span>Loading LLM...</span>
            </>
          ) : enhanceStatus.phase === 'thinking' ? (
            <>
              <Brain size={10} className="text-chip-purple animate-pulse" />
              <span>Thinking...</span>
            </>
          ) : (
            <>
              <PenLine size={10} className="text-accent-blue animate-pulse" />
              <span>Writing...</span>
            </>
          )}
        </div>
      )}
      {!isEnhancing && promptEnhanceError && (
        <div
          role="alert"
          className="mb-1.5 rounded-lg border border-indicator-error/40 bg-indicator-error/10 px-2.5 py-1.5 text-[10px] text-indicator-error"
        >
          {promptEnhanceError}
        </div>
      )}
      {usesH3Plan && h3WindowPlan && (
        <div className="mb-1.5">
          <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-border bg-bg-tertiary/70">
            <button
              type="button"
              onClick={() => setWindowPlanOpen(open => !open)}
              className="flex-1 min-w-0 flex items-center gap-1.5 text-left"
              title="Review the complete Context-IR prompt assigned to each H3 continuation window."
            >
              {windowPlanOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              <span className="text-[10px] font-medium text-text-secondary truncate">
                Exact H3 prompts · {h3WindowPlan.window_count} {usesH3SequencePlanner && !h3NativeSequence ? 'clips' : 'windows'}
              </span>
              {h3PlanIsStale && (
                <span className="text-[9px] text-amber-400">Needs update</span>
              )}
              {h3WindowPlan.planned_by === 'deterministic_fallback' && (
                <span className="text-[9px] text-amber-400">Fallback</span>
              )}
            </button>
            <button
              type="button"
              onClick={() => enhancePrompt()}
              disabled={isEnhancing}
              title={`Rebuild the H3 ${usesH3SequencePlanner ? 'reference sequence' : 'window plan'} from the current idea and timing.`}
              className="p-1 text-text-muted hover:text-accent-blue disabled:opacity-50"
            >
              <RefreshCw size={11} className={isEnhancing ? 'animate-spin' : ''} />
            </button>
          </div>
          {!!h3WindowPlan.planning_warnings?.length && (
            <div
              role="alert"
              className="mt-1.5 rounded-lg border border-amber-400/35 bg-amber-400/10 px-2.5 py-2 text-[10px] leading-relaxed text-amber-300"
            >
              <div className="font-medium">Maestro repaired this H3 plan</div>
              {h3WindowPlan.planning_warnings.map((warning, index) => (
                <div key={`${index}-${warning}`} className="mt-0.5">
                  {warning}
                </div>
              ))}
              <div className="mt-1 text-text-muted">
                Review the exact window prompts below or press refresh to try the AI planner again.
              </div>
            </div>
          )}
          {windowPlanOpen && (
            <div className="mt-2 space-y-3">
              {h3WindowPlan.windows.map((window, index) => (
                <div
                  key={`${window.index}-${window.start_frame}`}
                  className="space-y-1"
                >
                  <div className={`flex items-center justify-between text-[9px] ${
                    activeH3Window === window.index ? 'text-accent-blue' : 'text-text-muted'
                  }`}>
                    <span>
                      {usesH3SequencePlanner && !h3NativeSequence ? 'Clip' : 'Window'} {window.index}: {window.title || `Beat ${window.index}`}
                      {activeH3Window === window.index ? ' · Generating now' : ''}
                    </span>
                    <span className={usesH3WindowPlanner ? (() => {
                      const tokens = estimateH3TextTokens(window.prompt)
                      return tokens > H3_TEXT_TOKEN_LIMIT
                        ? 'text-indicator-error'
                        : tokens > H3_TEXT_TOKEN_TARGET
                          ? 'text-amber-400'
                          : ''
                    })() : ''}>
                      {window.start_seconds.toFixed(1)}–{window.end_seconds.toFixed(1)}s
                      {usesH3WindowPlanner && ` · ~${estimateH3TextTokens(window.prompt)}/${H3_TEXT_TOKEN_LIMIT} tokens`}
                    </span>
                  </div>
                  <H3WindowPromptTextarea
                    value={window.prompt}
                    onChange={value => updateH3WindowPrompt(index, value)}
                    readOnly={!!matchingActiveH3Phase}
                    title={matchingActiveH3Phase
                      ? 'This is the exact prompt already submitted for the active generation.'
                      : 'Edit this exact window prompt before the next generation.'}
                    active={activeH3Window === window.index}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {usesManualWindowPrompts && (
        <div className="mb-1.5 flex items-center justify-between gap-2 text-[10px] text-text-muted">
          <span>One non-empty line per {manualPromptUnit}</span>
          <span className={manualPromptLineCount === manualPromptCount ? 'text-text-secondary' : 'text-amber-400'}>
            {manualPromptLineCount}/{manualPromptCount} prompts
          </span>
        </div>
      )}
      {largestH3PromptCount > H3_TEXT_TOKEN_TARGET && (
        <div
          role="alert"
          className={`mb-1.5 rounded-lg border px-2.5 py-1.5 text-[10px] leading-relaxed ${
            largestH3PromptCount > H3_TEXT_TOKEN_LIMIT
              ? 'border-indicator-error/40 bg-indicator-error/10 text-indicator-error'
              : 'border-amber-400/35 bg-amber-400/10 text-amber-300'
          }`}
        >
          {h3FirstLastPromptCounts.length > 1 ? `Window ${largestH3PromptIndex + 1}` : 'This H3 prompt'} is approximately {largestH3PromptCount}/{H3_TEXT_TOKEN_LIMIT} text tokens.{' '}
          {largestH3PromptCount > H3_TEXT_TOKEN_LIMIT
            ? 'First / Last would cut off the ending. Shorten it or use Auto planning / Prompt Enhance; Maestro will reject it rather than silently lose dialogue.'
            : 'It fits, but is close to H3’s cutoff. Prompt Enhance targets a safer compact form.'}
        </div>
      )}
      <div className="relative mt-auto">
        <textarea
          ref={promptTextareaRef}
          rows={1}
          value={prompt}
          onChange={e => setParam('prompt', e.target.value)}
          placeholder={usesManualWindowPrompts
            ? `Line 1 = ${manualPromptUnit} 1, line 2 = ${manualPromptUnit} 2... (${manualPromptCount} total)`
            : usesH3Plan
            ? `Describe the complete video idea—Maestro will plan ${expectedPlanCount} H3 ${usesH3SequencePlanner ? 'reference clips' : 'windows'}.`
            : usesWindows
              ? (isLtxSequence
                  ? `Describe the complete video idea - Maestro will plan ${windowCount} LTX windows.`
                  : `Line 1 = window 1, line 2 = window 2... (${windowCount} windows)`)
            : modePlaceholder}
          className="block w-full min-h-[112px] resize-none overflow-hidden bg-bg-tertiary border border-border rounded-lg px-3 py-2 pr-10 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue transition-colors"
        />
        {prompt.trim() && !usesManualWindowPrompts && (
        isAudioOnly ? (
          /* TTS: mode-aware split button. Main button uses default mode based
             on voice-slot count; dropdown exposes both Speech and Dialogue
             explicitly so the user can override regardless of voice count.
             Previously the dropdown labels switched with isMultiVoice, leaving
             no way to enhance into dialogue format without first adding voice
             slots — bad UX trap especially with audio_mode_from_voice_count
             models like Scenema where the user may want a generated-voice
             dialogue script as a starting point. */
          <div ref={menuRef} className="absolute right-2 bottom-2">
            <div className="flex items-center">
              <button
                onClick={() => enhancePrompt(defaultMode)}
                disabled={isEnhancing}
                title={isMultiVoice
                  ? `Write ${voiceCount}-person dialogue (use dropdown to switch to speech)`
                  : 'Write a speech (use dropdown to switch to dialogue)'}
                className="p-1.5 rounded-l-md text-text-muted hover:text-accent-blue hover:bg-bg-hover transition-colors disabled:opacity-50"
              >
                {isEnhancing ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
              </button>
              <button
                onClick={() => setTtsMenuOpen(!ttsMenuOpen)}
                disabled={isEnhancing}
                className="p-1.5 rounded-r-md text-text-muted hover:text-accent-blue hover:bg-bg-hover transition-colors disabled:opacity-50 border-l border-border"
              >
                <ChevronUp size={10} />
              </button>
            </div>
            {ttsMenuOpen && (
              <div className="absolute bottom-full right-0 mb-1 bg-bg-secondary border border-border rounded-lg shadow-lg overflow-hidden min-w-[220px] z-50">
                <button
                  onClick={() => { setTtsMenuOpen(false); enhancePrompt('monologue') }}
                  className="w-full text-left px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors"
                >
                  Write Speech
                  <span className="block text-[9px] text-text-muted">Single speaker, with thinking</span>
                </button>
                <button
                  onClick={() => { setTtsMenuOpen(false); enhancePrompt('monologue_fast') }}
                  className="w-full text-left px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors border-t border-border"
                >
                  Write Speech
                  <span className="block text-[9px] text-text-muted">Single speaker, faster</span>
                </button>
                {supportsDialogue && (
                  <>
                    <button
                      onClick={() => { setTtsMenuOpen(false); enhancePrompt('dialogue') }}
                      className="w-full text-left px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors border-t border-border"
                    >
                      {voiceCount >= 2 ? `Write ${voiceCount}-Person Dialogue` : 'Write Dialogue (2 speakers)'}
                      <span className="block text-[9px] text-text-muted">With thinking — more creative</span>
                    </button>
                    <button
                      onClick={() => { setTtsMenuOpen(false); enhancePrompt('dialogue_fast') }}
                      className="w-full text-left px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors border-t border-border"
                    >
                      {voiceCount >= 2 ? `Write ${voiceCount}-Person Dialogue` : 'Write Dialogue (2 speakers)'}
                      <span className="block text-[9px] text-text-muted">No thinking — faster</span>
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={() => enhancePrompt()}
            disabled={isEnhancing}
            title="Enhance prompt with AI"
            className="absolute right-2 bottom-2 p-1.5 rounded-md text-text-muted hover:text-accent-blue hover:bg-bg-hover transition-colors disabled:opacity-50"
          >
            {isEnhancing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Sparkles size={14} />
            )}
          </button>
        )
        )}
      </div>
    </div>
  )
}
