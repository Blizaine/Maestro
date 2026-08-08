import { useState, useRef, useEffect, useLayoutEffect } from 'react'
import { Sparkles, Loader2, ChevronDown, ChevronUp, Brain, PenLine, RefreshCw } from 'lucide-react'
import { useStore } from '../../stores/useStore'

const placeholders: Record<string, string> = {
  image: 'Describe your image...',
  video: 'Describe your video...',
  audio: 'Enter text to speak or describe audio...',
  avatar: 'Describe your avatar animation...',
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const fitToContent = () => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    // scrollHeight includes padding but not the two one-pixel borders used
    // by this border-box textarea. Include them so the final line never clips.
    textarea.style.height = `${textarea.scrollHeight + 2}px`
  }

  useLayoutEffect(fitToContent, [value])
  useEffect(() => {
    window.addEventListener('resize', fitToContent)
    return () => window.removeEventListener('resize', fitToContent)
  }, [])

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
  const setParam = useStore(s => s.setParam)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const enhancePrompt = useStore(s => s.enhancePrompt)
  const isEnhancing = useStore(s => s.isEnhancing)
  const durationSeconds = useStore(s => s.durationSeconds)
  const slidingWindowSeconds = useStore(s => s.slidingWindowSeconds)
  const slidingWindowOverlap = useStore(s => s.slidingWindowOverlap)
  const modelOptions = useStore(s => s.modelOptions)
  const imageMode = useStore(s => s.params.image_mode)
  const h3WindowPlanningEnabled = useStore(s => s.params.minimax_h3_window_storyboard !== false)
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
  const windowCount = supportsSlidingWindows && stride > 0 && durationSeconds > slidingWindowSeconds
    ? 1 + Math.ceil((durationSeconds - slidingWindowSeconds + discardSec) / stride)
    : 1
  const usesWindows = generationMode === 'video' && supportsSlidingWindows && windowCount > 1 && imageMode !== 2
  const usesH3WindowPlanner = (
    usesWindows
    && modelOptions?.sliding_window_auto_prompt_pacing === true
    && h3WindowPlanningEnabled
  )
  const h3PlanIsStale = !!h3WindowPlan && (
    h3WindowPlan.source_prompt.trim() !== prompt.trim()
    || h3WindowPlan.window_count !== windowCount
    || h3WindowPlan.total_frames !== Math.max(1, Math.round(durationSeconds * fps))
    || h3WindowPlan.window_frames !== Math.max(1, Math.round(slidingWindowSeconds * fps))
  )
  const matchingActiveH3Phase = (
    h3WindowPlan?.signature === activeH3JobPlanSignature
      ? activeH3JobPhase
      : ''
  )
  const activeWindowMatch = matchingActiveH3Phase.match(/Sliding Window\s+(\d+)\/(\d+)/i)
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
    if (usesH3WindowPlanner && h3WindowPlan?.signature) {
      setWindowPlanOpen(true)
    }
  }, [usesH3WindowPlanner, h3WindowPlan?.signature])

  // grow shrink-0: fill spare vertical space when the sidebar is roomy, but
  // never shrink below the textarea's min-height. Dropping the old
  // `flex-1 min-h-0` stops the wrapper from collapsing under the textarea
  // (which made it overflow and overlap the section below).
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
      {usesH3WindowPlanner && h3WindowPlan && (
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
                Exact H3 prompts · {h3WindowPlan.window_count} windows
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
              title="Rebuild the H3 window plan from the current idea and timing."
              className="p-1 text-text-muted hover:text-accent-blue disabled:opacity-50"
            >
              <RefreshCw size={11} className={isEnhancing ? 'animate-spin' : ''} />
            </button>
          </div>
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
                      Window {window.index}: {window.title || `Beat ${window.index}`}
                      {activeH3Window === window.index ? ' · Generating now' : ''}
                    </span>
                    <span>{window.start_seconds.toFixed(1)}–{window.end_seconds.toFixed(1)}s</span>
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
      <textarea
        value={prompt}
        onChange={e => setParam('prompt', e.target.value)}
        placeholder={usesH3WindowPlanner
          ? `Describe the complete video idea—Maestro will plan ${windowCount} H3 windows.`
          : usesWindows
            ? `Line 1 = window 1, line 2 = window 2... (${windowCount} windows)`
          : modePlaceholder}
        className="w-full flex-1 bg-bg-tertiary border border-border rounded-lg px-3 py-2 pr-10 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue transition-colors"
        style={{ resize: 'none', minHeight: 112 }}
      />
      {prompt.trim() && (
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
  )
}
