import { useState, useCallback, useRef, useMemo, useEffect } from 'react'
import { Upload, Loader2, Music, RotateCcw, Check, X, ChevronRight, ChevronDown, ImageIcon, Play, Film, Mic, Sparkles, Send, Users, FileText, ListVideo } from 'lucide-react'
import { useStore, directorModelUsesFixedMediaStrength, getFamiliesForMode, getModelsForFamily, resolveResolution } from '../../stores/useStore'
import { fetchModelOptions, getFileUrl } from '../../api/client'
import { DirectorLoraSelector } from '../SettingsDrawer/DirectorLoraSelector'
import { DirectorSongSetup } from './DirectorSongSetup'
import { DirectorH3Optimizations } from './DirectorH3Optimizations'
import { OmniReferenceSection } from './OmniReferenceSection'
import { InfoTooltip } from './InfoTooltip'
import { formatSeconds, recommendedWindowProfile } from './DurationSlider'
import { DurationPresetControl } from './DurationPresetControl'
import { LONG_FORM_MAX_SECONDS, formatDuration } from '../../lib/durationPlanning'
import { formatEstimatedClock, formatEtaDuration } from '../../lib/format'
import type { DirectorPipelineType, DirectorShotImageGuidance, DirectorSkill, ModelOptions, ShortFilmCharacter, ShortFilmPath } from '../../types'

// AUDIO_ACCEPT lists both audio formats AND video formats. When a video
// file is uploaded, the backend's /api/v1/upload-audio endpoint extracts
// the audio track via ffmpeg and returns a WAV path. The user sees the
// same workflow either way — they can drop a music video here and get
// the soundtrack analyzed without converting first.
const AUDIO_ACCEPT = '.wav,.mp3,.flac,.ogg,.m4a,.mp4,.mov,.mkv,.webm,.avi,.m4v'
const IMAGE_ACCEPT = '.png,.jpg,.jpeg,.webp,.bmp'
const DIRECTOR_IMAGE_MODEL_NONE = '__none__'

function DirectorTargetDurationControl() {
  const duration = useStore(s => s.shortFilmTargetDuration)
  const setDuration = useStore(s => s.shortFilmSetTargetDuration)
  const prompt = useStore(s => s.directorSceneDescription)
  const references = useStore(s => s.directorH3References)
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const resolution = useStore(s => s.directorResolution)
  const aspectRatio = useStore(s => s.directorAspectRatio)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const [options, setOptions] = useState<ModelOptions | null>(null)
  const [planningMode, setPlanningMode] = useState<'duration' | 'windows' | 'auto'>('auto')

  useEffect(() => {
    let cancelled = false
    fetchModelOptions(videoModel)
      .then(value => { if (!cancelled) setOptions(value) })
      .catch(() => { if (!cancelled) setOptions(null) })
    return () => { cancelled = true }
  }, [videoModel])

  const fps = options?.fps || 24
  const resolvedResolution = resolveResolution(options, resolution, aspectRatio)
  const recommendation = recommendedWindowProfile(
    options?.director_memory_policy || options?.sliding_window_memory_policy,
    resolvedResolution,
    totalVramGb,
  )
  const defaults = options?.sliding_window_defaults
  const windowFrames = recommendation?.frames
    || defaults?.window_max
    || options?.frames_maximum
    || Math.round(14.4 * fps)
  const windowSeconds = Math.max(1, windowFrames / fps)
  const driveReference = references.find(reference => (
    reference.type === 'audio' && reference.audio_intent === 'drive'
  ))
  const driveDuration = Number(driveReference?.duration_seconds)

  return (
    <DurationPresetControl
      label="Target duration"
      value={duration}
      onChange={setDuration}
      minSeconds={10}
      maxSeconds={LONG_FORM_MAX_SECONDS}
      windowSeconds={windowSeconds}
      overlapSeconds={(defaults?.overlap_default || 0) / fps}
      discardSeconds={(defaults?.discard_last_frames || 0) / fps}
      enablePlanningModes
      planningMode={planningMode}
      onPlanningModeChange={setPlanningMode}
      autoPrompt={prompt}
      autoPlanningStyle="creative"
      autoSourceSeconds={Number.isFinite(driveDuration) && driveDuration > 0 ? driveDuration : null}
      autoSourceLabel="music / performance timeline"
      modelLimitLabel={`Director plans ${formatDuration(duration, true)} as restart-safe scenes; current automatic shot target is ${formatDuration(windowSeconds, true)}.`}
    />
  )
}

function directorWillGenerateShotImages(
  support: 'required' | 'optional' | 'direct_references' | undefined,
  guidance: DirectorShotImageGuidance,
  hasVisualReferences: boolean,
): boolean {
  // Explicit choices from the Image model selector always win. In
  // particular, "None" maps to prompt_only even for Director models whose
  // legacy capability metadata says generated starts are required.
  if (guidance === 'prompt_only') return false
  if (guidance === 'generate') return true
  if (!support || support === 'required') return true
  if (support === 'direct_references') return false
  return hasVisualReferences
}

function AudioScaleSlider() {
  const audioScale = useStore(s => s.directorAudioScale)
  const setAudioScale = useStore(s => s.setDirectorAudioScale)
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-text-muted whitespace-nowrap">Audio {audioScale.toFixed(1)}x</span>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={audioScale}
          onChange={e => setAudioScale(parseFloat(e.target.value))}
          className="flex-1 h-1"
        />
      </div>
      <div className="flex gap-2 text-[8px] text-text-muted">
        <span>1x</span>
        <span>3x TTS</span>
        <span>5x</span>
      </div>
    </div>
  )
}

const STEP_ORDER = ['upload', 'analyze', 'structure', 'style', 'plan', 'review', 'generate_images', 'plan_video', 'review_video'] as const
type DirectorStep = typeof STEP_ORDER[number]

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

const sectionColors: Record<string, string> = {
  intro: 'bg-blue-500/20 text-chip-blue',
  verse: 'bg-green-500/20 text-chip-green',
  chorus: 'bg-purple-500/20 text-chip-purple',
  bridge: 'bg-yellow-500/20 text-chip-yellow',
  outro: 'bg-gray-500/20 text-chip-gray',
  instrumental: 'bg-cyan-500/20 text-chip-cyan',
  // Short film scene types
  dialogue: 'bg-green-500/20 text-chip-green',
  action: 'bg-orange-500/20 text-chip-orange',
  opening: 'bg-blue-500/20 text-chip-blue',
  closing: 'bg-gray-500/20 text-chip-gray',
  scene: 'bg-teal-500/20 text-chip-teal',
}

const sectionBarColors: Record<string, string> = {
  intro: 'bg-blue-500',
  verse: 'bg-green-500',
  chorus: 'bg-purple-500',
  bridge: 'bg-yellow-500',
  outro: 'bg-gray-500',
  instrumental: 'bg-cyan-500',
  // Short film scene types
  dialogue: 'bg-green-500',
  action: 'bg-orange-500',
  opening: 'bg-blue-500',
  closing: 'bg-gray-500',
  scene: 'bg-teal-500',
}

/**
 * Textarea that auto-resizes its height to fit the content.
 *
 * Used for the per-clip image_prompt and video_prompt fields in the
 * Director chat review steps. Without this, long prompts produce a
 * scrollable inner textarea — and that textarea sits inside another
 * scrollable container, inside the chat panel which is itself
 * scrollable. The user has to triple-scroll to read a long prompt.
 *
 * With auto-resize the textarea grows to its full content height and
 * the only scroll is the parent chat panel's, matching the user's
 * "one scroll per surface" preference.
 *
 * Re-measures whenever `value` changes (controlled-component pattern):
 * setting height to 'auto' first lets it shrink as well as grow.
 *
 * `overflow-y: hidden` is forced via inline style so the textarea
 * never shows its own scrollbar — even when the browser would render
 * one defensively at the boundary between content height and box
 * height (Firefox especially does this). Without `hidden`, a wheel
 * event over the textarea gets captured by the textarea's would-be
 * scroll instead of bubbling up to the chat panel, so the user
 * can't scroll the chat when their cursor happens to be over a
 * prompt field.
 *
 * Optional `minHeight`/`maxHeight` (px) bound the growth — used by the
 * chat composer (issue #11), which keeps its resting 2-row size when
 * empty and stops growing at a cap. Past the cap the textarea scrolls
 * itself, so overflow flips to `auto` there; that's fine for the
 * composer because it sits OUTSIDE the scrollable chat panel — the
 * wheel-capture concern above doesn't apply.
 */
function AutoResizeTextarea({ minHeight, maxHeight, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  minHeight?: number
  maxHeight?: number
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    let h = el.scrollHeight
    if (minHeight) h = Math.max(h, minHeight)
    if (maxHeight) h = Math.min(h, maxHeight)
    el.style.height = `${h}px`
    if (maxHeight) el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [props.value, minHeight, maxHeight])
  // Merge any incoming style with our scrollbar-hiding override.
  // OUR override comes last so it wins — wheel-capture is the whole
  // point of the component, can't let a caller silently break it.
  const mergedStyle: React.CSSProperties = { ...(props.style || {}), overflowY: 'hidden' }
  return <textarea ref={ref} {...props} style={mergedStyle} />
}

function SectionBadge({ label }: { label: string }) {
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${sectionColors[label] || 'bg-bg-hover text-text-muted'}`}>
      {label}
    </span>
  )
}

function EnergyDot({ energy }: { energy: number }) {
  const color = energy > 0.6 ? 'bg-chip-red' : energy < 0.3 ? 'bg-chip-blue' : 'bg-chip-yellow'
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} title={`Energy: ${(energy * 100).toFixed(0)}%`} />
}

// Chat bubble wrapper
function SystemBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-bg-tertiary/50 rounded-lg p-3 border border-border/50 space-y-2">
      {children}
    </div>
  )
}

function UserBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-accent-blue/10 rounded-lg p-3 ml-8 border border-accent-blue/20">
      {children}
    </div>
  )
}

function LlmThinkingStream({ stage }: { stage: string }) {
  const [streamText, setStreamText] = useState('')
  const [streamDone, setStreamDone] = useState(false)
  const [expanded, setExpanded] = useState(true)
  const appendLlmLog = useStore(s => s.directorAppendLlmLog)
  // Latest stream text, readable from inside the poll loop's closure
  // (streamText state would be stale there). Used to persist the finished
  // stream into the chat history on the streaming→done transition.
  const latestTextRef = useRef('')
  // Inner scroll container — kept fixed-height so a long thinking
  // dump doesn't blow up the entire chat panel. Auto-scrolls to the
  // bottom as new tokens arrive so the user always sees the latest
  // generation, just like a terminal tail.
  const streamScrollRef = useRef<HTMLDivElement>(null)

  // Poll the stream-status endpoint continuously for the lifetime of
  // the component. Two failure modes have to be handled:
  //
  //   1. We mount BEFORE the LLM starts streaming (rare but possible
  //      when the user clicks a button that kicks off planning). Need
  //      to wait for `done: false` before treating subsequent
  //      `done: true` as completion — otherwise we'd flash "done"
  //      immediately and never show any text.
  //
  //   2. Multiple LLM calls happen during the same component mount —
  //      e.g. short-film story mode where Pass 1 (screenplay) and
  //      Pass 2 (shot breakdown) both run while the chat sits at
  //      step='plan'. The previous implementation BROKE the polling
  //      loop after the first stream finished, so Pass 2 never showed.
  //      Now we keep polling: when we detect a fresh transition from
  //      done→streaming, we reset streamText and streamDone so the
  //      new stream renders cleanly from the top.
  //
  // The poll interval (400ms) is unchanged — fine for a streaming
  // text-display use case, server endpoint is cheap.
  useEffect(() => {
    let active = true
    const poll = async () => {
      // wasStreaming tracks the last poll's state so we can detect
      // transitions: streaming→done means current run finished;
      // done→streaming means a NEW LLM call has started.
      let wasStreaming = false
      while (active) {
        try {
          const res = await fetch('/api/v1/llm/stream-status')
          if (res.ok) {
            const data = await res.json()
            if (!active) break
            const isStreaming = !data.done
            // Transition: fresh stream started (done→streaming).
            // Mark not-done so the "thinking" indicator returns.
            // We don't wipe streamText here — the streamText update
            // below replaces it with the new stream's first tokens
            // in the same render, avoiding a flash of empty content.
            if (!wasStreaming && isStreaming) {
              setStreamDone(false)
            }
            // Transition: stream just finished (streaming→done).
            // Keep the final text visible, just stop the indicator —
            // and persist the completed stream into the chat history so
            // it survives this component unmounting at the next step.
            if (wasStreaming && !isStreaming) {
              setStreamDone(true)
              appendLlmLog(stage, data.text || latestTextRef.current)
            }
            // While actively streaming, push every chunk into state so
            // the user sees text grow live. Skip when idle so we don't
            // overwrite the previous stream's final text with stale or
            // empty backend buffer between runs.
            if (isStreaming) {
              setStreamText(data.text || '')
              latestTextRef.current = data.text || ''
            }
            wasStreaming = isStreaming
          }
        } catch { /* ignore — endpoint unreachable, retry next tick */ }
        await new Promise(r => setTimeout(r, 400))
      }
    }
    poll()
    return () => { active = false }
  }, [stage, appendLlmLog])

  // Auto-scroll the inner preview box to its bottom whenever new
  // tokens arrive. Uses scrollTop (NOT scrollIntoView) so we don't
  // also drag the outer chat panel — this is a self-contained tail.
  useEffect(() => {
    const el = streamScrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [streamText])

  // Separate thinking from output
  const thinkMatch = streamText.match(/<think>([\s\S]*?)(<\/think>|$)/)
  const thinking = thinkMatch ? thinkMatch[1].trim() : ''
  const isStillThinking = thinkMatch ? !thinkMatch[2].includes('</think>') : false
  const output = streamText.replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim()

  const displayText = thinking || output || ''
  if (!displayText) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        {isStillThinking ? 'Thinking...' : thinking && !output ? 'Thinking complete' : output ? 'Writing scenes...' : 'LLM Output'}
      </button>
      {expanded && (
        // Fixed-height preview box. It's a "what's happening right now"
        // tail, not a full transcript — capping the height keeps the
        // chat scroll position stable while the LLM streams long
        // thinking dumps. Inner scroll auto-tails to the bottom (see
        // streamScrollRef effect above).
        <div
          ref={streamScrollRef}
          className="mt-1 rounded bg-bg-primary/50 border border-border/30 p-2 max-h-32 overflow-y-auto"
        >
          {thinking && (
            <pre className="text-[10px] text-text-muted whitespace-pre-wrap font-mono leading-relaxed">
              {thinking}
              {isStillThinking && <span className="animate-pulse">|</span>}
            </pre>
          )}
          {output && (
            <pre className="text-[10px] text-accent-blue/70 whitespace-pre-wrap font-mono leading-relaxed mt-1 pt-1 border-t border-border/30">
              {output}
              {!streamDone && !isStillThinking && <span className="animate-pulse">|</span>}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

/** Collapsed, persistent record of completed LLM streams for one stage.
 *  Replaces the old behavior where the thinking/output box vanished the
 *  moment a stage finished. Default-collapsed so history stays compact. */
function LlmLogStage({ stage, label }: { stage: string; label: string }) {
  const log = useStore(s => s.directorLlmLog)
  const [openIdx, setOpenIdx] = useState<number | null>(null)
  const entries = log.filter(e => e.stage === stage)
  if (entries.length === 0) return null
  return (
    <div className="space-y-1">
      {entries.map((entry, i) => {
        const open = openIdx === i
        // Same thinking/output split as the live stream box
        const thinkMatch = entry.text.match(/<think>([\s\S]*?)(<\/think>|$)/)
        const thinking = thinkMatch ? thinkMatch[1].trim() : ''
        const output = entry.text.replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim()
        return (
          <div key={i}>
            <button
              onClick={() => setOpenIdx(open ? null : i)}
              className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
            >
              {open ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              {label}{entries.length > 1 ? ` · pass ${i + 1}` : ''} (done)
            </button>
            {open && (
              <div className="mt-1 rounded bg-bg-primary/50 border border-border/30 p-2 max-h-48 overflow-y-auto">
                {thinking && (
                  <pre className="text-[10px] text-text-muted whitespace-pre-wrap font-mono leading-relaxed">{thinking}</pre>
                )}
                {output && (
                  <pre className={`text-[10px] text-accent-blue/70 whitespace-pre-wrap font-mono leading-relaxed ${thinking ? 'mt-1 pt-1 border-t border-border/30' : ''}`}>{output}</pre>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function DirectorChat() {
  const step = useStore(s => s.directorStep)
  const loading = useStore(s => s.directorLoading)
  // Sub-status text ("Loading transcription model (first use downloads
  // ~300MB)...", "Transcribing audio...", etc.) updated by the polling
  // loop in directorUploadAndAnalyze. Falls back to a static message
  // in the loading spinner when null.
  const loadingMessage = useStore(s => s.directorLoadingMessage)
  // Tracks whether ANY generation job is currently running. Used to
  // gate the "Generate" button in the video-prompts review step so it
  // doesn't look pressable while the system is auto-generating (auto
  // mode) or already generating from a previous click (manual mode).
  const isGenerating = useStore(s => s.isGenerating)
  const error = useStore(s => s.directorError)
  const analysis = useStore(s => s.directorAnalysis)
  const plannedClips = useStore(s => s.directorPlannedClips)
  const energyBias = useStore(s => s.directorEnergyBias)
  const clipPlans = useStore(s => s.directorClipPlans)
  const selectedDirectorShotImageSupport = useStore(s => s.models.find(
    model => model.model_type === (s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'),
  )?.director?.shot_image_support)
  const directorUsesOmniManifest = useStore(s => {
    const selected = s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    return selected.toLowerCase().startsWith('minimax_h3_ref2va')
      || s.models.find(model => model.model_type === selected)
        ?.director?.video_strategy === 'omni_reference'
  })
  const directorShotImageGuidance = useStore(s => s.directorShotImageGuidance)
  const directorHasVisualReferences = useStore(s => Boolean(
    s.directorReferenceImage
    || s.directorReferenceImagePath
    || s.directorCharacterRefs.length
    || s.directorCharacterRefPaths.length
    || s.directorLocationRefs.length
    || s.directorLocationRefPaths.length
    || (
      s.models.find(model => model.model_type === (
        s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
      ))?.director?.video_strategy === 'omni_reference'
      && s.directorH3References.some(
        reference => reference.type === 'image' || reference.type === 'video',
      )
    )
  ))
  const sceneDescription = useStore(s => s.directorSceneDescription)
  const audioFile = useStore(s => s.directorAudioFile)
  const referenceImage = useStore(s => s.directorReferenceImage)
  const clipImages = useStore(s => s.directorClipImages)
  const setClipImage = useStore(s => s.directorSetClipImage)
  const imageGenProgress = useStore(s => s.directorImageGenProgress)
  const uploadAndAnalyze = useStore(s => s.directorUploadAndAnalyze)
  const setEnergyBias = useStore(s => s.directorSetEnergyBias)
  const confirmStructure = useStore(s => s.directorConfirmStructure)
  const setSceneDescription = useStore(s => s.directorSetSceneDescription)
  const setReferenceImage = useStore(s => s.directorSetReferenceImage)
  const planPrompts = useStore(s => s.directorPlanPrompts)
  const planVideoPrompts = useStore(s => s.directorPlanVideoPrompts)
  const generateStartImages = useStore(s => s.directorGenerateStartImages)
  const applyToClips = useStore(s => s.directorApplyToClips)
  const directorGenerate = useStore(s => s.directorGenerate)
  const editClipPlan = useStore(s => s.directorEditClipPlan)
  const reset = useStore(s => s.directorReset)
  const speakers = useStore(s => s.directorSpeakers)
  const speakerMappings = useStore(s => s.directorSpeakerMappings)
  const setSpeakerMapping = useStore(s => s.directorSetSpeakerMapping)
  const insertSpeakerMention = useStore(s => s.directorInsertSpeakerMention)
  const autoMode = useStore(s => s.directorAutoMode)
  const skill = useStore(s => s.directorSkill)
  const setSkill = useStore(s => s.setDirectorSkill)
  const musicSource = useStore(s => s.directorMusicSource)
  const setMusicSource = useStore(s => s.setDirectorMusicSource)
  const songDescription = useStore(s => s.directorSongDescription)
  const setSongDescription = useStore(s => s.setDirectorSongDescription)
  const generateTrack = useStore(s => s.directorGenerateTrack)

  // Short film specific
  const shortFilmCharacters = useStore(s => s.shortFilmCharacters)
  const shortFilmSetCharacters = useStore(s => s.shortFilmSetCharacters)
  const shortFilmUploadAndAnalyze = useStore(s => s.shortFilmUploadAndAnalyze)
  const shortFilmSetPacingBias = useStore(s => s.shortFilmSetPacingBias)
  const shortFilmPlanPrompts = useStore(s => s.shortFilmPlanPrompts)
  const shortFilmPlanVideoPrompts = useStore(s => s.shortFilmPlanVideoPrompts)
  const shortFilmPath = useStore(s => s.shortFilmPath)
  const shortFilmSetPath = useStore(s => s.shortFilmSetPath)
  const shortFilmPlanFromStory = useStore(s => s.shortFilmPlanFromStory)
  const shortFilmTargetDuration = useStore(s => s.shortFilmTargetDuration)
  const shortFilmNarrative = useStore(s => s.shortFilmNarrative)
  const shortFilmSetNarrative = useStore(s => s.shortFilmSetNarrative)
  const startDirectorPipeline = useStore(s => s.startDirectorPipeline)
  const pipelineStatus = useStore(s => s.pipelineStatus)
  const pipelinePhase = pipelineStatus?.phase
  const pipelineActive = Boolean(
    pipelineStatus && !['completed', 'failed', 'cancelled'].includes(pipelineStatus.status),
  )
  const directorQueue = useStore(s => s.directorQueue)
  const directorQueueLoading = useStore(s => s.directorQueueLoading)
  const directorQueueEditingEntryId = useStore(s => s.directorQueueEditingEntryId)
  const queueCurrentDirectorPipeline = useStore(s => s.queueCurrentDirectorPipeline)
  const usesShotImages = directorWillGenerateShotImages(
    selectedDirectorShotImageSupport,
    directorShotImageGuidance,
    directorHasVisualReferences,
  )

  const isShortFilm = skill === 'short_film'
  const isStoryPath = isShortFilm && shortFilmPath === 'story'
  const isMusicVideo = !!skill && !isShortFilm
  // Music Video "Generate a track" setup: the bottom chat IS the song
  // description, and Send kicks off the whole write-song → render → video chain.
  const isMvGenerate = isMusicVideo && musicSource === 'generate'
  const mvGenerateSetup = isMvGenerate && step === 'upload'

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [localBias, setLocalBias] = useState<number | null>(null)
  const [showAnalysisDetails, setShowAnalysisDetails] = useState(false)
  const sliderRef = useRef<number | null>(null)
  const [chatInput, setChatInput] = useState('')
  const [draftQueuePending, setDraftQueuePending] = useState(false)
  const [draftQueueConfirmation, setDraftQueueConfirmation] = useState<string | null>(null)

  // Sync chatInput with store's sceneDescription when entering style step
  useEffect(() => {
    if (step === 'style' && sceneDescription && !chatInput) {
      setChatInput(sceneDescription)
    }
  }, [chatInput, sceneDescription, step])

  useEffect(() => {
    if (!draftQueueConfirmation) return
    const timer = window.setTimeout(() => setDraftQueueConfirmation(null), 5000)
    return () => window.clearTimeout(timer)
  }, [draftQueueConfirmation])

  const refImagePreview = useMemo(
    () => referenceImage ? URL.createObjectURL(referenceImage) : null,
    [referenceImage]
  )

  const speakerSamples = useMemo(() => {
    const samples: Record<string, string[]> = {}
    if (analysis?.lyrics) {
      for (const seg of analysis.lyrics) {
        if (seg.speaker && !samples[seg.speaker]) {
          samples[seg.speaker] = []
        }
        if (seg.speaker && samples[seg.speaker].length < 2) {
          samples[seg.speaker].push(seg.text)
        }
      }
    }
    return samples
  }, [analysis?.lyrics])

  const currentIndex = STEP_ORDER.indexOf(step)
  const pastStep = (s: DirectorStep) => currentIndex > STEP_ORDER.indexOf(s)
  const atStep = (s: DirectorStep) => step === s
  const directorPathReady = Boolean(skill && (!isShortFilm || shortFilmPath))
  // Once prompts exist, controls reopen as the settings for the next immutable
  // revision. The active renderer keeps its frozen request, so adjusting a
  // LoRA/model while it runs can never mutate work already in flight.
  const directorSetupLocked = currentIndex >= STEP_ORDER.indexOf('plan')
    && step !== 'review_video'

  const handleFile = useCallback((file: File) => {
    // Accept audio/* MIME OR video/* MIME (backend extracts the audio
    // track from video) OR a matching file extension. Some browsers /
    // OSes don't set MIME on drag-drop, so the extension fallback is
    // load-bearing.
    const mimeOk = file.type.startsWith('audio/') || file.type.startsWith('video/')
    const extOk = AUDIO_ACCEPT.split(',').some(ext => file.name.toLowerCase().endsWith(ext))
    if (!mimeOk && !extOk) {
      return
    }
    if (isShortFilm) {
      shortFilmUploadAndAnalyze(file)
    } else {
      uploadAndAnalyze(file)
    }
  }, [uploadAndAnalyze, shortFilmUploadAndAnalyze, isShortFilm])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const totalClipDuration = useMemo(
    () => plannedClips.length > 0 ? plannedClips[plannedClips.length - 1].end : 0,
    [plannedClips]
  )

  const beatDistribution = useMemo(() => {
    const counts: Record<number, number> = {}
    for (const c of plannedClips) {
      counts[c.beat_count] = (counts[c.beat_count] || 0) + 1
    }
    return Object.entries(counts)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([beats, count]) => `${count}x${beats}-beat`)
      .join(', ')
  }, [plannedClips])

  // Auto-scroll to bottom on step/loading changes. loadingMessage and error
  // are included so progress-text updates (e.g. "Generating music track…",
  // analyze phases) and new errors pull the view down to the newest content.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [step, loading, loadingMessage, error, clipPlans.length, clipImages.length, skill])

  const handleChatSubmit = () => {
    // Music Video "Generate a track": the chat is the song description, and
    // Send runs write-song → render track → analyze → plan → images → video.
    if (mvGenerateSetup) {
      if (songDescription.trim() && !loading) void generateTrack('now')
      return
    }
    if (step === 'style' && chatInput.trim()) {
      setSceneDescription(chatInput.trim())
      if (autoMode) {
        // Auto mode: run entire flow server-side via pipeline
        startDirectorPipeline()
      } else if (isStoryPath) {
        shortFilmPlanFromStory()
      } else if (isShortFilm) {
        shortFilmPlanPrompts()
      } else {
        planPrompts()
      }
    }
  }

  const handleQueueDraft = async () => {
    const description = (mvGenerateSetup ? songDescription : chatInput).trim()
    if (!description || !chatInputEnabled || draftQueuePending || directorQueueLoading) return

    // Keep the store authoritative even if the user clicks Queue immediately
    // after typing. startDirectorPipeline freezes this value along with every
    // selected model, LoRA, reference, and Director option.
    if (!mvGenerateSetup) setSceneDescription(description)
    setDraftQueuePending(true)
    setDraftQueueConfirmation(null)
    const before = useStore.getState()
    const editingEntryId = before.directorQueueEditingEntryId
    const beforeIds = new Set((before.directorQueue?.entries || []).map(entry => entry.id))
    try {
      if (mvGenerateSetup) {
        await generateTrack('queue')
      } else {
        await queueCurrentDirectorPipeline()
      }
      const after = useStore.getState()
      const queue = after.directorQueue
      const savedEntry = editingEntryId
        ? queue?.entries.find(entry => entry.id === editingEntryId)
        : queue?.entries.find(entry => !beforeIds.has(entry.id))
      if (!queue || !savedEntry || after.directorError) return
      const waitingCount = queue.entries.filter(
        entry => ['held', 'queued', 'running'].includes(entry.status),
      ).length
      setDraftQueueConfirmation(
        editingEntryId
          ? 'Queue changes saved. The project remains paused until Start Queue.'
          : `Added to Queue · ${waitingCount} Director ${waitingCount === 1 ? 'project' : 'projects'} waiting. Configure another idea or press Start Queue when ready.`,
      )
    } finally {
      setDraftQueuePending(false)
    }
  }

  // Determine chat input state
  const chatInputEnabled = (step === 'style' || mvGenerateSetup) && !loading
  const chatInputPlaceholder = !skill
    ? 'Choose a skill above...'
    : mvGenerateSetup
    ? 'Describe your music video — subject, vibe, mood, setting…'
    : isShortFilm && !shortFilmPath
    ? 'Choose a path above...'
    : step === 'upload' || step === 'analyze'
    ? isMvGenerate
      ? 'Generating your music video…'
      : isShortFilm ? 'Upload dialogue audio to begin...' : 'Upload audio to begin...'
    : step === 'style'
    ? isStoryPath
      ? 'Describe the story... e.g., Two detectives argue over evidence in a dark office.'
      : isShortFilm
        ? 'Describe the story setting and mood... e.g., A tense interrogation in a dimly lit room.'
        : speakers.length >= 2
          ? 'Describe the scene... e.g., Rap music video in a gym. Neon lights and grunge aesthetic.'
          : 'Describe the scene and characters...'
    : step === 'structure'
    ? isShortFilm ? 'Adjust scene pacing above...' : 'Adjust clip structure above...'
    : 'Reviewing...'

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {/* Header with Start Over */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            {isShortFilm ? <Film size={14} className="text-accent-blue" /> : <Music size={14} className="text-accent-blue" />}
            <span className="text-xs font-medium text-text-primary">Director</span>
            {analysis && !isShortFilm && (
              <span className="text-[10px] text-text-muted">
                {analysis.bpm.toFixed(0)} BPM
              </span>
            )}
            {analysis && isShortFilm && (
              <span className="text-[10px] text-text-muted">
                {formatTime(analysis.duration)}
              </span>
            )}
            {!analysis && isStoryPath && (
              <span className="text-[10px] text-text-muted">
                {shortFilmTargetDuration}s
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => useStore.getState().setDashboardOpen(true)}
              className="text-[10px] text-accent-blue hover:text-accent-blue/80 flex items-center gap-0.5 transition-colors"
              title="Open pipeline dashboard"
            >
              Dashboard
            </button>
            {(skill || step !== 'upload') && (
              <button
                onClick={reset}
                className="text-[10px] text-text-muted hover:text-text-primary flex items-center gap-0.5 transition-colors"
                title="Start over"
              >
                <RotateCcw size={10} /> Start Over
              </button>
            )}
          </div>
        </div>

        {/* Welcome message */}
        <SystemBubble>
          <p className="text-xs text-text-secondary">
            Welcome to Maestro Director. Choose a skill to get started.
          </p>
        </SystemBubble>

        {/* Skill selector */}
        {!skill ? (
          <SkillSelector onSelect={setSkill} />
        ) : (
          <UserBubble>
            <div className="flex items-center gap-1.5 text-xs text-text-primary">
              {isShortFilm ? <Film size={12} className="text-accent-blue" /> : <Music size={12} className="text-accent-blue" />}
              <span>{isShortFilm ? 'Short Film' : 'Music Video'}</span>
            </div>
          </UserBubble>
        )}

        {/* Short Film path chooser */}
        {isShortFilm && skill && !shortFilmPath && (
          <SystemBubble>
            <p className="text-xs text-text-secondary mb-2">How would you like to create your short film?</p>
            <PathChooser onSelect={(path: ShortFilmPath) => {
              shortFilmSetPath(path)
              if (path === 'story') {
                useStore.setState({ directorStep: 'style' })
              }
            }} />
          </SystemBubble>
        )}
        {isShortFilm && shortFilmPath && (
          <UserBubble>
            <div className="flex items-center gap-1.5 text-xs text-text-primary">
              {shortFilmPath === 'story' ? <FileText size={12} className="text-accent-blue" /> : <Upload size={12} className="text-accent-blue" />}
              <span>{shortFilmPath === 'story' ? 'Describe a Story' : 'Upload Audio'}</span>
            </div>
          </UserBubble>
        )}

        {/* Upload step — hidden for story path and before short film path is chosen */}
        {/* Core project choices stay mounted throughout Director. Uploading or
            analyzing media must not make the selected models disappear. */}
        {directorPathReady && (
          <SystemBubble>
            <DirectorSetupPanel locked={directorSetupLocked} />
          </SystemBubble>
        )}

        {pipelineActive && step === 'review_video' && (
          <div className="space-y-1 rounded-lg border border-accent-blue/25 bg-accent-blue/10 px-3 py-2 text-[10px] leading-relaxed text-text-secondary">
            <div>
              The current render is frozen. Changes here apply to a new revision; Generate will add it to the held queue.
            </div>
            {pipelineStatus?.progress?.current_clip && (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-t border-accent-blue/15 pt-1 text-text-primary">
                <span>
                  Clip {pipelineStatus.progress.current_clip}/{pipelineStatus.progress.total_clips || '?'}
                </span>
                {pipelineStatus.progress.clip_eta_seconds != null && (
                  <span>
                    {formatEtaDuration(pipelineStatus.progress.clip_eta_seconds)} remaining
                    {pipelineStatus.progress.clip_completion_at
                      ? ` · around ${formatEstimatedClock(pipelineStatus.progress.clip_completion_at)}`
                      : ''}
                  </span>
                )}
                {pipelineStatus.progress.project_eta_seconds != null && (
                  <span className="text-text-muted">
                    Full render {formatEtaDuration(pipelineStatus.progress.project_eta_seconds)}
                    {pipelineStatus.progress.project_completion_at
                      ? ` · around ${formatEstimatedClock(pipelineStatus.progress.project_completion_at)}`
                      : ''}
                  </span>
                )}
                {pipelineStatus.progress.eta_confidence === 'calibrating' && (
                  <span className="text-text-muted">Calibrating ETA…</span>
                )}
                {(pipelineStatus.progress.eta_history_samples ?? 0) > 0 && (
                  <span className="text-text-muted">
                    Based on {pipelineStatus.progress.eta_history_samples} {pipelineStatus.progress.eta_history_match === 'exact' ? 'matching' : 'related'} local render{pipelineStatus.progress.eta_history_samples === 1 ? '' : 's'}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {skill && (!isShortFilm || shortFilmPath === 'audio') && (atStep('upload') || atStep('analyze') || pastStep('analyze')) && (
          <>
            {!audioFile && !pastStep('analyze') ? (
              <SystemBubble>
                <p className="text-xs text-text-secondary mb-2">
                  {isShortFilm
                    ? directorUsesOmniManifest
                      ? 'Upload dialogue audio, then add ordered H3 Omni identity, motion, scene, voice, or style references.'
                      : 'Upload dialogue audio, then add optional character and location references.'
                    : directorUsesOmniManifest
                      ? 'Upload or generate a track, then add ordered H3 Omni identity, motion, scene, voice, or style references.'
                      : 'Upload or generate a track, then add optional visual references.'}
                </p>
                <div className="space-y-3">
                  {/* Music Video: upload a track OR generate one with the selected music model. */}
                  {!isShortFilm && (
                    <div className="flex gap-1.5 p-1 bg-bg-tertiary rounded-lg border border-border">
                      {(['upload', 'generate'] as const).map(opt => {
                        const active = (musicSource || 'upload') === opt
                        return (
                          <button
                            key={opt}
                            onClick={() => setMusicSource(opt)}
                            className={`flex-1 px-2 py-1.5 rounded-md text-[11px] font-medium transition-all ${
                              active ? 'bg-accent-blue text-white' : 'text-text-secondary hover:text-text-primary'
                            }`}
                          >
                            {opt === 'upload' ? 'Upload a track' : 'Generate a track'}
                          </button>
                        )
                      })}
                    </div>
                  )}
                  {!isShortFilm && musicSource === 'generate' ? (
                    !loading && <DirectorSongSetup />
                  ) : (
                    <UploadZone
                      dragOver={dragOver}
                      setDragOver={setDragOver}
                      handleDrop={handleDrop}
                      handleFile={handleFile}
                      loading={loading && atStep('analyze')}
                      loadingMessage={loadingMessage}
                      audioFile={audioFile}
                      isShortFilm={isShortFilm}
                    />
                  )}
                  <DirectorReferenceInputs
                    referenceImage={referenceImage}
                    refImagePreview={refImagePreview}
                    setReferenceImage={setReferenceImage}
                  />
                  {isShortFilm && referenceImage && (
                    <CharacterNaming
                      characters={shortFilmCharacters}
                      setCharacters={shortFilmSetCharacters}
                    />
                  )}
                  {/* Keep the newest track-generation activity at the bottom
                      of the input group so the chat scroll anchor reveals it. */}
                  {!isShortFilm && musicSource === 'generate' && loading && (
                    <div className="flex items-center gap-2 py-2">
                      <Loader2 size={14} className="animate-spin text-accent-blue" />
                      <span className="text-xs text-text-muted">{loadingMessage || 'Generating…'}</span>
                    </div>
                  )}
                </div>
              </SystemBubble>
            ) : audioFile && (atStep('analyze') || atStep('upload')) ? (
              <SystemBubble>
                <div className="space-y-3">
                  <UploadZone
                    dragOver={dragOver}
                    setDragOver={setDragOver}
                    handleDrop={handleDrop}
                    handleFile={handleFile}
                    loading={loading}
                    loadingMessage={loadingMessage}
                    audioFile={audioFile}
                    isShortFilm={isShortFilm}
                  />
                  {/* Keep the reference selections VISIBLE during analysis —
                      they used to unmount behind a `!loading` gate, which read
                      as "my selections disappeared". Interaction is disabled
                      while loading; the state is untouched. */}
                  <div className={loading ? 'opacity-60 pointer-events-none' : ''}>
                    <DirectorReferenceInputs
                      referenceImage={referenceImage}
                      refImagePreview={refImagePreview}
                      setReferenceImage={setReferenceImage}
                      disabled={loading}
                    />
                  </div>
                </div>
              </SystemBubble>
            ) : audioFile && pastStep('analyze') ? (
              <UserBubble>
                <div className="flex items-center gap-2 text-xs text-text-primary">
                  {isShortFilm ? <Film size={12} className="text-text-muted" /> : <Music size={12} className="text-text-muted" />}
                  <span className="truncate">{audioFile.name}</span>
                  {referenceImage && refImagePreview && (
                    <img src={refImagePreview} alt="Ref" className="w-8 h-8 object-cover rounded border border-border ml-auto" />
                  )}
                </div>
              </UserBubble>
            ) : null}
          </>
        )}

        {/* Analysis result — hidden for story path */}
        {/* Model-specific generation options follow the media/reference inputs
            and remain editable until prompt planning begins. */}
        {directorPathReady && !isStoryPath && !directorSetupLocked && (
          <SystemBubble>
            <DirectorGenerationOptions />
          </SystemBubble>
        )}

        {!isStoryPath && analysis && pastStep('analyze') && (
          <SystemBubble>
            <AnalysisSummary
              analysis={analysis}
              showDetails={showAnalysisDetails}
              setShowDetails={setShowAnalysisDetails}
              speakerMappings={speakerMappings}
              isShortFilm={isShortFilm}
            />
            {/* Allow adding/changing reference photo after analysis */}
            {!pastStep('style') && (
              <div className="mt-2 pt-2 border-t border-border/50">
                <DirectorReferenceInputs
                  referenceImage={referenceImage}
                  refImagePreview={refImagePreview}
                  setReferenceImage={setReferenceImage}
                />
              </div>
            )}
          </SystemBubble>
        )}

        {/* Error */}
        {error && (
          <div className="text-[11px] text-red-400 bg-red-500/10 rounded px-2 py-1.5 border border-red-500/20">
            {error}
          </div>
        )}

        {/* Structure step — hidden for story path */}
        {!isStoryPath && (atStep('structure') || pastStep('structure')) && (
          <>
            <SystemBubble>
              <StructureView
                plannedClips={plannedClips}
                energyBias={energyBias}
                localBias={localBias}
                setLocalBias={setLocalBias}
                sliderRef={sliderRef}
                setEnergyBias={isShortFilm ? shortFilmSetPacingBias : setEnergyBias}
                loading={loading}
                totalClipDuration={totalClipDuration}
                beatDistribution={beatDistribution}
                confirmStructure={confirmStructure}
                isActive={atStep('structure')}
                isShortFilm={isShortFilm}
              />
            </SystemBubble>
            {pastStep('structure') && (
              <UserBubble>
                <div className="flex items-center gap-1.5 text-xs text-text-primary">
                  <Check size={12} className="text-indicator-success" />
                  <span>{plannedClips.length} {isShortFilm ? 'scenes' : 'clips'} confirmed</span>
                  <span className="text-text-muted">({formatTime(totalClipDuration)})</span>
                </div>
              </UserBubble>
            )}
          </>
        )}

        {/* Style step */}
        {(atStep('style') || pastStep('style')) && (
          <>
            {/* Story path: show reference image + characters + duration here (since no upload step) */}
            {isStoryPath && atStep('style') && (
              <SystemBubble>
                <p className="text-xs text-text-secondary mb-2">
                  {directorUsesOmniManifest
                    ? 'Set up your short film with ordered H3 Omni image, video, and audio references, then set the target duration.'
                    : 'Set up your short film. Upload a reference photo, name your characters, and set the target duration.'}
                </p>
                <div className="space-y-3">
                  <DirectorReferenceInputs
                    referenceImage={referenceImage}
                    refImagePreview={refImagePreview}
                    setReferenceImage={setReferenceImage}
                  />
                  {referenceImage && (
                    <CharacterNaming
                      characters={shortFilmCharacters}
                      setCharacters={shortFilmSetCharacters}
                    />
                  )}
                  <DirectorTargetDurationControl />
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={shortFilmNarrative}
                      onChange={e => shortFilmSetNarrative(e.target.checked)}
                      className="accent-accent-blue"
                    />
                    <div>
                      <span className="text-[10px] text-text-primary">Narrative storytelling</span>
                      <p className="text-[9px] text-text-muted leading-tight">
                        Structure scenes around a character arc with rising tension and emotional resolution
                      </p>
                    </div>
                  </label>
                  <div className="pt-1 border-t border-border/50">
                    <DirectorGenerationOptions />
                  </div>
                </div>
              </SystemBubble>
            )}
            {isStoryPath && pastStep('style') && referenceImage && refImagePreview && (
              <UserBubble>
                <div className="flex items-center gap-2 text-xs text-text-primary">
                  <img src={refImagePreview} alt="Ref" className="w-8 h-8 object-cover rounded border border-border" />
                  <span>{shortFilmTargetDuration}s film</span>
                </div>
              </UserBubble>
            )}
            <SystemBubble>
              <StyleForm
                speakers={speakers}
                speakerMappings={speakerMappings}
                speakerSamples={speakerSamples}
                setSpeakerMapping={setSpeakerMapping}
                insertSpeakerMention={insertSpeakerMention}
                isActive={atStep('style')}
                isShortFilm={isShortFilm}
                isStoryPath={isStoryPath}
              />
            </SystemBubble>
            {pastStep('style') && sceneDescription && (
              <UserBubble>
                <p className="text-xs text-text-primary">{sceneDescription}</p>
              </UserBubble>
            )}
          </>
        )}

        {/* Plan loading with LLM thinking stream */}
        {atStep('plan') && loading && (
          <SystemBubble>
            <div className="flex items-center gap-2 py-1">
              <Loader2 size={14} className="animate-spin text-accent-blue" />
              <span className="text-xs text-text-muted">
                {pipelineStatus?.progress?.message
                  || (pipelinePhase === 'polishing_prompts'
                  ? 'Polishing prompts (3rd pass)...'
                  : isStoryPath
                    ? `Planning scenes and writing ${usesShotImages ? 'prompts' : 'video prompts'}...`
                    : isShortFilm
                      ? `Writing ${usesShotImages ? 'scene prompts' : 'video prompts'}...`
                      : `Writing ${usesShotImages ? 'image and video prompts' : 'video prompts'}...`)}
              </span>
            </div>
            {pipelinePhase !== 'polishing_prompts' && <LlmThinkingStream stage="plan" />}
          </SystemBubble>
        )}

        {/* Completed planning streams stay in the chat history */}
        {(pastStep('plan') || (atStep('plan') && !loading)) && (
          <LlmLogStage stage="plan" label={isShortFilm ? 'Scene planning' : usesShotImages ? 'Image and video prompts' : 'Video planning'} />
        )}

        {/* Review step (image prompts) */}
        {usesShotImages && (atStep('review') || pastStep('review')) && (
          <SystemBubble>
            <ImagePromptsReview
              clipPlans={clipPlans}
              plannedClips={plannedClips}
              speakerMappings={speakerMappings}
              editClipPlan={editClipPlan}
              planPrompts={isStoryPath ? shortFilmPlanFromStory : isShortFilm ? shortFilmPlanPrompts : planPrompts}
              planVideoPrompts={isShortFilm ? shortFilmPlanVideoPrompts : planVideoPrompts}
              generateStartImages={generateStartImages}
              loading={loading}
              isActive={atStep('review')}
              isShortFilm={isShortFilm}
            />
          </SystemBubble>
        )}

        {/* Image generation step */}
        {usesShotImages && (atStep('generate_images') || pastStep('generate_images')) && (
          <SystemBubble>
            <ImageGenView
              loading={loading}
              imageGenProgress={imageGenProgress}
              clipImages={clipImages}
              planVideoPrompts={planVideoPrompts}
            />
          </SystemBubble>
        )}

        {/* Plan video loading */}
        {atStep('plan_video') && loading && (
          <SystemBubble>
            <div className="flex items-center gap-2 py-1">
              <Loader2 size={14} className="animate-spin text-accent-blue" />
              <span className="text-xs text-text-muted">Writing video prompts...</span>
            </div>
            <LlmThinkingStream stage="plan_video" />
          </SystemBubble>
        )}

        {/* Completed video-prompt streams stay in the chat history */}
        {(pastStep('plan_video') || (atStep('plan_video') && !loading) || atStep('review_video')) && (
          <LlmLogStage stage="plan_video" label="Video prompts" />
        )}

        {/* Video review step */}
        {atStep('review_video') && (
          <SystemBubble>
            <VideoPromptsReview
              clipPlans={clipPlans}
              plannedClips={plannedClips}
              clipImages={clipImages}
              setClipImage={setClipImage}
              allowSceneImageUploads={!usesShotImages && !autoMode}
              speakerMappings={speakerMappings}
              editClipPlan={editClipPlan}
              planVideoPrompts={isShortFilm ? shortFilmPlanVideoPrompts : planVideoPrompts}
              directorGenerate={directorGenerate}
              queueCurrent={queueCurrentDirectorPipeline}
              applyToClips={applyToClips}
              loading={loading}
              isShortFilm={isShortFilm}
              // Any active render changes Generate into an immutable queued
              // variant instead of trying to mutate or parallelize that run.
              isGenerating={isGenerating || pipelineActive || Boolean(directorQueue?.running)}
              isAutoGenerating={autoMode && pipelineActive}
              editingQueueEntryId={directorQueueEditingEntryId}
            />
          </SystemBubble>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat input bar */}
      <div className="px-4 py-3 border-t border-border space-y-2">
        <div className="flex items-end gap-2">
          {/* Auto-grows with content (issue #11). The composer bar is the
              last child of the panel's flex column, so extra height is
              taken from the messages area above — the box visually
              expands UPWARD from its bottom-anchored position. Rests at
              2 rows (min 56px), caps at 240px (~11 lines), scrolls with
              a visible thumb past that. */}
          <AutoResizeTextarea
            value={mvGenerateSetup ? songDescription : chatInput}
            onChange={e => {
              const v = e.target.value
              setDraftQueueConfirmation(null)
              if (mvGenerateSetup) { setSongDescription(v); return }
              setChatInput(v)
              if (step === 'style') setSceneDescription(v)
            }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey && chatInputEnabled) {
                e.preventDefault()
                handleChatSubmit()
              }
            }}
            placeholder={chatInputPlaceholder}
            disabled={!chatInputEnabled}
            rows={2}
            minHeight={56}
            maxHeight={240}
            className="flex-1 bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted resize-none focus:outline-none focus:border-accent-blue transition-colors disabled:opacity-50 disabled:cursor-not-allowed scrollbar-visible"
          />
          <div className="flex shrink-0 overflow-hidden rounded-lg border border-accent-blue/60">
            <button
              onClick={handleChatSubmit}
              disabled={!chatInputEnabled || draftQueuePending || !(mvGenerateSetup ? songDescription : chatInput).trim()}
              className="p-2 bg-accent-blue text-white hover:bg-accent-blue-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              title={mvGenerateSetup ? 'Generate the song and start this Director project' : 'Start this Director project now'}
              aria-label={mvGenerateSetup ? 'Generate song and start Director project' : 'Start Director project now'}
            >
              {loading && (step === 'style' || isMusicVideo) && !draftQueuePending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
            </button>
            <button
              onClick={() => void handleQueueDraft()}
              disabled={!chatInputEnabled || draftQueuePending || directorQueueLoading || !(mvGenerateSetup ? songDescription : chatInput).trim()}
              className="border-l border-white/20 bg-accent-blue/85 px-2 text-white hover:bg-accent-blue-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              title={mvGenerateSetup
                ? 'Generate the song, then hold the complete Director project in the paused queue'
                : 'Add this complete Director project to the paused queue without starting it'}
              aria-label={directorQueueEditingEntryId ? 'Save Director queue changes' : 'Add Director project to queue'}
            >
              {draftQueuePending || directorQueueLoading
                ? <Loader2 size={14} className="animate-spin" />
                : draftQueueConfirmation
                  ? <Check size={14} />
                  : <ListVideo size={14} />}
            </button>
          </div>
        </div>
        {draftQueueConfirmation && (
          <div
            role="status"
            aria-live="polite"
            className="rounded-md border border-green-500/20 bg-green-500/5 px-2.5 py-2 text-[10px] leading-relaxed text-indicator-success"
          >
            {draftQueueConfirmation}
          </div>
        )}
      </div>
    </div>
  )
}

// --- Sub-components ---

function CharacterNaming({
  characters, setCharacters,
}: {
  characters: ShortFilmCharacter[]
  setCharacters: (characters: ShortFilmCharacter[]) => void
}) {
  const addCharacter = () => {
    setCharacters([...characters, { name: '', description: '' }])
  }

  const updateCharacter = (index: number, field: 'name' | 'description', value: string) => {
    const updated = characters.map((c, i) =>
      i === index ? { ...c, [field]: value } : c
    )
    setCharacters(updated)
  }

  const removeCharacter = (index: number) => {
    setCharacters(characters.filter((_, i) => i !== index))
  }

  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider block mb-1.5">
        <Users size={10} className="inline mr-1" />
        Name the Characters
      </label>
      <div className="space-y-1.5">
        {characters.map((char, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <input
              type="text"
              value={char.name}
              onChange={e => updateCharacter(i, 'name', e.target.value)}
              placeholder={`Character ${i + 1} name`}
              className="flex-1 bg-bg-secondary border border-border rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue transition-colors"
            />
            <input
              type="text"
              value={char.description}
              onChange={e => updateCharacter(i, 'description', e.target.value)}
              placeholder="brief description"
              className="flex-1 bg-bg-secondary border border-border rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue transition-colors"
            />
            <button
              onClick={() => removeCharacter(i)}
              className="p-1 rounded hover:bg-bg-hover transition-colors shrink-0"
            >
              <X size={10} className="text-text-muted" />
            </button>
          </div>
        ))}
      </div>
      <button
        onClick={addCharacter}
        className="mt-1.5 text-[10px] text-accent-blue hover:text-accent-blue-hover transition-colors"
      >
        + Add character
      </button>
      <span className="text-[10px] text-text-muted block mt-1">
        Name the people visible in the reference photo so the AI can identify them.
      </span>
    </div>
  )
}

function DirectorAspectRatioSelector({ disabled = false }: { disabled?: boolean }) {
  const ratio = useStore(s => s.directorAspectRatio)
  const setRatio = useStore(s => s.setDirectorAspectRatio)
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const supportsUltraWide = videoModel.toLowerCase().startsWith('minimax_h3')

  useEffect(() => {
    if (!supportsUltraWide && ratio === '21:9') setRatio('16:9')
  }, [ratio, setRatio, supportsUltraWide])

  const presets = [
    ...(supportsUltraWide
      ? [{ value: '21:9' as const, label: '21:9', desc: 'Cinema' }]
      : []),
    { value: '16:9' as const, label: '16:9', desc: 'Wide' },
    { value: '9:16' as const, label: '9:16', desc: 'Portrait' },
    { value: '1:1' as const, label: '1:1', desc: 'Square' },
    { value: '4:3' as const, label: '4:3', desc: 'Classic' },
    { value: '3:4' as const, label: '3:4', desc: 'Tall' },
  ]
  return (
    <div>
      <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Aspect Ratio</label>
      <div className="flex gap-1.5">
        {presets.map(p => (
          <button
            key={p.value}
            onClick={() => setRatio(p.value)}
            disabled={disabled}
            className={`flex-1 py-1.5 rounded-lg border text-xs transition-all ${
              ratio === p.value
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="font-medium">{p.label}</div>
            <div className="text-[9px] mt-0.5 opacity-60">{p.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}

function DirectorResolutionSelector({ disabled = false }: { disabled?: boolean }) {
  const resolution = useStore(s => s.directorResolution)
  const aspectRatio = useStore(s => s.directorAspectRatio)
  const setResolution = useStore(s => s.setDirectorResolution)
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const [options, setOptions] = useState<ModelOptions | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchModelOptions(videoModel)
      .then(value => { if (!cancelled) setOptions(value) })
      .catch(() => { if (!cancelled) setOptions(null) })
    return () => { cancelled = true }
  }, [videoModel])

  const fallbackOrder = ['480p', '540p', '720p', '1080p'] as const
  const modelPresetOrder = (options?.resolution_preset_order || []).filter(
    value => value !== 'auto',
  )
  const presetOrder = modelPresetOrder.length > 0
    ? modelPresetOrder
    : [...fallbackOrder]
  const presets = presetOrder.map(value => ({
    value,
    label: options?.resolution_presets?.[value]?.label || value,
  }))
  const resolvedResolution = resolveResolution(options, resolution, aspectRatio)
  const recommendation = recommendedWindowProfile(
    options?.director_memory_policy || options?.sliding_window_memory_policy,
    resolvedResolution,
    totalVramGb,
  )
  const fps = options?.fps || 24
  const selectedConfig = options?.resolution_presets?.[resolution]
  return (
    <div>
      <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Resolution</label>
      <div className="flex gap-1.5">
        {presets.map(p => (
          <button
            key={p.value}
            onClick={() => setResolution(p.value)}
            disabled={disabled}
            className={`flex-1 py-2 rounded-lg border text-xs font-medium transition-all ${
              resolution === p.value
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="mt-1 text-[10px] text-text-muted">
        {resolvedResolution}
        {recommendation?.frames != null && totalVramGb > 0 && (
          <> &middot; Auto max shot {formatSeconds(recommendation.frames / fps)} on {totalVramGb.toFixed(0)} GB</>
        )}
      </div>
      {recommendation?.supported === false && (
        <div className="mt-1 text-[10px] text-amber-400">
          Auto recommends {recommendation.fallbackResolution || 'a lower resolution'} on this GPU. An Advanced manual shot-length override is experimental.
        </div>
      )}
      {selectedConfig?.hint && (
        <div className={`mt-1 text-[10px] ${selectedConfig.experimental ? 'text-amber-400' : 'text-text-muted'}`}>
          {selectedConfig.hint}
        </div>
      )}
    </div>
  )
}

function DirectorSetupPanel({ locked }: { locked: boolean }) {
  const autoMode = useStore(s => s.directorAutoMode)
  const setAutoMode = useStore(s => s.setDirectorAutoMode)
  const seamless = useStore(s => s.directorSeamless)
  const setSeamless = useStore(s => s.setDirectorSeamless)
  const selectedVideoSupportsSeamless = useStore(s => {
    const selected = s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const model = s.models.find(item => item.model_type === selected)
    return model?.director
      ? model.director.video.seamless.compatible === true
      : true
  })

  return (
    <div className="space-y-3">
      <DirectorAspectRatioSelector disabled={locked} />
      <DirectorResolutionSelector disabled={locked} />

      <div className="pt-2 border-t border-border/50 space-y-1.5">
        <span className="text-[10px] text-text-muted uppercase tracking-wider block">Workflow</span>
        <div className="flex items-center gap-4">
          <label
            className={`flex items-center gap-1.5 select-none ${
              locked || !selectedVideoSupportsSeamless
                ? 'cursor-not-allowed opacity-50'
                : 'cursor-pointer'
            }`}
            title={selectedVideoSupportsSeamless
              ? 'Render one continuous sliding-window timeline, carrying motion and audio between windows'
              : 'The selected video model cannot carry a continuous timeline between native windows'}
          >
            <input
              type="checkbox"
              checked={seamless}
              disabled={locked || !selectedVideoSupportsSeamless}
              onChange={e => setSeamless(e.target.checked)}
              className="accent-accent-blue w-3 h-3"
            />
            <span className="text-[10px] text-text-secondary">Seamless</span>
          </label>
          <label
            className={`flex items-center gap-1.5 select-none ${locked ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
            title="Skip all review steps and generate automatically"
          >
            <input
              type="checkbox"
              checked={autoMode}
              disabled={locked}
              onChange={e => setAutoMode(e.target.checked)}
              className="accent-red-500 w-3 h-3"
            />
            <span className={`text-[10px] ${autoMode ? 'text-red-400' : 'text-text-secondary'}`}>Auto</span>
          </label>
        </div>
      </div>

      <div className="pt-2 border-t border-border/50 space-y-1.5">
        <span className="text-[10px] text-text-muted uppercase tracking-wider block">Models</span>
        <DirectorModelSelection disabled={locked} />
      </div>

      {locked && (
        <p className="text-[9px] text-text-muted">
          Project setup is locked after planning begins.
        </p>
      )}
    </div>
  )
}

function SkillSelector({ onSelect }: { onSelect: (skill: DirectorSkill) => void }) {
  const skills = [
    { id: 'music_video' as DirectorSkill, label: 'Music Video', desc: 'Automated music video from audio', icon: Music, active: true },
    { id: 'short_film' as DirectorSkill, label: 'Short Film', desc: 'Dialogue-driven scenes from audio', icon: Film, active: true },
    { id: 'music_video' as DirectorSkill, label: 'Video Podcast', desc: 'Coming Soon', icon: Mic, active: false },
    { id: 'music_video' as DirectorSkill, label: 'Viral Video', desc: 'Coming Soon', icon: Sparkles, active: false },
  ]

  return (
    <div className="grid grid-cols-2 gap-2">
      {skills.map((s) => (
        <button
          key={s.label}
          onClick={() => s.active && onSelect(s.id)}
          disabled={!s.active}
          className={`relative p-3 rounded-lg border text-left transition-all ${
            s.active
              ? 'border-accent-blue/30 bg-bg-tertiary/50 hover:border-accent-blue hover:bg-accent-blue/5 cursor-pointer'
              : 'border-border/30 bg-bg-tertiary/20 opacity-50 cursor-not-allowed'
          }`}
        >
          <s.icon size={16} className={s.active ? 'text-accent-blue mb-1.5' : 'text-text-muted mb-1.5'} />
          <div className="text-xs font-medium text-text-primary">{s.label}</div>
          <div className="text-[10px] text-text-muted mt-0.5">{s.desc}</div>
          {!s.active && (
            <span className="absolute top-1.5 right-1.5 text-[8px] bg-bg-hover text-text-muted px-1.5 py-0.5 rounded-full">
              Soon
            </span>
          )}
        </button>
      ))}
    </div>
  )
}

function PathChooser({ onSelect }: { onSelect: (path: ShortFilmPath) => void }) {
  const paths = [
    { id: 'audio' as ShortFilmPath, label: 'Upload Audio', desc: 'Upload recorded dialogue', icon: Upload },
    { id: 'story' as ShortFilmPath, label: 'Describe a Story', desc: 'AI writes the script', icon: FileText },
  ]
  return (
    <div className="grid grid-cols-2 gap-2">
      {paths.map((p) => (
        <button
          key={p.id}
          onClick={() => onSelect(p.id)}
          className="p-3 rounded-lg border border-accent-blue/30 bg-bg-tertiary/50 hover:border-accent-blue hover:bg-accent-blue/5 cursor-pointer text-left transition-all"
        >
          <p.icon size={16} className="text-accent-blue mb-1.5" />
          <div className="text-xs font-medium text-text-primary">{p.label}</div>
          <div className="text-[10px] text-text-muted mt-0.5">{p.desc}</div>
        </button>
      ))}
    </div>
  )
}

function UploadZone({
  dragOver, setDragOver, handleDrop, handleFile, loading, loadingMessage, audioFile, isShortFilm,
}: {
  dragOver: boolean
  setDragOver: (v: boolean) => void
  handleDrop: (e: React.DragEvent) => void
  handleFile: (file: File) => void
  loading: boolean
  /** Sub-status string from the analyze polling loop. Falls back
   *  to the default ("Analyzing audio..." / "Transcribing dialogue...")
   *  when null. */
  loadingMessage: string | null
  audioFile: File | null
  isShortFilm?: boolean
}) {
  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
        dragOver ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:border-border-light'
      }`}
    >
      {loading ? (
        <div className="flex flex-col items-center gap-2 py-2">
          <Loader2 size={20} className="animate-spin text-accent-blue" />
          {/* Sub-status (set by directorUploadAndAnalyze polling loop) takes
              precedence over the static fallback. Reflects backend phase:
              "Loading transcription model (first use downloads ~300MB)..." etc. */}
          <span className="text-[11px] text-text-muted text-center px-2">
            {loadingMessage || (isShortFilm ? 'Transcribing dialogue...' : 'Analyzing audio...')}
          </span>
        </div>
      ) : audioFile ? (
        <div className="flex flex-col items-center gap-1">
          <Music size={16} className="text-text-muted" />
          <span className="text-xs text-text-secondary truncate max-w-full">{audioFile.name}</span>
        </div>
      ) : (
        <label className="cursor-pointer flex flex-col items-center gap-1.5">
          <Music size={20} className="text-accent-blue/60" />
          <span className="text-xs text-text-secondary">{isShortFilm ? 'Drop dialogue audio or click to upload' : 'Drop a song or video or click to upload'}</span>
          <span className="text-[10px] text-text-muted">audio: wav/mp3/flac/ogg/m4a · video: mp4/mov/mkv/webm/avi (audio extracted)</span>
          <input
            type="file"
            accept={AUDIO_ACCEPT}
            className="hidden"
            onChange={e => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
            }}
          />
        </label>
      )}
    </div>
  )
}

function ReferenceImageUpload({
  referenceImage, refImagePreview, setReferenceImage,
}: {
  referenceImage: File | null
  refImagePreview: string | null
  setReferenceImage: (file: File | null) => void
}) {
  const fixedMediaStrength = useStore(s => {
    const selected = s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const model = s.models.find(item => item.model_type === selected)
    return directorModelUsesFixedMediaStrength(selected, model?.architecture)
  })
  const strengthLabel = useStore(s => s.modelOptions?.input_video_strength_label ?? '')
  const inputVideoStrength = useStore(s => s.params.input_video_strength ?? 1.0)
  const setParam = useStore(s => s.setParam)
  const [dragOver, setDragOver] = useState(false)

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) setReferenceImage(file)
  }, [setReferenceImage])

  return (
    <div className="space-y-2">
      {referenceImage && refImagePreview ? (
        <div className="relative">
          <label className="cursor-pointer block">
            <img
              src={refImagePreview}
              alt="Reference"
              className="w-full h-24 object-cover rounded-lg border border-border hover:border-accent-blue transition-colors"
              title="Click to change photo"
            />
            <input
              type="file"
              accept={IMAGE_ACCEPT}
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) setReferenceImage(f) }}
            />
          </label>
          <button
            onClick={() => setReferenceImage(null)}
            className="absolute top-1.5 right-1.5 bg-bg-primary/80 rounded-full p-1 hover:bg-bg-hover transition-colors"
            title="Remove"
          >
            <X size={12} className="text-text-muted" />
          </button>
          <span className="absolute bottom-1.5 left-1.5 text-[9px] text-white/80 bg-black/50 px-1.5 py-0.5 rounded">
            Reference photo &middot; click to change
          </span>
        </div>
      ) : (
        <label
          className={`cursor-pointer block border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
            dragOver ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:border-border-light'
          }`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <div className="flex flex-col items-center gap-1.5">
            <ImageIcon size={20} className="text-accent-blue/60" />
            <span className="text-xs text-text-secondary">Drop reference photo or click to upload</span>
            <span className="text-[10px] text-text-muted">Creates start images for each clip</span>
          </div>
          <input
            type="file"
            accept={IMAGE_ACCEPT}
            className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) setReferenceImage(f) }}
          />
        </label>
      )}
      {referenceImage && !fixedMediaStrength && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-[11px] text-text-secondary">{strengthLabel || 'Image Strength'}</label>
            <span className="text-[11px] text-text-muted tabular-nums">{inputVideoStrength.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={1} step={0.01} value={inputVideoStrength}
            onChange={e => setParam('input_video_strength', parseFloat(e.target.value))}
            className="w-full h-1 accent-accent-blue" />
          <p className="text-[9px] text-text-muted">Lower values can increase motion</p>
        </div>
      )}
    </div>
  )
}

function DirectorReferenceInputs({
  referenceImage,
  refImagePreview,
  setReferenceImage,
  disabled = false,
}: {
  referenceImage: File | null
  refImagePreview: string | null
  setReferenceImage: (file: File | null) => void
  disabled?: boolean
}) {
  const usesOmniManifest = useStore(s => {
    const selected = s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    return selected.toLowerCase().startsWith('minimax_h3_ref2va')
      || s.models.find(model => model.model_type === selected)
        ?.director?.video_strategy === 'omni_reference'
  })

  if (usesOmniManifest) {
    return <OmniReferenceSection scope="director" disabled={disabled} />
  }
  return (
    <>
      <ReferenceImageUpload
        referenceImage={referenceImage}
        refImagePreview={refImagePreview}
        setReferenceImage={setReferenceImage}
      />
      <AdditionalRefsSection />
    </>
  )
}

function DraggableRefRow({ file, label, index, onRemove, onLabelChange, onReorder, placeholder }: {
  file: File; label: string; index: number
  onRemove: (i: number) => void
  onLabelChange: (i: number, v: string) => void
  onReorder: (from: number, to: number) => void
  placeholder: string
}) {
  const [dragOver, setDragOver] = useState(false)

  return (
    <div
      draggable
      onDragStart={e => { e.dataTransfer.setData('text/plain', String(index)); e.dataTransfer.effectAllowed = 'move' }}
      onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => {
        e.preventDefault(); setDragOver(false)
        const from = parseInt(e.dataTransfer.getData('text/plain'), 10)
        if (!isNaN(from) && from !== index) onReorder(from, index)
      }}
      className={`flex items-center gap-1.5 group cursor-grab active:cursor-grabbing transition-colors rounded ${
        dragOver ? 'bg-accent-blue/10 border border-accent-blue/30' : ''
      }`}
    >
      <div className="relative flex-shrink-0">
        <img src={URL.createObjectURL(file)} alt={`Ref ${index+1}`}
          className="w-[60px] h-[60px] object-cover rounded border border-border pointer-events-none" />
        <button onClick={() => onRemove(index)}
          className="absolute -top-1 -right-1 bg-red-500 rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity z-10">
          <X size={8} className="text-white" />
        </button>
        <span className="absolute bottom-0 left-0 bg-black/60 text-white text-[7px] px-1 rounded-br rounded-tl pointer-events-none">
          {index + 1}
        </span>
      </div>
      <input
        type="text"
        value={label}
        onChange={e => onLabelChange(index, e.target.value)}
        placeholder={placeholder}
        className="flex-1 min-w-0 bg-bg-secondary border border-border rounded px-1.5 py-0.5 text-[10px] text-text-primary placeholder:text-text-muted focus:border-accent-blue outline-none"
      />
    </div>
  )
}

function AdditionalRefsSection() {
  const charRefs = useStore(s => s.directorCharacterRefs)
  const charLabels = useStore(s => s.directorCharacterRefLabels)
  const locRefs = useStore(s => s.directorLocationRefs)
  const locLabels = useStore(s => s.directorLocationRefLabels)
  const addCharRef = useStore(s => s.directorAddCharacterRef)
  const removeCharRef = useStore(s => s.directorRemoveCharacterRef)
  const setCharLabel = useStore(s => s.directorSetCharacterRefLabel)
  const reorderCharRefs = useStore(s => s.directorReorderCharacterRefs)
  const addLocRef = useStore(s => s.directorAddLocationRef)
  const removeLocRef = useStore(s => s.directorRemoveLocationRef)
  const setLocLabel = useStore(s => s.directorSetLocationRefLabel)
  const reorderLocRefs = useStore(s => s.directorReorderLocationRefs)
  const voiceRef = useStore(s => s.directorVoiceRef)
  const setVoiceRef = useStore(s => s.setDirectorVoiceRef)
  const identityScale = useStore(s => s.directorIdentityGuidanceScale)
  const setIdentityScale = useStore(s => s.setDirectorIdentityGuidanceScale)
  const voiceReferenceEnabled = useStore(s => s.servicesConfig?.voice_reference_enabled ?? false)
  const selectedVideoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const supportsVoiceReference = useStore(s => (
    s.models.find(model => model.model_type === selectedVideoModel)
      ?.director?.supports_voice_reference ?? false
  ))
  const voiceReferenceMode = useStore(s => (
    s.models.find(model => model.model_type === selectedVideoModel)
      ?.director?.voice_reference_mode ?? 'none'
  ))
  const [expanded, setExpanded] = useState(charRefs.length > 0 || locRefs.length > 0 || voiceRef !== null)

  const handleFiles = useCallback((files: FileList | null, type: 'char' | 'loc') => {
    if (!files) return
    const add = type === 'char' ? addCharRef : addLocRef
    Array.from(files).forEach(f => { if (f.type.startsWith('image/')) add(f) })
  }, [addCharRef, addLocRef])

  const nativeVoiceReference = voiceReferenceMode === 'native_reference'
  const showVoiceReference = supportsVoiceReference
    && (nativeVoiceReference || voiceReferenceEnabled)
  const totalRefs = charRefs.length + locRefs.length + (showVoiceReference && voiceRef ? 1 : 0)

  return (
    <div className="mt-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors w-full"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <Users size={10} />
        <span>Additional references</span>
        {totalRefs > 0 && <span className="ml-auto bg-accent-blue/20 text-accent-blue px-1.5 rounded-full text-[9px]">{totalRefs}</span>}
      </button>
      {expanded && (
        <div className="mt-1.5 space-y-2 pl-1">
          {/* Character References */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-text-secondary">Character refs</span>
              <label className="cursor-pointer text-[9px] text-accent-blue hover:underline">
                + Add
                <input type="file" accept={IMAGE_ACCEPT} multiple className="hidden"
                  onChange={e => handleFiles(e.target.files, 'char')} />
              </label>
            </div>
            {charRefs.length > 0 && (
              <div className="space-y-1">
                {charRefs.map((f, i) => (
                  <DraggableRefRow key={`c${i}-${f.name}`} file={f} label={charLabels[i] || ''} index={i}
                    onRemove={removeCharRef} onLabelChange={setCharLabel} onReorder={reorderCharRefs}
                    placeholder="e.g. Thor - blonde, hammer" />
                ))}
              </div>
            )}
            {charRefs.length === 0 && (
              <p className="text-[9px] text-text-muted italic">Individual character close-ups improve identity</p>
            )}
          </div>
          {/* Location References */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-text-secondary">Location refs</span>
              <label className="cursor-pointer text-[9px] text-accent-blue hover:underline">
                + Add
                <input type="file" accept={IMAGE_ACCEPT} multiple className="hidden"
                  onChange={e => handleFiles(e.target.files, 'loc')} />
              </label>
            </div>
            {locRefs.length > 0 && (
              <div className="space-y-1">
                {locRefs.map((f, i) => (
                  <DraggableRefRow key={`l${i}-${f.name}`} file={f} label={locLabels[i] || ''} index={i}
                    onRemove={removeLocRef} onLabelChange={setLocLabel} onReorder={reorderLocRefs}
                    placeholder="e.g. backstage, leather couches" />
                ))}
              </div>
            )}
            {locRefs.length === 0 && (
              <p className="text-[9px] text-text-muted italic">Scene/environment reference images</p>
            )}
          </div>
          {/* LTX uses an ID-LoRA; H3 Omni maps the sample as a native voice
              reference in each shot's Ref2VA manifest. */}
          {showVoiceReference && <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-text-secondary"><Mic size={9} className="inline mr-0.5" />Voice ref</span>
              {!voiceRef ? (
                <label className="cursor-pointer text-[9px] text-accent-blue hover:underline">
                  + Add
                  <input type="file" accept={AUDIO_ACCEPT} className="hidden"
                    onChange={e => { const f = e.target.files?.[0]; if (f) setVoiceRef(f); e.target.value = '' }} />
                </label>
              ) : (
                <button onClick={() => setVoiceRef(null)} className="text-[9px] text-red-400 hover:text-red-300">Remove</button>
              )}
            </div>
            {voiceRef ? (
              <div className="space-y-1">
                <div className="flex items-center gap-1.5 bg-bg-tertiary rounded px-1.5 py-1">
                  <Mic size={10} className="text-accent-blue shrink-0" />
                  <span className="text-[9px] text-text-secondary truncate">{voiceRef.name}</span>
                </div>
                {!nativeVoiceReference && <div className="flex items-center gap-1.5">
                  <span className="text-[9px] text-text-muted whitespace-nowrap">Identity scale</span>
                  <input type="range" min={0} max={10} step={0.5} value={identityScale}
                    onChange={e => setIdentityScale(parseFloat(e.target.value))}
                    className="flex-1 h-1 accent-accent-blue" />
                  <span className="text-[9px] text-text-muted w-5 text-right">{identityScale}</span>
                </div>}
              </div>
            ) : (
              <p className="text-[9px] text-text-muted italic">
                {nativeVoiceReference
                  ? 'Voice sample used by H3 Omni for the primary speaking character'
                  : '~5 sec voice sample for consistent voice across clips'}
              </p>
            )}
          </div>}
        </div>
      )}
    </div>
  )
}

function AnalysisSummary({
  analysis, showDetails, setShowDetails, isShortFilm,
}: {
  analysis: NonNullable<ReturnType<typeof useStore.getState>['directorAnalysis']>
  showDetails: boolean
  setShowDetails: (v: boolean | ((p: boolean) => boolean)) => void
  speakerMappings: ReturnType<typeof useStore.getState>['directorSpeakerMappings']
  isShortFilm?: boolean
}) {
  // Count unique speakers
  const speakerCount = new Set(
    (analysis.lyrics || []).map(l => l.speaker).filter(Boolean)
  ).size

  return (
    <div className="space-y-1">
      <p className="text-xs text-text-secondary mb-1">
        {isShortFilm ? 'Transcription complete' : 'Analysis complete'}
      </p>
      <button
        onClick={() => setShowDetails(v => !v)}
        className="flex items-center gap-3 text-[11px] text-text-muted w-full hover:text-text-secondary transition-colors"
      >
        <ChevronDown size={10} className={`transition-transform ${showDetails ? '' : '-rotate-90'}`} />
        <span>{formatTime(analysis.duration)}</span>
        {!isShortFilm && <span>{analysis.bpm.toFixed(0)} BPM</span>}
        {isShortFilm && speakerCount > 0 && <span>{speakerCount} speaker{speakerCount > 1 ? 's' : ''}</span>}
        {!isShortFilm && <span>{analysis.sections.length} sections</span>}
        {analysis.lyrics && <span>{analysis.lyrics.length} {isShortFilm ? 'dialogue lines' : 'lyric segments'}</span>}
      </button>

      {showDetails && (
        // No inner scroll — chat panel handles scrolling.
        <div className="bg-bg-tertiary rounded-lg p-2 space-y-2 text-[10px]">
          <div>
            <div className="text-text-muted uppercase tracking-wider mb-1 font-medium">Sections</div>
            <div className="space-y-0.5">
              {analysis.sections.map((sec, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-text-muted w-16 shrink-0">
                    {formatTime(sec.start)}-{formatTime(sec.end)}
                  </span>
                  <SectionBadge label={sec.label} />
                  <EnergyDot energy={sec.energy} />
                  <span className="text-text-muted">{(sec.energy * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>

          {analysis.lyrics && analysis.lyrics.length > 0 && (
            <div>
              <div className="text-text-muted uppercase tracking-wider mb-1 font-medium">
                Lyrics {analysis.song_structure?.length ? '(LLM Structure)' : '(Whisper)'}
              </div>
              <div className="space-y-0.5">
                {analysis.song_structure && analysis.song_structure.length > 0 ? (
                  analysis.song_structure.map((section, si) => {
                    const nextStart = si < analysis.song_structure!.length - 1
                      ? analysis.song_structure![si + 1].start
                      : Infinity
                    const sectionLyrics = analysis.lyrics!.filter(
                      seg => seg.start >= section.start && seg.start < nextStart
                    )
                    return (
                      <div key={si} className="mb-1.5">
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <SectionBadge label={section.label} />
                          <span className="text-text-muted">{formatTime(section.start)}</span>
                          <span className="text-text-secondary font-medium">[{section.display_label}]</span>
                        </div>
                        {sectionLyrics.map((seg, li) => (
                          <div key={li} className="flex gap-2 pl-2">
                            <span className="text-text-muted w-14 shrink-0 text-right">
                              {formatTime(seg.start)}
                            </span>
                            <span className="text-text-secondary">
                              {seg.speaker && (
                                <span className="text-accent-blue text-[9px] mr-1">[{seg.speaker}]</span>
                              )}
                              {seg.text}
                            </span>
                          </div>
                        ))}
                        {sectionLyrics.length === 0 && (
                          <div className="pl-2 text-text-muted italic">(instrumental)</div>
                        )}
                      </div>
                    )
                  })
                ) : (
                  analysis.lyrics.map((seg, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-text-muted w-16 shrink-0">
                        {formatTime(seg.start)}-{formatTime(seg.end)}
                      </span>
                      <span className="text-text-secondary">
                        {seg.speaker && (
                          <span className="text-accent-blue text-[9px] mr-1">[{seg.speaker}]</span>
                        )}
                        {seg.text}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StructureView({
  plannedClips, energyBias, localBias, setLocalBias, sliderRef, setEnergyBias,
  loading, totalClipDuration, beatDistribution, confirmStructure, isActive, isShortFilm,
}: {
  plannedClips: ReturnType<typeof useStore.getState>['directorPlannedClips']
  energyBias: number
  localBias: number | null
  setLocalBias: (v: number | null) => void
  sliderRef: React.MutableRefObject<number | null>
  setEnergyBias: (bias: number) => Promise<void>
  loading: boolean
  totalClipDuration: number
  beatDistribution: string
  confirmStructure: () => void
  isActive: boolean
  isShortFilm?: boolean
}) {
  return (
    <div className="space-y-3">
      <p className="text-xs text-text-secondary">
        {isShortFilm
          ? 'Here are the scenes based on dialogue pacing. Adjust the scene pacing if needed.'
          : 'Here\'s the clip structure based on the audio analysis. Adjust the cut speed if needed.'}
      </p>

      {isActive && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">{isShortFilm ? 'Scene Pacing' : 'Cut Speed'}</label>
            <span className="text-xs text-text-secondary">
              {(localBias ?? energyBias) > 0 ? '+' : ''}{localBias ?? energyBias}
            </span>
          </div>
          <input
            type="range"
            min={-2}
            max={2}
            step={1}
            value={localBias ?? energyBias}
            onChange={e => {
              const v = Number(e.target.value)
              setLocalBias(v)
              sliderRef.current = v
            }}
            onMouseUp={() => {
              if (sliderRef.current !== null && sliderRef.current !== energyBias) {
                setEnergyBias(sliderRef.current)
              }
              setLocalBias(null)
              sliderRef.current = null
            }}
            onTouchEnd={() => {
              if (sliderRef.current !== null && sliderRef.current !== energyBias) {
                setEnergyBias(sliderRef.current)
              }
              setLocalBias(null)
              sliderRef.current = null
            }}
            className="w-full"
          />
          <div className="flex items-center justify-between mt-1 text-[10px] text-text-muted">
            <span>{isShortFilm ? 'Longer scenes' : 'Slower cuts'}</span>
            <span>{isShortFilm ? 'Shorter scenes' : 'Faster cuts'}</span>
          </div>
        </div>
      )}

      <div className="bg-bg-tertiary rounded-lg p-2 space-y-2">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-text-secondary font-medium">{plannedClips.length} {isShortFilm ? 'scenes' : 'clips'}</span>
          <span className="text-text-muted">{formatTime(totalClipDuration)} total</span>
        </div>

        {loading ? (
          <div className="flex items-center gap-1.5 text-[10px] text-text-muted py-1">
            <Loader2 size={10} className="animate-spin" /> Recalculating...
          </div>
        ) : (
          <>
            <div className="flex gap-px h-8 rounded overflow-hidden">
              {plannedClips.map((clip, i) => {
                const clipDur = clip.end - clip.start
                const totalDur = plannedClips.reduce((s, c) => s + (c.end - c.start), 0)
                const widthPct = isShortFilm
                  ? Math.max((clipDur / totalDur) * 100, 1.5)
                  : Math.max((clip.beat_count / plannedClips.reduce((s, c) => s + c.beat_count, 0)) * 100, 1.5)
                const barColor = sectionBarColors[clip.section_label] || 'bg-gray-500'
                const tooltipLabel = isShortFilm
                  ? `Scene ${i + 1}: ${clip.section_label} (${clipDur.toFixed(1)}s)`
                  : `Clip ${i + 1}: ${clip.section_label}, ${clip.beat_count} beats (${clipDur.toFixed(1)}s)`
                return (
                  <div
                    key={i}
                    className={`${barColor} opacity-70 hover:opacity-100 transition-opacity relative group cursor-default`}
                    style={{ width: `${widthPct}%` }}
                    title={tooltipLabel}
                  >
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 pointer-events-none">
                      <div className="bg-bg-primary border border-border rounded px-1.5 py-1 text-[9px] text-text-secondary whitespace-nowrap shadow-lg">
                        {tooltipLabel}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="text-[9px] text-text-muted space-y-1">
              {!isShortFilm && <div>{beatDistribution}</div>}
              <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                {Object.entries(sectionBarColors).map(([label, color]) => {
                  const count = plannedClips.filter(c => c.section_label === label).length
                  if (count === 0) return null
                  return (
                    <div key={label} className="flex items-center gap-1">
                      <span className={`w-2 h-2 rounded-sm ${color}`} />
                      <span>{label} ({count})</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </>
        )}
      </div>

      {isActive && (
        <button
          onClick={confirmStructure}
          disabled={loading || plannedClips.length === 0}
          className="w-full py-2 rounded-lg bg-accent-blue text-white text-xs font-medium hover:bg-accent-blue-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
        >
          <ChevronRight size={12} /> Continue
        </button>
      )}
    </div>
  )
}

/**
 * DirectorAdvancedAccordion — collapsed-by-default panel exposing Director's
 * model-specific generation controls and post-processing knobs. It sits in
 * the chat sidebar alongside the LoRA accordion so per-shoot tweaks are
 * co-located with the rest of the per-shoot setup.
 *
 * Defaults are intentionally "off" for all controls so a user who
 * never opens this accordion gets clean unprocessed output. Each
 * control has a one-line description making clear what it does and
 * what it costs (e.g. "may introduce artifacts" for the refiner)
 * rather than implying a quality hierarchy.
 *
 * No "Quality" preset bundling — the three controls are independent
 * with distinct purposes (resolution change vs aesthetic vs
 * experimental). See the design discussion captured in commit notes.
 */
function DirectorAdvancedAccordion() {
  const [open, setOpen] = useState(false)
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const videoStepsByModel = useStore(s => s.directorVideoInferenceStepsByModel)
  const setVideoSteps = useStore(s => s.setDirectorVideoInferenceSteps)
  const maxShotFramesByModel = useStore(s => s.directorVideoMaxShotFramesByModel)
  const setMaxShotFrames = useStore(s => s.setDirectorVideoMaxShotFrames)
  const turboModeByModel = useStore(s => s.directorH3TurboModeByModel)
  const turboPresetByModel = useStore(s => s.directorH3TurboPresetByModel)
  const savedVideoLoras = useStore(s => s.savedLoraPerMode.video)
  const directorResolution = useStore(s => s.directorResolution)
  const directorAspectRatio = useStore(s => s.directorAspectRatio)
  const totalVramGb = useStore(s => s.systemStats?.gpu.vram_total_gb ?? 0)
  const [directorVideoOptions, setDirectorVideoOptions] = useState<ModelOptions | null>(null)
  const shotImageSupport = useStore(s => s.models.find(
    model => model.model_type === videoModel,
  )?.director?.shot_image_support)
  const shotImageGuidance = useStore(s => s.directorShotImageGuidance)
  const setShotImageGuidance = useStore(s => s.setDirectorShotImageGuidance)
  const hasVisualReferences = useStore(s => Boolean(
    s.directorReferenceImage
    || s.directorReferenceImagePath
    || s.directorCharacterRefs.length
    || s.directorCharacterRefPaths.length
    || s.directorLocationRefs.length
    || s.directorLocationRefPaths.length
    || (
      s.models.find(model => model.model_type === videoModel)
        ?.director?.video_strategy === 'omni_reference'
      && s.directorH3References.some(
        reference => reference.type === 'image' || reference.type === 'video',
      )
    )
  ))
  const generateShotImages = directorWillGenerateShotImages(
    shotImageSupport,
    shotImageGuidance,
    hasVisualReferences,
  )

  // Image post-processing
  const imgUpsampling = useStore(s => s.directorImageSpatialUpsampling)
  const setImgUpsampling = useStore(s => s.setDirectorImageSpatialUpsampling)
  const imgGrain = useStore(s => s.directorImageFilmGrainIntensity)
  const setImgGrain = useStore(s => s.setDirectorImageFilmGrainIntensity)
  const imgGrainSat = useStore(s => s.directorImageFilmGrainSaturation)
  const setImgGrainSat = useStore(s => s.setDirectorImageFilmGrainSaturation)

  // Video post-processing
  const vidUpsampling = useStore(s => s.directorVideoSpatialUpsampling)
  const setVidUpsampling = useStore(s => s.setDirectorVideoSpatialUpsampling)
  const vidGrain = useStore(s => s.directorVideoFilmGrainIntensity)
  const setVidGrain = useStore(s => s.setDirectorVideoFilmGrainIntensity)
  const vidGrainSat = useStore(s => s.directorVideoFilmGrainSaturation)
  const setVidGrainSat = useStore(s => s.setDirectorVideoFilmGrainSaturation)
  const vidSelfRefiner = useStore(s => s.directorVideoSelfRefiner)
  const setVidSelfRefiner = useStore(s => s.setDirectorVideoSelfRefiner)

  // Fetch the selected Director model's sampling contract without routing it
  // through Studio's global modelOptions/params state. That separation is the
  // point of this control: changing steps in either surface must not silently
  // modify the other one.
  useEffect(() => {
    let cancelled = false
    fetchModelOptions(videoModel)
      .then(options => {
        if (cancelled) return
        setDirectorVideoOptions(options)
        const rawDefault = options.default_num_inference_steps
        if (rawDefault != null && Number.isFinite(rawDefault)) {
          const current = useStore.getState().directorVideoInferenceStepsByModel[videoModel]
          if (current == null) setVideoSteps(videoModel, rawDefault)
        }
      })
      .catch(() => {
        if (!cancelled) setDirectorVideoOptions(null)
      })
    return () => { cancelled = true }
  }, [setVideoSteps, videoModel])

  const activeDirectorVideoOptions = directorVideoOptions?.model_type === videoModel
    ? directorVideoOptions
    : null
  const videoStepsMin = Math.max(
    1,
    Math.round(Number(activeDirectorVideoOptions?.inference_steps_min ?? 1)),
  )
  const videoStepsMax = Math.max(
    videoStepsMin,
    Math.round(Number(activeDirectorVideoOptions?.inference_steps_max ?? 50)),
  )
  const clampVideoSteps = (value: number) => (
    Math.max(videoStepsMin, Math.min(videoStepsMax, Math.round(value)))
  )
  const rawDefaultVideoSteps = activeDirectorVideoOptions?.default_num_inference_steps
  const defaultVideoSteps = rawDefaultVideoSteps == null
    ? null
    : clampVideoSteps(rawDefaultVideoSteps)
  const configuredVideoSteps = videoStepsByModel[videoModel]
  const videoSteps = activeDirectorVideoOptions?.lock_inference_steps
    ? defaultVideoSteps
    : (configuredVideoSteps == null
        ? defaultVideoSteps
        : clampVideoSteps(configuredVideoSteps))
  const videoStepsLocked = activeDirectorVideoOptions?.lock_inference_steps === true
  const resolvedVideoResolution = resolveResolution(
    activeDirectorVideoOptions,
    directorResolution,
    directorAspectRatio,
  )
  const windowRecommendation = recommendedWindowProfile(
    activeDirectorVideoOptions?.director_memory_policy
      || activeDirectorVideoOptions?.sliding_window_memory_policy,
    resolvedVideoResolution,
    totalVramGb,
  )
  const safeShotFrames = windowRecommendation?.frames ?? null
  const manualMaxShotFrames = maxShotFramesByModel[videoModel] ?? null
  const framesMinimum = activeDirectorVideoOptions?.frames_minimum ?? 1
  const framesMaximum = activeDirectorVideoOptions?.frames_maximum ?? framesMinimum
  const framesStep = activeDirectorVideoOptions?.frames_steps ?? 1
  const nativeShotChoices = [124, 158, 175, 243, 345].filter(frames => (
    frames >= framesMinimum
    && frames <= framesMaximum
    && (frames - framesMinimum) % Math.max(1, framesStep) === 0
  ))
  const turboOption = activeDirectorVideoOptions?.minimax_h3_turbo
  const turboPresets = turboOption?.presets?.length
    ? turboOption.presets
    : turboOption
      ? [{
          id: turboOption.preset_id,
          label: turboOption.version_label,
          status: 'validated',
          filename: turboOption.filename,
          steps: turboOption.steps,
          weight: turboOption.weight,
          weight_min: 0.5,
          weight_max: 1.0,
          description: turboOption.guide,
          revision: '',
        }]
      : []
  const selectedTurboPreset = (
    turboPresets.find(preset => preset.id === turboPresetByModel[videoModel])
    || turboPresets.find(preset => preset.id === turboOption?.preset_id)
    || turboPresets[0]
  )
  const turboSelected = Boolean(
    turboOption && selectedTurboPreset
    && turboModeByModel[videoModel] === true
    && savedVideoLoras?.activated_loras?.includes(selectedTurboPreset.filename)
  )

  const upsamplingOptions = [
    { value: '', label: 'Off' },
    { value: 'lanczos1.5', label: 'Lanczos 1.5×' },
    { value: 'lanczos2', label: 'Lanczos 2×' },
  ]

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-2.5 py-1.5 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors"
      >
        <span>Advanced</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && (
        <div className="px-2.5 pb-2.5 space-y-3">
          {/* Director mode owns this accordion; the shared Studio Advanced
              drawer is not mounted while DirectorChat is active. */}
          <DirectorH3Optimizations />

          {shotImageSupport && shotImageSupport !== 'required' && (
            <div className="space-y-1 pt-1">
              <label className="text-[11px] text-text-secondary block">Shot image guidance</label>
              <select
                value={shotImageGuidance}
                onChange={e => setShotImageGuidance(e.target.value as DirectorShotImageGuidance)}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
              >
                <option value="auto">Auto (recommended)</option>
                <option value="prompt_only">
                  {shotImageSupport === 'direct_references' ? 'Direct references only' : 'Prompt only'}
                </option>
                <option value="generate">Generate shot images</option>
              </select>
              <p className="text-[10px] text-text-muted">
                {shotImageSupport === 'direct_references'
                  ? shotImageGuidance === 'generate'
                    ? 'Creates a composition image for each shot before H3 uses the references.'
                    : !hasVisualReferences
                      ? 'Add at least one main, character, or location image, or choose generated shot images.'
                    : 'H3 uses your character and location references directly; no image model runs.'
                  : generateShotImages
                    ? 'Creates start frames because visual references are present or generation was requested.'
                    : 'H3 renders each scene directly from its video prompt.'}
              </p>
            </div>
          )}
          {/* IMAGE section */}
          {generateShotImages && <div className="space-y-2">
            <div className="text-[10px] text-text-muted uppercase tracking-wider">Image</div>

            <div>
              <label className="text-[11px] text-text-secondary block mb-1">Upsampling</label>
              <select
                value={imgUpsampling}
                onChange={e => setImgUpsampling(e.target.value)}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
              >
                {upsamplingOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <p className="text-[10px] text-text-muted mt-0.5">
                Render then upscale the start image. Adds time per shot.
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] text-text-secondary">Film grain</label>
                <span className="text-[10px] text-text-muted tabular-nums">{imgGrain.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0} max={1} step={0.01} value={imgGrain}
                onChange={e => setImgGrain(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-[10px] text-text-muted mt-0.5">
                Aesthetic film-grain texture. 0 = off.
              </p>
              {imgGrain > 0 && (
                <div className="mt-1.5">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[10px] text-text-muted">Grain saturation</label>
                    <span className="text-[10px] text-text-muted tabular-nums">{imgGrainSat.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min={0} max={1} step={0.01} value={imgGrainSat}
                    onChange={e => setImgGrainSat(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </div>}

          {/* VIDEO section */}
          <div className="space-y-2 pt-1 border-t border-border">
            <div className="text-[10px] text-text-muted uppercase tracking-wider pt-2">Video</div>

            <div title="Applies to every newly generated Director shot and is saved with the project for later repair or regeneration.">
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] text-text-secondary">
                  {activeDirectorVideoOptions?.inference_steps_label || 'Inference steps'}
                </label>
                <div className="flex items-center gap-1.5">
                  {!videoStepsLocked && !turboSelected && defaultVideoSteps != null && configuredVideoSteps != null
                    && configuredVideoSteps !== defaultVideoSteps && (
                    <button
                      type="button"
                      onClick={() => setVideoSteps(videoModel, defaultVideoSteps)}
                      className="text-[9px] text-accent-blue hover:text-accent-blue/80"
                    >
                      Default
                    </button>
                  )}
                  <input
                    type="number"
                    min={videoStepsMin}
                    max={videoStepsMax}
                    step={1}
                    value={videoSteps ?? ''}
                    disabled={videoSteps == null || videoStepsLocked || turboSelected}
                    onChange={e => {
                      const value = Number(e.target.value)
                      if (Number.isFinite(value)) {
                        setVideoSteps(videoModel, clampVideoSteps(value))
                      }
                    }}
                    className="w-14 bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[11px] text-text-primary text-center focus:outline-none focus:border-accent-blue disabled:opacity-50"
                  />
                </div>
              </div>
              <input
                type="range"
                min={videoStepsMin}
                max={videoStepsMax}
                step={1}
                value={videoSteps ?? 1}
                disabled={videoSteps == null || videoStepsLocked || turboSelected}
                onChange={e => setVideoSteps(
                  videoModel,
                  clampVideoSteps(Number(e.target.value)),
                )}
                className="w-full disabled:opacity-50"
              />
              <p className="text-[10px] text-text-muted mt-0.5">
                {turboSelected
                  ? `H3 Turbo uses its ${selectedTurboPreset?.steps ?? turboOption?.steps ?? 6}-step recipe.`
                  : videoStepsLocked
                  ? 'Fixed by this model.'
                  : videoSteps == null
                    ? 'Loading model default...'
                    : activeDirectorVideoOptions?.inference_steps_help
                      || `Director setting for this model${defaultVideoSteps === videoSteps ? ' (default)' : ''}.`}
              </p>
            </div>

            {(activeDirectorVideoOptions?.director_memory_policy
              || activeDirectorVideoOptions?.sliding_window_memory_policy)
              && nativeShotChoices.length > 0 && (
              <div>
                <div className="flex items-center justify-between gap-2 mb-1">
                  <label className="text-[11px] text-text-secondary">Maximum planned shot</label>
                  <select
                    value={manualMaxShotFrames ?? ''}
                    onChange={event => setMaxShotFrames(
                      videoModel,
                      event.target.value ? Number(event.target.value) : null,
                    )}
                    className="bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue"
                  >
                    <option value="">Auto</option>
                    {nativeShotChoices.map(frames => (
                      <option key={frames} value={frames}>
                        {formatSeconds(frames / (activeDirectorVideoOptions.fps || 24))}
                      </option>
                    ))}
                  </select>
                </div>
                <p className={`text-[10px] ${
                  manualMaxShotFrames != null
                  && safeShotFrames != null
                  && manualMaxShotFrames > safeShotFrames
                    ? 'text-amber-400'
                    : 'text-text-muted'
                }`}>
                  {manualMaxShotFrames == null
                    ? safeShotFrames != null
                      ? `Auto plans at most ${formatSeconds(safeShotFrames / (activeDirectorVideoOptions.fps || 24))} per shot for ${resolvedVideoResolution} on ${totalVramGb.toFixed(0)} GB.`
                      : `Auto derives the one-pass limit from the selected canvas and GPU.`
                    : safeShotFrames != null && manualMaxShotFrames > safeShotFrames
                      ? `Manual override exceeds Auto's ${formatSeconds(safeShotFrames / (activeDirectorVideoOptions.fps || 24))} recommendation and may run out of VRAM.`
                      : `Manual native-shot limit. Director will plan dialogue and action to this duration.`}
                </p>
              </div>
            )}

            <div>
              <label className="text-[11px] text-text-secondary block mb-1">Upsampling</label>
              <select
                value={vidUpsampling}
                onChange={e => setVidUpsampling(e.target.value)}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
              >
                {upsamplingOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <p className="text-[10px] text-text-muted mt-0.5">
                Render then upscale the video. Adds time per shot.
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] text-text-secondary">Film grain</label>
                <span className="text-[10px] text-text-muted tabular-nums">{vidGrain.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0} max={1} step={0.01} value={vidGrain}
                onChange={e => setVidGrain(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-[10px] text-text-muted mt-0.5">
                Aesthetic film-grain texture. 0 = off.
              </p>
              {vidGrain > 0 && (
                <div className="mt-1.5">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[10px] text-text-muted">Grain saturation</label>
                    <span className="text-[10px] text-text-muted tabular-nums">{vidGrainSat.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min={0} max={1} step={0.01} value={vidGrainSat}
                    onChange={e => setVidGrainSat(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}
            </div>

            {activeDirectorVideoOptions?.self_refiner === true && <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] text-text-secondary">Self refiner</label>
                <span className="text-[9px] uppercase tracking-wider text-text-muted bg-bg-tertiary border border-border rounded px-1 py-px">
                  Experimental
                </span>
              </div>
              <select
                value={vidSelfRefiner}
                onChange={e => setVidSelfRefiner(Number(e.target.value))}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
              >
                <option value={0}>Off</option>
                <option value={1}>P1-Norm</option>
                <option value={2}>P2-Norm</option>
              </select>
              <p className="text-[10px] text-text-muted mt-0.5">
                Re-passes the rendered video through the refiner. May improve detail or introduce artifacts.
              </p>
            </div>}
          </div>
        </div>
      )}
    </div>
  )
}

/** Compact model picker for Director. Director's automated stages have a
 *  stricter input contract than Studio, so the backend publishes explicit
 *  per-workflow compatibility metadata for this selector to enforce. */
function DirectorModelPicker({ mode, value, onChange, disabled = false }: {
  mode: 'image' | 'video'
  value: string
  onChange: (modelType: string) => void
  disabled?: boolean
}) {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)
  const directorSkill = useStore(s => s.directorSkill)
  const shortFilmPath = useStore(s => s.shortFilmPath)
  const seamless = useStore(s => s.directorSeamless)

  const pipelineType: DirectorPipelineType = directorSkill === 'music_video'
    ? 'music_video'
    : shortFilmPath === 'audio'
      ? 'short_film_audio'
      : 'short_film_story'

  const groups = useMemo(() =>
    getFamiliesForMode(mode, families).map(family => ({
      family,
      models: getModelsForFamily(family.id, models, mode)
        .filter(m => enabledModels.has(m.model_type))
        .filter(m => !m.nsfw_only || nsfwMode)
        .filter(m => mode === 'image'
          ? m.director?.image.compatible === true
          : m.director?.video[pipelineType].compatible === true
            && (!seamless || m.director?.video.seamless.compatible === true)),
    })).filter(g => g.models.length > 0),
  [mode, families, models, enabledModels, nsfwMode, pipelineType, seamless])

  const compatibleModels = useMemo(
    () => groups.flatMap(group => group.models),
    [groups],
  )
  const noneSelected = mode === 'image' && value === DIRECTOR_IMAGE_MODEL_NONE
  const known = noneSelected || compatibleModels.some(model => model.model_type === value)
  const preferredId = mode === 'image' ? 'flux2_klein_9b' : 'ltx2_22B_distilled_1_1'
  const fallback = compatibleModels.find(model => model.model_type === preferredId)
    || compatibleModels[0]
  const selectedValue = known ? value : (fallback?.model_type || '')
  const selectedModel = compatibleModels.find(model => model.model_type === selectedValue)

  useEffect(() => {
    if (!disabled && !known && fallback && fallback.model_type !== value) {
      onChange(fallback.model_type)
    }
  }, [disabled, fallback, known, onChange, value])

  const title = mode === 'image'
    ? 'Choose a compatible image model for generated scene starts, or None to render from prompts and optional manual scene images.'
    : pipelineType === 'short_film_story'
      ? 'Only models that can render Director-planned shots with synchronized native audio are shown.'
      : 'Only models that can follow the uploaded soundtrack or dialogue timeline are shown.'

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-text-muted uppercase tracking-wider w-11 shrink-0">
        {mode === 'image' ? 'Image' : 'Video'}
      </span>
      <select
        value={selectedValue}
        onChange={e => onChange(e.target.value)}
        disabled={disabled || (mode === 'video' && compatibleModels.length === 0)}
        title={title}
        className="flex-1 min-w-0 bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {mode === 'image' && (
          <option value={DIRECTOR_IMAGE_MODEL_NONE}>None — no generated images</option>
        )}
        {compatibleModels.length === 0 && mode === 'video' && (
          <option value="">No compatible models enabled</option>
        )}
        {groups.map(({ family, models: famModels }) => (
          <optgroup key={family.id} label={family.label}>
            {famModels.map(m => (
              <option key={m.model_type} value={m.model_type}>{m.name}</option>
            ))}
          </optgroup>
        ))}
      </select>
      {selectedModel?.selector_help && (
        <InfoTooltip
          text={selectedModel.selector_help}
          label={`About ${selectedModel.name}`}
        />
      )}
    </div>
  )
}

function DirectorModelSelection({ disabled = false }: { disabled?: boolean }) {
  const imageModel = useStore(s => s.selectedModelPerMode.image || 'flux2_klein_9b')
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const shotImageSupport = useStore(s => s.models.find(
    model => model.model_type === videoModel,
  )?.director?.shot_image_support)
  const shotImageGuidance = useStore(s => s.directorShotImageGuidance)
  const hasVisualReferences = useStore(s => Boolean(
    s.directorReferenceImage
    || s.directorReferenceImagePath
    || s.directorCharacterRefs.length
    || s.directorCharacterRefPaths.length
    || s.directorLocationRefs.length
    || s.directorLocationRefPaths.length
    || (
      s.models.find(model => model.model_type === (
        s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
      ))?.director?.video_strategy === 'omni_reference'
      && s.directorH3References.some(
        reference => reference.type === 'image' || reference.type === 'video',
      )
    )
  ))
  const selectDirectorImageModel = useStore(s => s.selectDirectorImageModel)
  const selectDirectorVideoModel = useStore(s => s.selectDirectorVideoModel)
  const setShotImageGuidance = useStore(s => s.setDirectorShotImageGuidance)
  const generateShotImages = directorWillGenerateShotImages(
    shotImageSupport,
    shotImageGuidance,
    hasVisualReferences,
  )
  const imagePickerValue = generateShotImages
    ? imageModel
    : DIRECTOR_IMAGE_MODEL_NONE

  const selectImageWorkflow = (modelType: string) => {
    if (modelType === DIRECTOR_IMAGE_MODEL_NONE) {
      setShotImageGuidance('prompt_only')
      return
    }
    selectDirectorImageModel(modelType)
    // A concrete selection explicitly requests generated scene starts,
    // including for H3 First / Last where Auto may otherwise choose T2V.
    setShotImageGuidance('generate')
  }

  return (
    <div className="space-y-1">
      {/* Video comes first because it determines Director compatibility and
          whether generated scene-start images are useful or required. */}
      <DirectorModelPicker
        mode="video"
        value={videoModel}
        onChange={selectDirectorVideoModel}
        disabled={disabled}
      />
      <DirectorModelPicker
        mode="image"
        value={imagePickerValue}
        onChange={selectImageWorkflow}
        disabled={disabled}
      />
    </div>
  )
}

function DirectorLoraAccordion() {
  // Resolve Director's per-shoot image and video models from saved
  // per-mode selections, falling back to the same Director defaults
  // the pipeline submission uses (useStore.ts:5574). The previous
  // fallback to `s.params.model_type` was wrong — that field carries
  // the CURRENTLY-ACTIVE Studio model, which on fresh launch is
  // whatever Studio happens to be in (usually video). On a fresh
  // launch, that meant Director's "Image LoRAs" accordion would
  // fetch the video model's LoRA dir and display LTX loras under
  // the image header. Falling back to a known image-class default
  // (flux2_klein_9b) instead of the active Studio model fixes that.
  const imageModel = useStore(s => s.selectedModelPerMode.image || 'flux2_klein_9b')
  const videoModel = useStore(s => s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1')
  const videoLorasDisabled = useStore(s => s.models.find(
    model => model.model_type === videoModel,
  )?.loras_disabled === true)
  const shotImageSupport = useStore(s => s.models.find(
    model => model.model_type === videoModel,
  )?.director?.shot_image_support)
  const shotImageGuidance = useStore(s => s.directorShotImageGuidance)
  const hasVisualReferences = useStore(s => Boolean(
    s.directorReferenceImage
    || s.directorReferenceImagePath
    || s.directorCharacterRefs.length
    || s.directorCharacterRefPaths.length
    || s.directorLocationRefs.length
    || s.directorLocationRefPaths.length
    || (
      s.models.find(model => model.model_type === videoModel)
        ?.director?.video_strategy === 'omni_reference'
      && s.directorH3References.some(
        reference => reference.type === 'image' || reference.type === 'video',
      )
    )
  ))
  const generateShotImages = directorWillGenerateShotImages(
    shotImageSupport,
    shotImageGuidance,
    hasVisualReferences,
  )
  const [imageOpen, setImageOpen] = useState(false)
  const [videoOpen, setVideoOpen] = useState(false)

  return (
    <div className="space-y-1">
      {/* Image LoRAs */}
      {generateShotImages && imageModel && (
        <div className="border border-border rounded-lg overflow-hidden">
          <button
            onClick={() => setImageOpen(!imageOpen)}
            className="w-full flex items-center justify-between px-2.5 py-1.5 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors"
          >
            <span>Image LoRAs</span>
            {imageOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
          {imageOpen && (
            <div className="px-2.5 pb-2">
              <DirectorLoraSelector mode="image" modelType={imageModel} />
            </div>
          )}
        </div>
      )}
      {/* Video LoRAs */}
      {videoModel && !videoLorasDisabled && (
        <div className="border border-border rounded-lg overflow-hidden">
          <button
            onClick={() => setVideoOpen(!videoOpen)}
            className="w-full flex items-center justify-between px-2.5 py-1.5 text-[11px] text-text-secondary hover:bg-bg-hover transition-colors"
          >
            <span>Video LoRAs</span>
            {videoOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
          {videoOpen && (
            <div className="px-2.5 pb-2">
              <DirectorLoraSelector mode="video" modelType={videoModel} />
            </div>
          )}
        </div>
      )}
      {videoModel && videoLorasDisabled && (
        <p className="rounded-lg border border-amber-500/25 bg-amber-500/8 px-2.5 py-2 text-[9px] leading-relaxed text-text-muted">
          Video LoRAs are disabled because this model already contains its Turbo and Mystic adapters.
        </p>
      )}
    </div>
  )
}

function DirectorGenerationOptions() {
  const audioFile = useStore(s => s.directorAudioFile)
  const fixedMediaStrength = useStore(s => {
    const selected = s.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const model = s.models.find(item => item.model_type === selected)
    return directorModelUsesFixedMediaStrength(selected, model?.architecture)
  })

  return (
    <div className="space-y-2">
      <span className="text-[10px] text-text-muted uppercase tracking-wider block">
        Generation Options
      </span>
      <DirectorLoraAccordion />
      <DirectorAdvancedAccordion />
      {audioFile && !fixedMediaStrength && (
        <div className="pt-2 border-t border-border/50">
          <AudioScaleSlider />
        </div>
      )}
    </div>
  )
}

function StyleForm({
  speakers, speakerMappings, speakerSamples, setSpeakerMapping, insertSpeakerMention, isActive, isShortFilm, isStoryPath,
}: {
  speakers: string[]
  speakerMappings: ReturnType<typeof useStore.getState>['directorSpeakerMappings']
  speakerSamples: Record<string, string[]>
  setSpeakerMapping: (speakerId: string, name: string, role: 'rapping' | 'singing' | 'speaking' | '') => void
  insertSpeakerMention: (speakerId: string) => void
  isActive: boolean
  isShortFilm?: boolean
  isStoryPath?: boolean
}) {
  if (!isActive) {
    return (
      <div className="space-y-2">
        <p className="text-xs text-text-muted">
          {isStoryPath ? 'Story submitted. Planning scenes and writing prompts...'
            : isShortFilm ? 'Story description submitted. Planning scenes...'
            : 'Scene description submitted. Planning shots...'}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-text-secondary">
        {isStoryPath
          ? 'Describe the story you want to tell. The AI will plan scenes, write dialogue, and create all prompts.'
          : isShortFilm
            ? 'Describe the story setting, mood, and visual style for your short film.'
            : 'Describe the scene, characters, and visual style.'}
      </p>

      {/* Speaker Mapping — hidden for story path (no audio = no detected speakers) */}
      {!isStoryPath && speakers.length >= 1 && (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider block mb-1">Speakers Detected</label>
          <div className="space-y-2">
            {speakerMappings.map((mapping) => (
              <div key={mapping.speakerId} className="bg-bg-tertiary rounded-lg p-2 space-y-1">
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => insertSpeakerMention(mapping.speakerId)}
                    className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 shrink-0 transition-colors"
                    title={`Insert @${mapping.speakerId} into description`}
                  >
                    {mapping.speakerId}
                  </button>
                  <input
                    type="text"
                    value={mapping.name}
                    onChange={e => setSpeakerMapping(mapping.speakerId, e.target.value, mapping.role)}
                    placeholder="e.g. man in green hoodie"
                    className="flex-1 bg-bg-secondary border border-border rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue transition-colors"
                  />
                  <select
                    value={mapping.role}
                    onChange={e => setSpeakerMapping(mapping.speakerId, mapping.name, e.target.value as typeof mapping.role)}
                    className="bg-bg-secondary border border-border rounded px-1.5 py-1 text-[10px] text-text-secondary focus:outline-none focus:border-accent-blue transition-colors"
                  >
                    <option value="">role</option>
                    {!isShortFilm && <option value="rapping">rapping</option>}
                    {!isShortFilm && <option value="singing">singing</option>}
                    <option value="speaking">speaking</option>
                  </select>
                </div>
                {speakerSamples[mapping.speakerId] && (
                  <div className="text-[9px] text-text-muted pl-1 italic">
                    {speakerSamples[mapping.speakerId].map((line, li) => (
                      <div key={li} className="truncate">&ldquo;{line}&rdquo;</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">
            Name each speaker so the director knows who to show. Click a chip to insert into description.
          </span>
        </div>
      )}

      <p className="text-[11px] text-text-muted">
        {isStoryPath
          ? 'Describe your story in the input below and press send. The AI will plan everything.'
          : isShortFilm
            ? 'Type your story description in the input below and press send.'
            : 'Type your scene description in the input below and press send.'}
      </p>
    </div>
  )
}

function ImagePromptsReview({
  clipPlans, plannedClips, speakerMappings, editClipPlan, planPrompts,
  generateStartImages, loading, isActive, isShortFilm,
}: {
  clipPlans: ReturnType<typeof useStore.getState>['directorClipPlans']
  plannedClips: ReturnType<typeof useStore.getState>['directorPlannedClips']
  speakerMappings: ReturnType<typeof useStore.getState>['directorSpeakerMappings']
  editClipPlan: (index: number, field: 'video_prompt' | 'image_prompt', value: string) => void
  planPrompts: () => Promise<void>
  planVideoPrompts: () => Promise<void>
  generateStartImages: () => Promise<void>
  loading: boolean
  isActive: boolean
  isShortFilm?: boolean
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Start Image Prompts</label>
        {isActive && (
          <button
            onClick={planPrompts}
            disabled={loading}
            className="text-[10px] text-accent-blue hover:text-accent-blue-hover flex items-center gap-0.5"
          >
            <RotateCcw size={10} /> Regenerate
          </button>
        )}
      </div>

      {/* No inner scroll — the chat panel handles scrolling. The list
          extends to the natural total height of all clip cards. */}
      <div className="space-y-2">
        {clipPlans.map((plan, i) => {
          const clip = plannedClips[i]
          return (
            <div key={i} className="bg-bg-tertiary rounded-lg p-2 space-y-1.5">
              <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
                <span className="font-medium text-text-secondary">{isShortFilm ? 'Shot' : 'Clip'} {i + 1}</span>
                {clip && (
                  <>
                    <span>{formatTime(clip.start)}-{formatTime(clip.end)}</span>
                    {!isShortFilm && <span>{clip.beat_count}b</span>}
                    <SectionBadge label={clip.section_label} />
                    {!isShortFilm && <EnergyDot energy={clip.energy} />}
                    {clip.dominant_speaker && (
                      <span className="text-accent-blue">
                        {speakerMappings.find(m => m.speakerId === clip.dominant_speaker)?.name || clip.dominant_speaker}
                      </span>
                    )}
                  </>
                )}
              </div>
              {/* AutoResizeTextarea grows with content — no internal
                  scroll on long prompts. rows={4} provides a sensible
                  initial height before content is loaded. */}
              <AutoResizeTextarea
                value={plan.image_prompt}
                onChange={e => editClipPlan(i, 'image_prompt', e.target.value)}
                rows={4}
                disabled={!isActive}
                className="w-full bg-bg-secondary border border-border rounded px-2 py-1.5 text-xs text-text-primary resize-none focus:outline-none focus:border-accent-blue transition-colors disabled:opacity-60"
              />
            </div>
          )
        })}
      </div>

      {isActive && (
        <button
          onClick={generateStartImages}
          disabled={loading}
          className="w-full py-2 rounded-lg bg-accent-blue text-white text-xs font-medium hover:bg-accent-blue-hover transition-colors flex items-center justify-center gap-1.5"
        >
          {/* Always available now — directorGenerateStartImages generates an
              establishing/anchor image first when no reference was provided. */}
          <ImageIcon size={12} /> Generate Start Images
        </button>
      )}
    </div>
  )
}

function ImageGenView({
  loading, imageGenProgress, clipImages,
}: {
  loading: boolean
  imageGenProgress: ReturnType<typeof useStore.getState>['directorImageGenProgress']
  clipImages: ReturnType<typeof useStore.getState>['directorClipImages']
  planVideoPrompts: () => Promise<void>
}) {
  // Architecture-mismatch advisories from the backend's image-gen filter.
  // Surfacing these in chat (vs only in the console) lets the user see
  // immediately why some of their selected LoRAs didn't get applied —
  // most commonly a Flux 2 Dev–trained LoRA that won't load against
  // Klein 9B's narrower hidden dim.
  const loraWarnings = useStore(s => s.pipelineStatus?.lora_warnings) || []
  return (
    <div className="space-y-3">
      <label className="text-[11px] text-text-muted uppercase tracking-wider block">Generating Start Images</label>

      {loraWarnings.length > 0 && (
        <div className="space-y-1.5">
          {loraWarnings.map((w, i) => (
            <div key={i} className="px-2.5 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-[11px] text-text-primary leading-snug whitespace-pre-line">
              {w}
            </div>
          ))}
        </div>
      )}


      {imageGenProgress && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-text-secondary">
              {imageGenProgress.status === 'done'
                ? 'All images ready — planning video shots...'
                : `Clip ${imageGenProgress.current + 1} of ${imageGenProgress.total}`}
            </span>
            <span className="text-text-muted">
              {imageGenProgress.currentClipLabel}
              {imageGenProgress.status !== 'done' && ` — ${imageGenProgress.status}`}
            </span>
          </div>
          <div className="w-full bg-bg-tertiary rounded-full h-1.5">
            <div
              className="bg-accent-blue h-1.5 rounded-full transition-all"
              style={{
                width: `${imageGenProgress.status === 'done'
                  ? 100
                  : ((imageGenProgress.current + (imageGenProgress.status === 'polling' ? 0.5 : 0)) / imageGenProgress.total) * 100
                }%`,
              }}
            />
          </div>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center gap-2 py-2">
          <Loader2 size={16} className="animate-spin text-accent-blue" />
          <span className="text-[11px] text-text-muted">
            {imageGenProgress?.status === 'generating' ? 'Submitting...' :
             imageGenProgress?.status === 'polling' ? 'Waiting for result...' :
             imageGenProgress?.status === 'downloading' ? 'Downloading...' : 'Processing...'}
          </span>
        </div>
      )}

      {clipImages.length > 0 && (
        // No inner scroll — chat panel scrolls. Grid wraps naturally
        // and extends downward as more images come in.
        <div className="grid grid-cols-3 gap-1.5">
          {clipImages.map((img, i) => (
            <div key={i} className="relative">
              <img
                src={img.file ? URL.createObjectURL(img.file) : getFileUrl(img.filename)}
                alt={`Clip ${img.clipIndex + 1}`}
                className="w-full aspect-square object-cover rounded-lg border border-border"
              />
              <span className="absolute bottom-0.5 left-0.5 text-[8px] bg-black/60 text-white px-1 py-0.5 rounded">
                {img.clipIndex + 1}
              </span>
            </div>
          ))}
        </div>
      )}

      {!loading && imageGenProgress?.status === 'error' && (
        <button
          onClick={() => { useStore.setState({ directorStep: 'review_video' }) }}
          className="w-full py-2 rounded-lg bg-accent-blue text-white text-xs font-medium hover:bg-accent-blue-hover transition-colors flex items-center justify-center gap-1.5"
        >
          <ChevronRight size={12} /> Continue to Video Prompts
        </button>
      )}
    </div>
  )
}

function VideoPromptsReview({
  clipPlans, plannedClips, clipImages, setClipImage, allowSceneImageUploads,
  speakerMappings, editClipPlan,
  planVideoPrompts, directorGenerate, queueCurrent, applyToClips, loading, isShortFilm,
  isGenerating, isAutoGenerating, editingQueueEntryId,
}: {
  clipPlans: ReturnType<typeof useStore.getState>['directorClipPlans']
  plannedClips: ReturnType<typeof useStore.getState>['directorPlannedClips']
  clipImages: ReturnType<typeof useStore.getState>['directorClipImages']
  setClipImage: (clipIndex: number, file: File | null) => void
  allowSceneImageUploads?: boolean
  speakerMappings: ReturnType<typeof useStore.getState>['directorSpeakerMappings']
  editClipPlan: (index: number, field: 'video_prompt' | 'image_prompt', value: string) => void
  planVideoPrompts: () => Promise<void>
  directorGenerate: () => void
  queueCurrent: () => Promise<void>
  applyToClips: () => void
  loading: boolean
  isShortFilm?: boolean
  /** True when any render is active. Generate remains available, but saves
   *  the edited state as a held/queued immutable revision. */
  isGenerating?: boolean
  /** True specifically when an auto-mode pipeline is active. */
  isAutoGenerating?: boolean
  editingQueueEntryId?: string | null
}) {
  const queueBusy = useStore(s => s.directorQueueLoading)
  const [queueConfirmation, setQueueConfirmation] = useState<string | null>(null)

  useEffect(() => {
    if (!queueConfirmation) return
    const timer = window.setTimeout(() => setQueueConfirmation(null), 5000)
    return () => window.clearTimeout(timer)
  }, [queueConfirmation])

  const handleAddToQueue = async () => {
    setQueueConfirmation(null)
    const beforeIds = new Set(
      (useStore.getState().directorQueue?.entries || []).map(entry => entry.id),
    )
    await queueCurrent()
    const state = useStore.getState()
    const queue = state.directorQueue
    const added = queue?.entries.find(entry => !beforeIds.has(entry.id))
    if (!queue || !added || state.directorError) return
    const heldCount = queue.entries.filter(
      entry => ['held', 'queued', 'running'].includes(entry.status),
    ).length
    setQueueConfirmation(
      `Added to Queue · ${heldCount} Director ${heldCount === 1 ? 'project' : 'projects'} waiting. Open the queue in the top bar when ready.`,
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-text-muted uppercase tracking-wider">Video Prompts</label>
        <button
          onClick={planVideoPrompts}
          disabled={loading}
          className="text-[10px] text-accent-blue hover:text-accent-blue-hover flex items-center gap-0.5"
        >
          <RotateCcw size={10} /> Regenerate
        </button>
      </div>

      {allowSceneImageUploads && (
        <p className="text-[10px] text-text-muted leading-snug">
          Scene images are optional. Add one to anchor a shot, or leave it blank to render from its video prompt.
        </p>
      )}

      {clipImages.length > 0 && !allowSceneImageUploads && (
        <div className="grid grid-cols-5 gap-1 mb-1">
          {clipImages.map((img, i) => (
            <div key={i} className="relative">
              <img
                src={img.file ? URL.createObjectURL(img.file) : getFileUrl(img.filename)}
                alt={`Clip ${img.clipIndex + 1}`}
                className="w-full aspect-square object-cover rounded border border-border"
              />
              <span className="absolute bottom-0 left-0 text-[7px] bg-black/60 text-white px-0.5 rounded-br">
                {img.clipIndex + 1}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* No inner scroll — chat panel handles it. AutoResizeTextarea
          grows each prompt to its full height so long video prompts
          don't double-scroll. */}
      <div className="space-y-2">
        {clipPlans.map((plan, i) => {
          const clip = plannedClips[i]
          const clipImage = clipImages.find(image => image.clipIndex === i)
          return (
            <div key={i} className="bg-bg-tertiary rounded-lg p-2 space-y-1.5">
              <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
                <span className="font-medium text-text-secondary">{isShortFilm ? 'Shot' : 'Clip'} {i + 1}</span>
                {clip && (
                  <>
                    <span>{formatTime(clip.start)}-{formatTime(clip.end)}</span>
                    <SectionBadge label={clip.section_label} />
                    {clip.dominant_speaker && (
                      <span className="text-accent-blue">
                        {speakerMappings.find(m => m.speakerId === clip.dominant_speaker)?.name || clip.dominant_speaker}
                      </span>
                    )}
                  </>
                )}
              </div>
              {allowSceneImageUploads && (
                <div className="flex items-center gap-2 rounded-md border border-border bg-bg-secondary p-1.5">
                  {clipImage && (
                    <img
                      src={clipImage.file ? URL.createObjectURL(clipImage.file) : getFileUrl(clipImage.filename)}
                      alt={`${isShortFilm ? 'Shot' : 'Clip'} ${i + 1} start`}
                      className="h-10 w-10 shrink-0 rounded object-cover"
                    />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[10px] text-text-secondary">
                      {clipImage?.filename || 'No scene image'}
                    </div>
                    <div className="text-[9px] text-text-muted">
                      {clipImage ? 'Used as this shot’s start image' : 'Prompt-only video'}
                    </div>
                  </div>
                  <label className="shrink-0 cursor-pointer rounded border border-border px-2 py-1 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary">
                    <input
                      type="file"
                      accept={IMAGE_ACCEPT}
                      className="hidden"
                      onChange={event => {
                        const file = event.target.files?.[0]
                        if (file) setClipImage(i, file)
                        event.currentTarget.value = ''
                      }}
                    />
                    {clipImage ? 'Replace' : 'Upload'}
                  </label>
                  {clipImage && (
                    <button
                      type="button"
                      onClick={() => setClipImage(i, null)}
                      title="Remove scene image"
                      className="shrink-0 rounded p-1 text-text-muted hover:bg-bg-hover hover:text-red-400"
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
              )}
              <AutoResizeTextarea
                value={plan.video_prompt}
                onChange={e => editClipPlan(i, 'video_prompt', e.target.value)}
                rows={4}
                className="w-full bg-bg-secondary border border-border rounded px-2 py-1.5 text-xs text-text-primary resize-none focus:outline-none focus:border-accent-blue transition-colors"
              />
            </div>
          )
        })}
      </div>

      <div className="space-y-2">
        <button
          onClick={directorGenerate}
          disabled={(loading && !isGenerating) || queueBusy}
          className="w-full py-2.5 rounded-lg bg-accent-green hover:bg-accent-green-hover text-white text-sm font-semibold transition-colors flex items-center justify-center gap-1.5 disabled:opacity-50"
          title={isGenerating
            ? 'Freeze these edited settings as a queued revision; the active run is unchanged'
            : editingQueueEntryId
              ? 'Replace the held queue entry with these edited settings'
              : 'Render this Director project as a new revision'}
        >
          {isGenerating || editingQueueEntryId
            ? <ListVideo size={14} /> : <Play size={14} fill="white" />}
          {editingQueueEntryId
            ? 'Save Queue Changes'
            : isGenerating
            ? (isAutoGenerating ? 'Queue Edited Variant' : 'Add Variant to Queue')
            : 'Generate'}
        </button>
        {!isGenerating && !editingQueueEntryId && (
          <>
            <button
              onClick={() => void handleAddToQueue()}
              disabled={loading || queueBusy || Boolean(queueConfirmation)}
              className={`w-full py-2 rounded-lg border text-xs font-medium transition-colors flex items-center justify-center gap-1.5 disabled:opacity-70 ${
                queueConfirmation
                  ? 'border-green-500/30 bg-green-500/10 text-indicator-success'
                  : 'border-accent-blue/30 bg-accent-blue/5 text-accent-blue hover:bg-accent-blue/10'
              }`}
              title="Hold this complete project in the persistent queue without starting it"
            >
              {queueBusy
                ? <Loader2 size={12} className="animate-spin" />
                : queueConfirmation
                  ? <Check size={12} />
                  : <ListVideo size={12} />}
              {queueBusy ? 'Adding…' : queueConfirmation ? 'Added to Queue' : 'Add to Queue'}
            </button>
            {queueConfirmation && (
              <div
                role="status"
                aria-live="polite"
                className="rounded-md border border-green-500/20 bg-green-500/5 px-2.5 py-2 text-[10px] leading-relaxed text-indicator-success"
              >
                {queueConfirmation}
              </div>
            )}
          </>
        )}
        <button
          onClick={applyToClips}
          className="w-full py-2 rounded-lg border border-border text-text-secondary text-xs font-medium hover:bg-bg-hover hover:text-text-primary transition-colors flex items-center justify-center gap-1.5"
        >
          <ChevronRight size={12} /> Edit in Studio
        </button>
      </div>
    </div>
  )
}
