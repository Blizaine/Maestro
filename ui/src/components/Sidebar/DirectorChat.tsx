import { useState, useCallback, useRef, useMemo, useEffect } from 'react'
import { Upload, Loader2, Music, RotateCcw, Check, X, ChevronRight, ChevronDown, ImageIcon, Play, Film, Mic, Sparkles, Send, Users, FileText, Clock } from 'lucide-react'
import { useStore, getFamiliesForMode, getModelsForFamily } from '../../stores/useStore'
import { getFileUrl } from '../../api/client'
import { DirectorLoraSelector } from '../SettingsDrawer/DirectorLoraSelector'
import { DirectorSongSetup } from './DirectorSongSetup'
import type { DirectorSkill, ShortFilmCharacter, ShortFilmPath } from '../../types'

// AUDIO_ACCEPT lists both audio formats AND video formats. When a video
// file is uploaded, the backend's /api/v1/upload-audio endpoint extracts
// the audio track via ffmpeg and returns a WAV path. The user sees the
// same workflow either way — they can drop a music video here and get
// the soundtrack analyzed without converting first.
const AUDIO_ACCEPT = '.wav,.mp3,.flac,.ogg,.m4a,.mp4,.mov,.mkv,.webm,.avi,.m4v'
const IMAGE_ACCEPT = '.png,.jpg,.jpeg,.webp,.bmp'

function AudioScaleSlider() {
  const audioScale = useStore(s => s.directorAudioScale)
  const setAudioScale = useStore(s => s.setDirectorAudioScale)
  return (
    <>
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
      <div className="flex gap-2 text-[8px] text-text-muted">
        <span>1x</span>
        <span>3x TTS</span>
        <span>5x</span>
      </div>
    </>
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
 */
function AutoResizeTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [props.value])
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
  const sceneDescription = useStore(s => s.directorSceneDescription)
  const audioFile = useStore(s => s.directorAudioFile)
  const referenceImage = useStore(s => s.directorReferenceImage)
  const clipImages = useStore(s => s.directorClipImages)
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
  const setAutoMode = useStore(s => s.setDirectorAutoMode)
  const seamless = useStore(s => s.directorSeamless)
  const setSeamless = useStore(s => s.setDirectorSeamless)
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
  const shortFilmSetTargetDuration = useStore(s => s.shortFilmSetTargetDuration)
  const shortFilmNarrative = useStore(s => s.shortFilmNarrative)
  const shortFilmSetNarrative = useStore(s => s.shortFilmSetNarrative)
  const startDirectorPipeline = useStore(s => s.startDirectorPipeline)
  const pipelinePhase = useStore(s => s.pipelineStatus?.phase)

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

  // Sync chatInput with store's sceneDescription when entering style step
  useEffect(() => {
    if (step === 'style' && sceneDescription && !chatInput) {
      setChatInput(sceneDescription)
    }
  }, [step])

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
      if (songDescription.trim() && !loading) generateTrack()
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

        {/* Aspect ratio + resolution selectors (always visible before skill selection) */}
        {!skill && (
          <div className="space-y-2">
            <DirectorAspectRatioSelector />
            <DirectorResolutionSelector />
          </div>
        )}

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
        {skill && (!isShortFilm || shortFilmPath === 'audio') && (atStep('upload') || atStep('analyze') || pastStep('analyze')) && (
          <>
            {!audioFile && !pastStep('analyze') ? (
              <SystemBubble>
                <p className="text-xs text-text-secondary mb-2">
                  {isShortFilm
                    ? 'Upload a reference photo of your characters and dialogue audio to get started.'
                    : 'Add a reference photo (optional), then upload a track or generate one.'}
                </p>
                <div className="space-y-3">
                  <ReferenceImageUpload
                    referenceImage={referenceImage}
                    refImagePreview={refImagePreview}
                    setReferenceImage={setReferenceImage}
                  />
                  {<AdditionalRefsSection />}
                  {isShortFilm && referenceImage && (
                    <CharacterNaming
                      characters={shortFilmCharacters}
                      setCharacters={shortFilmSetCharacters}
                    />
                  )}
                  {/* Music Video: upload a track OR generate one with ACE-Step */}
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
                  {/* Music Video: up-front options (LoRA + post-processing) live
                      here at the first step instead of buried mid-flow. */}
                  {!isShortFilm && (
                    <div className="pt-1 border-t border-border/50 space-y-1">
                      <DirectorLoraAccordion />
                      <DirectorAdvancedAccordion />
                    </div>
                  )}
                  {/* Track-generation progress renders LAST in the bubble so
                      the newest activity is the bottom-most chat content (the
                      scroll anchor brings it into view) — previously it sat
                      above the LoRA/Advanced accordions. */}
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
                  {/* Keep the reference selections VISIBLE during analysis —
                      they used to unmount behind a `!loading` gate, which read
                      as "my selections disappeared". Interaction is disabled
                      while loading; the state is untouched. */}
                  <div className={loading ? 'opacity-60 pointer-events-none' : ''}>
                    <ReferenceImageUpload
                      referenceImage={referenceImage}
                      refImagePreview={refImagePreview}
                      setReferenceImage={setReferenceImage}
                    />
                    {<AdditionalRefsSection />}
                  </div>
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
                <ReferenceImageUpload
                  referenceImage={referenceImage}
                  refImagePreview={refImagePreview}
                  setReferenceImage={setReferenceImage}
                  compact
                />
                {<AdditionalRefsSection />}
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
                  Set up your short film. Upload a reference photo, name your characters, and set the target duration.
                </p>
                <div className="space-y-3">
                  <ReferenceImageUpload
                    referenceImage={referenceImage}
                    refImagePreview={refImagePreview}
                    setReferenceImage={setReferenceImage}
                  />
                  {<AdditionalRefsSection />}
                  {referenceImage && (
                    <CharacterNaming
                      characters={shortFilmCharacters}
                      setCharacters={shortFilmSetCharacters}
                    />
                  )}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-text-muted flex items-center gap-1">
                        <Clock size={10} /> Target Duration
                      </span>
                      <span className="text-[10px] text-text-primary font-medium">{shortFilmTargetDuration}s</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={300}
                      step={5}
                      value={shortFilmTargetDuration}
                      onChange={e => shortFilmSetTargetDuration(Number(e.target.value))}
                      className="w-full h-1 bg-bg-hover rounded-lg appearance-none cursor-pointer accent-accent-blue"
                    />
                    <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
                      <span>10s</span>
                      <span>5m</span>
                    </div>
                  </div>
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
                {pipelinePhase === 'polishing_prompts'
                  ? 'Polishing prompts (3rd pass)...'
                  : isStoryPath ? 'Planning scenes and writing prompts...' : isShortFilm ? 'Writing scene prompts...' : 'Writing image prompts...'}
              </span>
            </div>
            {pipelinePhase !== 'polishing_prompts' && <LlmThinkingStream stage="plan" />}
          </SystemBubble>
        )}

        {/* Completed planning streams stay in the chat history */}
        {(pastStep('plan') || (atStep('plan') && !loading)) && (
          <LlmLogStage stage="plan" label={isShortFilm ? 'Scene planning' : 'Image prompts'} />
        )}

        {/* Review step (image prompts) */}
        {(atStep('review') || pastStep('review')) && (
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
        {(atStep('generate_images') || pastStep('generate_images')) && (
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
              speakerMappings={speakerMappings}
              editClipPlan={editClipPlan}
              planVideoPrompts={isShortFilm ? shortFilmPlanVideoPrompts : planVideoPrompts}
              directorGenerate={directorGenerate}
              applyToClips={applyToClips}
              loading={loading}
              isShortFilm={isShortFilm}
              // "Generating" is true if EITHER a manual job is running
              // (isGenerating) OR an auto-mode pipeline is actively
              // running. Auto mode doesn't push jobs into the same job
              // queue — it has its own pipelineStatus state machine —
              // so without the OR the button looks pressable during
              // auto generation. The label flips to "Auto Generating…"
              // when the pipeline is what's running, so it's clear
              // *why* the button is disabled.
              isGenerating={isGenerating || pipelinePhase !== undefined}
              isAutoGenerating={autoMode && pipelinePhase !== undefined}
            />
          </SystemBubble>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Persistent toggles + Chat input bar */}
      <div className="px-4 py-3 border-t border-border space-y-2">
        {skill && (
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-1.5 cursor-pointer select-none" title="Each clip's end frame uses the next clip's start image for smooth transitions">
                <input
                  type="checkbox"
                  checked={seamless}
                  onChange={e => setSeamless(e.target.checked)}
                  className="accent-accent-blue w-3 h-3"
                />
                <span className="text-[10px] text-text-secondary">Seamless</span>
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer select-none" title="Skip all review steps and generate automatically">
                <input
                  type="checkbox"
                  checked={autoMode}
                  onChange={e => setAutoMode(e.target.checked)}
                  className="accent-red-500 w-3 h-3"
                />
                <span className={`text-[10px] ${autoMode ? 'text-red-400' : 'text-text-secondary'}`}>Auto</span>
              </label>
            </div>
            {audioFile && (
              <div className="flex items-center gap-2">
                <AudioScaleSlider />
                <div className="flex gap-2 text-[8px] text-text-muted">
                  <span>1x</span>
                  <span>3x TTS</span>
                  <span>5x</span>
                </div>
              </div>
            )}
          </div>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={mvGenerateSetup ? songDescription : chatInput}
            onChange={e => {
              const v = e.target.value
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
            className="flex-1 bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted resize-none focus:outline-none focus:border-accent-blue transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleChatSubmit}
            disabled={!chatInputEnabled || !(mvGenerateSetup ? songDescription : chatInput).trim()}
            className="p-2 rounded-lg bg-accent-blue text-white hover:bg-accent-blue-hover transition-colors disabled:opacity-30 disabled:cursor-not-allowed shrink-0"
          >
            {loading && (step === 'style' || isMusicVideo) ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
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

function DirectorAspectRatioSelector() {
  const ratio = useStore(s => s.directorAspectRatio)
  const setRatio = useStore(s => s.setDirectorAspectRatio)
  const presets = [
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
            className={`flex-1 py-1.5 rounded-lg border text-xs transition-all ${
              ratio === p.value
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            }`}
          >
            <div className="font-medium">{p.label}</div>
            <div className="text-[9px] mt-0.5 opacity-60">{p.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}

function DirectorResolutionSelector() {
  const resolution = useStore(s => s.directorResolution)
  const setResolution = useStore(s => s.setDirectorResolution)
  // Just the resolution labels — the previous Fast/Faster/Balanced/
  // High Quality descriptors were inaccurate (480p was labeled "Fast"
  // but 540p was "Faster", which is the wrong direction) and added
  // visual noise without adding info a user couldn't infer from the
  // resolution number itself.
  const presets = [
    { value: '480p' as const, label: '480p' },
    { value: '540p' as const, label: '540p' },
    { value: '720p' as const, label: '720p' },
    { value: '1080p' as const, label: '1080p' },
  ]
  return (
    <div>
      <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5 block">Resolution</label>
      <div className="flex gap-1.5">
        {presets.map(p => (
          <button
            key={p.value}
            onClick={() => setResolution(p.value)}
            className={`flex-1 py-2 rounded-lg border text-xs font-medium transition-all ${
              resolution === p.value
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
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
  referenceImage, refImagePreview, setReferenceImage, compact,
}: {
  referenceImage: File | null
  refImagePreview: string | null
  setReferenceImage: (file: File | null) => void
  compact?: boolean
}) {
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

  if (compact) {
    return (
      <div className="space-y-2">
        {referenceImage && refImagePreview ? (
          <div className="flex items-center gap-2">
            <label className="cursor-pointer">
              <img
                src={refImagePreview}
                alt="Reference"
                className="w-10 h-10 object-cover rounded-lg border border-border hover:border-accent-blue transition-colors"
                title="Click to change"
              />
              <input
                type="file"
                accept={IMAGE_ACCEPT}
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) setReferenceImage(f) }}
              />
            </label>
            <div className="flex-1 min-w-0">
              <span className="text-[10px] text-text-muted">Reference photo</span>
            </div>
            <button
              onClick={() => setReferenceImage(null)}
              className="p-1 rounded hover:bg-bg-hover transition-colors"
              title="Remove"
            >
              <X size={12} className="text-text-muted" />
            </button>
          </div>
        ) : (
          <label className="cursor-pointer flex items-center gap-2 border border-dashed border-border rounded px-2 py-1.5 hover:border-accent-blue transition-colors">
            <ImageIcon size={12} className="text-text-muted" />
            <span className="text-[10px] text-text-muted">Add reference photo</span>
            <input
              type="file"
              accept={IMAGE_ACCEPT}
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) setReferenceImage(f) }}
            />
          </label>
        )}
        {referenceImage && (
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
      {referenceImage && (
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
  const [expanded, setExpanded] = useState(charRefs.length > 0 || locRefs.length > 0 || voiceRef !== null)

  const handleFiles = useCallback((files: FileList | null, type: 'char' | 'loc') => {
    if (!files) return
    const add = type === 'char' ? addCharRef : addLocRef
    Array.from(files).forEach(f => { if (f.type.startsWith('image/')) add(f) })
  }, [addCharRef, addLocRef])

  const totalRefs = charRefs.length + locRefs.length + (voiceRef ? 1 : 0)

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
          {/* Voice Reference (ID-LoRA) — gated by Services toggle */}
          {useStore.getState().servicesConfig?.voice_reference_enabled && <div>
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
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] text-text-muted whitespace-nowrap">Identity scale</span>
                  <input type="range" min={0} max={10} step={0.5} value={identityScale}
                    onChange={e => setIdentityScale(parseFloat(e.target.value))}
                    className="flex-1 h-1 accent-accent-blue" />
                  <span className="text-[9px] text-text-muted w-5 text-right">{identityScale}</span>
                </div>
              </div>
            ) : (
              <p className="text-[9px] text-text-muted italic">~5 sec voice sample for consistent voice across clips</p>
            )}
          </div>}
        </div>
      )}
    </div>
  )
}

function AnalysisSummary({
  analysis, showDetails, setShowDetails, speakerMappings: _speakerMappings, isShortFilm,
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
 * DirectorAdvancedAccordion — collapsed-by-default panel exposing the
 * post-processing knobs (spatial upsampling, film grain, self refiner)
 * that used to live in the Director Parameters settings tab. Sits in
 * the chat sidebar alongside the LoRA accordion so per-shoot tweaks
 * are co-located with the rest of the per-shoot setup.
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
          {/* IMAGE section */}
          <div className="space-y-2">
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
          </div>

          {/* VIDEO section */}
          <div className="space-y-2 pt-1 border-t border-border">
            <div className="text-[10px] text-text-muted uppercase tracking-wider pt-2">Video</div>

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

            <div>
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
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/** Compact model picker for Director — same visibility rules as the Studio
 *  ModelSelector (enabledModels + Mature Mode gate), grouped by family.
 *  Changing it updates selectedModelPerMode, which the pipeline submission
 *  reads AND the LoRA accordions below re-fetch from (their modelType prop). */
function DirectorModelPicker({ mode, value, onChange }: {
  mode: 'image' | 'video'
  value: string
  onChange: (modelType: string) => void
}) {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)

  const groups = useMemo(() =>
    getFamiliesForMode(mode, families).map(family => ({
      family,
      models: getModelsForFamily(family.id, models, mode)
        .filter(m => enabledModels.has(m.model_type))
        .filter(m => !m.nsfw_only || nsfwMode),
    })).filter(g => g.models.length > 0),
  [mode, families, models, enabledModels, nsfwMode])

  const known = groups.some(g => g.models.some(m => m.model_type === value))

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-text-muted uppercase tracking-wider w-11 shrink-0">
        {mode === 'image' ? 'Image' : 'Video'}
      </span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="flex-1 min-w-0 bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue"
      >
        {!known && <option value={value}>{value}</option>}
        {groups.map(({ family, models: famModels }) => (
          <optgroup key={family.id} label={family.label}>
            {famModels.map(m => (
              <option key={m.model_type} value={m.model_type}>{m.name}</option>
            ))}
          </optgroup>
        ))}
      </select>
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
  const selectDirectorImageModel = useStore(s => s.selectDirectorImageModel)
  const selectDirectorVideoModel = useStore(s => s.selectDirectorVideoModel)
  const [imageOpen, setImageOpen] = useState(false)
  const [videoOpen, setVideoOpen] = useState(false)

  return (
    <div className="space-y-1">
      {/* Models the pipeline will use — visible + changeable here so the
          user can verify before starting; LoRA lists below follow. */}
      <div className="space-y-1 mb-1.5">
        <DirectorModelPicker mode="image" value={imageModel} onChange={selectDirectorImageModel} />
        <DirectorModelPicker mode="video" value={videoModel} onChange={selectDirectorVideoModel} />
      </div>
      {/* Image LoRAs */}
      {imageModel && (
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
      {videoModel && (
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

      {/* LoRA + advanced post-processing. Short Film shows them here at the
          scene-description step; Music Video surfaces them at the first setup
          step instead (so all up-front choices live together). */}
      {isShortFilm && (
        <>
          <DirectorLoraAccordion />
          <DirectorAdvancedAccordion />
        </>
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
  planVideoPrompts: _planVideoPrompts, generateStartImages, loading, isActive, isShortFilm,
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
  loading, imageGenProgress, clipImages, planVideoPrompts: _planVideoPrompts2,
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
  clipPlans, plannedClips, clipImages, speakerMappings, editClipPlan,
  planVideoPrompts, directorGenerate, applyToClips, loading, isShortFilm,
  isGenerating, isAutoGenerating,
}: {
  clipPlans: ReturnType<typeof useStore.getState>['directorClipPlans']
  plannedClips: ReturnType<typeof useStore.getState>['directorPlannedClips']
  clipImages: ReturnType<typeof useStore.getState>['directorClipImages']
  speakerMappings: ReturnType<typeof useStore.getState>['directorSpeakerMappings']
  editClipPlan: (index: number, field: 'video_prompt' | 'image_prompt', value: string) => void
  planVideoPrompts: () => Promise<void>
  directorGenerate: () => void
  applyToClips: () => void
  loading: boolean
  isShortFilm?: boolean
  /** True when ANY generation job is currently running. The Generate
   *  button needs this so it can show a disabled "Generating..."
   *  state instead of looking pressable — important in auto mode
   *  where the system auto-triggers Generate after planning, and as
   *  a double-click guard in manual mode. */
  isGenerating?: boolean
  /** True specifically when auto-mode pipeline is the thing running.
   *  Used only to swap the button label to "Auto Generating..." so
   *  the user knows the system is driving itself, not waiting on
   *  them to click. */
  isAutoGenerating?: boolean
}) {
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

      {clipImages.length > 0 && (
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
        {/* Generate button has two render modes:
            - Idle (no jobs running): bright green CTA, click triggers
              directorGenerate. Applies in manual mode where the user
              actively kicks off generation from this review screen.
            - Generating (any job in flight): muted disabled state with
              spinner + "Generating..." label. Applies in BOTH:
                * Auto mode, where directorGenerate auto-triggers right
                  after the chat reaches review_video — without this
                  guard, the button looked pressable while generation
                  was already running, confusing the user.
                * Manual mode after the user clicked Generate — guards
                  against double-submission. */}
        {isGenerating ? (
          // Muted disabled state. opacity-60 dims the whole control
          // (including the spinner) so it reads as "not interactive"
          // even against the bright accent backgrounds Golden Hour and
          // similar themes use. Border is dropped to a subtler tone so
          // it doesn't compete with the active CTA color elsewhere on
          // screen. Label switches to "Auto Generating…" when the
          // system is driving the pipeline by itself, so the user
          // understands they're not waiting on a click.
          <button
            disabled
            className="w-full py-2.5 rounded-lg bg-bg-tertiary border border-border/40 text-text-muted text-sm font-medium flex items-center justify-center gap-1.5 cursor-not-allowed opacity-60"
          >
            <Loader2 size={14} className="animate-spin" />
            {isAutoGenerating ? 'Auto Generating...' : 'Generating...'}
          </button>
        ) : (
          <button
            onClick={directorGenerate}
            className="w-full py-2.5 rounded-lg bg-accent-green hover:bg-accent-green-hover text-white text-sm font-semibold transition-colors flex items-center justify-center gap-1.5"
          >
            <Play size={14} fill="white" /> Generate
          </button>
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
