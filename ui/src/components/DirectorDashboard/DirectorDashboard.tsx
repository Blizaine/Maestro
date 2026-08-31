import { useState, useEffect, useRef, Component, type ReactNode } from 'react'
import { X, ChevronDown, ChevronRight, Play, ImageIcon, Check, AlertTriangle, Clock, Brain, Sparkles, Loader2, Camera, Film, Combine, Pencil, Trash2 } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { getFileUrl } from '../../api/client'
import type { PipelineClipState, SavedPipelineState } from '../../types'

/** Safely coerce any value to a displayable string */
function safeStr(val: unknown): string {
  if (val == null) return ''
  if (typeof val === 'string') return val
  if (typeof val === 'object') return JSON.stringify(val)
  return String(val)
}

/** Error boundary to prevent dashboard crash from bad data */
class DashboardErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(err: Error) { return { error: err.message } }
  render() {
    if (this.state.error) {
      return (
        <div className="p-4 text-center">
          <p className="text-red-400 text-sm mb-2">Dashboard render error: {this.state.error}</p>
          <button onClick={() => this.setState({ error: null })}
            className="text-xs text-accent-blue hover:underline">Try again</button>
        </div>
      )
    }
    return this.props.children
  }
}

function formatTime(sec: number | null): string {
  if (!sec) return '--'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m}m ${s}s`
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function PipelineProgressBar({ pipeline }: { pipeline: SavedPipelineState }) {
  const phases = [
    { key: 'planning', label: 'LLM Planning', time: pipeline.llm_log?.planning_time_sec },
    { key: 'images', label: 'Image Gen', time: pipeline.clips.reduce((sum, c) => sum + (c.image_gen_time_sec || 0), 0) || null },
    { key: 'video', label: 'Video Gen', time: pipeline.clips.reduce((sum, c) => sum + (c.video_gen_time_sec || 0), 0) || null },
  ]
  const total = phases.reduce((s, p) => s + (p.time || 0), 0) || 1
  const isComplete = pipeline.status === 'completed'

  return (
    <div className="space-y-1">
      <div className="flex h-2 rounded-full overflow-hidden bg-bg-tertiary">
        {phases.map((phase, i) => {
          const pct = (phase.time || 0) / total * 100
          const colors = ['bg-purple-500', 'bg-blue-500', 'bg-green-500']
          return pct > 0 ? (
            <div key={i} className={`${colors[i]} transition-all`} style={{ width: `${Math.max(pct, 3)}%` }} />
          ) : null
        })}
      </div>
      <div className="flex justify-between text-[9px] text-text-muted">
        {phases.map((phase, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${['bg-purple-500', 'bg-blue-500', 'bg-green-500'][i]}`} />
            {phase.label}: {formatTime(phase.time ?? null)}
          </span>
        ))}
        <span className="flex items-center gap-1">
          {isComplete ? <Check size={9} className="text-indicator-success" /> : <Clock size={9} />}
          Total: {formatTime(pipeline.total_time_sec)}
        </span>
      </div>
    </div>
  )
}

function LlmPassView({ pass: p, index }: { pass: { pass: string; system_prompt: string; user_prompt?: string; response_text: string; thinking_text?: string | null }; index: number }) {
  const [showSystem, setShowSystem] = useState(false)
  const [showUser, setShowUser] = useState(false)
  const [showResponse, setShowResponse] = useState(false)
  const [showThinking, setShowThinking] = useState(false)
  const label = p.pass.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="border border-border rounded p-2 space-y-1.5">
      <div className="text-[10px] font-medium text-text-primary">Pass {index + 1}: {label}</div>

      <button onClick={() => setShowSystem(!showSystem)}
        className="flex items-center gap-1 text-[9px] text-text-secondary hover:text-text-primary w-full text-left">
        {showSystem ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
        System Prompt ({p.system_prompt?.length || 0} chars)
      </button>
      {showSystem && (
        <pre className="text-[8px] text-text-muted bg-bg-tertiary rounded p-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
          {p.system_prompt || '(empty)'}
        </pre>
      )}

      {/* User Prompt — the actual story description / screenplay sent
          alongside the system prompt. Renders only when present so old
          pipeline JSON files (captured before user_prompt was tracked)
          don't show an "(empty)" row. */}
      {p.user_prompt !== undefined && p.user_prompt !== null && (
        <>
          <button onClick={() => setShowUser(!showUser)}
            className="flex items-center gap-1 text-[9px] text-text-secondary hover:text-text-primary w-full text-left">
            {showUser ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
            User Prompt ({p.user_prompt?.length || 0} chars)
          </button>
          {showUser && (
            <pre className="text-[8px] text-text-muted bg-bg-tertiary rounded p-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
              {p.user_prompt || '(empty)'}
            </pre>
          )}
        </>
      )}

      {p.thinking_text && (
        <>
          <button onClick={() => setShowThinking(!showThinking)}
            className="flex items-center gap-1 text-[9px] text-indicator-warning hover:text-indicator-warning/80 w-full text-left">
            {showThinking ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
            <Sparkles size={8} /> Thinking ({p.thinking_text.length} chars)
          </button>
          {showThinking && (
            <pre className="text-[8px] text-text-muted bg-bg-tertiary rounded p-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
              {p.thinking_text}
            </pre>
          )}
        </>
      )}

      <button onClick={() => setShowResponse(!showResponse)}
        className="flex items-center gap-1 text-[9px] text-text-secondary hover:text-text-primary w-full text-left">
        {showResponse ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
        Response ({p.response_text?.length || 0} chars)
      </button>
      {showResponse && (
        <pre className="text-[8px] text-text-muted bg-bg-tertiary rounded p-2 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
          {p.response_text || '(empty)'}
        </pre>
      )}
    </div>
  )
}

function LlmLogPanel({ pipeline }: { pipeline: SavedPipelineState }) {
  const log = pipeline.llm_log
  if (!log) return <p className="text-xs text-text-muted italic">No LLM log captured</p>

  const passes = (log as any).passes as Array<{ pass: string; system_prompt: string; user_prompt?: string; response_text: string; thinking_text?: string | null }> | undefined

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[10px] text-text-muted">
        <Brain size={12} className="text-chip-purple" />
        <span>{log.provider}/{log.model_id || 'unknown'}</span>
        <span className="ml-1 text-text-muted">({passes?.length || 1} pass{(passes?.length || 1) > 1 ? 'es' : ''})</span>
        <span className="ml-auto">{formatTime(log.planning_time_sec)}</span>
      </div>

      {passes && passes.length > 0 ? (
        <div className="space-y-2">
          {passes.map((p, i) => (
            <LlmPassView key={i} pass={p} index={i} />
          ))}
        </div>
      ) : (
        /* Fallback: show flat log (backward compat) */
        <LlmPassView pass={{
          pass: 'planning',
          system_prompt: log.system_prompt || '',
          response_text: log.response_text || '',
          thinking_text: log.thinking_text,
        }} index={0} />
      )}
    </div>
  )
}

function ClipCard({ clip, pipeline, busy = false, onTag, onRerunImage, onRerunVideo, onSavePrompt }: {
  clip: PipelineClipState
  pipeline: SavedPipelineState
  busy?: boolean
  onTag: (tag: 'good' | 'needs_work' | null) => void
  onRerunImage: (clipIndex: number, prompt?: string) => void
  onRerunVideo: (clipIndex: number, prompt?: string) => void
  onSavePrompt: (clipIndex: number, field: 'image_prompt' | 'video_prompt', value: string) => void
}) {
  const [expandImage, setExpandImage] = useState(false)
  const [expandVideo, setExpandVideo] = useState(false)
  const [showPolish, setShowPolish] = useState(false)
  const [editingImage, setEditingImage] = useState(false)
  const [editingVideo, setEditingVideo] = useState(false)
  const [editWindowPrompts, setEditWindowPrompts] = useState<string[]>(clip.window_prompts || [])
  const [editImagePrompt, setEditImagePrompt] = useState(clip.image_prompt || '')
  const [editVideoPrompt, setEditVideoPrompt] = useState(clip.video_prompt || '')
  const requiresShotImage = !pipeline.shot_image_policy
    || pipeline.shot_image_policy === 'generate'

  // hasPolish is true if ANY of the four polish snapshots was captured.
  // Window-prompt polish only fires for ≥21s shots; keyframe polish
  // only fires when the planner emitted keyframes. video_prompt polish
  // is skipped entirely on windowed shots (its content is unused at
  // generation time), so for those we rely on the window snapshot.
  const hasPolish =
    clip.image_prompt_pre_polish != null ||
    clip.video_prompt_pre_polish != null ||
    (clip.window_prompts_pre_polish != null && clip.window_prompts_pre_polish.length > 0) ||
    (clip.keyframe_prompts_pre_polish != null && clip.keyframe_prompts_pre_polish.length > 0)
  const tagColor = clip.tag === 'good' ? 'border-green-500 bg-green-500/5'
    : clip.tag === 'needs_work' ? 'border-amber-500 bg-amber-500/5'
    : 'border-border'

  return (
    <div className={`rounded-lg border-2 ${tagColor} bg-bg-secondary overflow-hidden`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-bg-tertiary border-b border-border">
        <span className="text-xs font-medium text-text-primary">
          Shot {clip.index + 1}
          {(clip.planned_clip as unknown as Record<string, unknown> | null)?.duration_sec ? (
            <span className="text-text-muted font-normal ml-1">({Math.round((clip.planned_clip as unknown as Record<string, unknown>).duration_sec as number)}s)</span>
          ) : null}
          {clip.window_count > 1 && (
            <span className="text-chip-purple font-normal ml-1 text-[9px]">{clip.window_count}W</span>
          )}
        </span>
        <div className="flex items-center gap-1">
          {clip.image_gen_time_sec && (
            <span className="text-[9px] text-text-muted"><ImageIcon size={8} className="inline" /> {formatTime(clip.image_gen_time_sec)}</span>
          )}
          {clip.video_gen_time_sec && (
            <span className="text-[9px] text-text-muted ml-1"><Play size={8} className="inline" /> {formatTime(clip.video_gen_time_sec)}</span>
          )}
          {/* Tag buttons */}
          <button onClick={() => onTag(clip.tag === 'good' ? null : 'good')}
            disabled={busy}
            className={`ml-2 p-0.5 rounded disabled:opacity-40 ${clip.tag === 'good' ? 'bg-green-500 text-white' : 'text-text-muted hover:text-indicator-success'}`}
            title="Mark as good">
            <Check size={12} />
          </button>
          <button onClick={() => onTag(clip.tag === 'needs_work' ? null : 'needs_work')}
            disabled={busy}
            className={`p-0.5 rounded disabled:opacity-40 ${clip.tag === 'needs_work' ? 'bg-amber-500 text-white' : 'text-text-muted hover:text-indicator-warning'}`}
            title="Needs work">
            <AlertTriangle size={12} />
          </button>
        </div>
      </div>

      <div className="p-2 space-y-2">
        {/* Image section */}
        <div className="flex gap-2">
          {/* Thumbnail */}
          <div className="w-20 h-20 shrink-0 rounded overflow-hidden bg-bg-tertiary border border-border">
            {clip.start_image_filename ? (
              <img src={getFileUrl(clip.start_image_filename)} alt={`Shot ${clip.index + 1}`}
                className="w-full h-full object-cover" loading="lazy" />
            ) : clip.video_filename ? (
              <video
                src={`${getFileUrl(clip.video_filename)}#t=0.1`}
                className="w-full h-full object-cover"
                muted
                playsInline
                preload="metadata"
                aria-label={`Shot ${clip.index + 1} video preview`}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-text-muted">
                <ImageIcon size={16} />
              </div>
            )}
          </div>
          {/* Image prompt */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-[9px] text-text-muted uppercase tracking-wider">
                {requiresShotImage ? 'Image Prompt' : 'Planned Visual'}
              </span>
              {requiresShotImage && <div className="flex items-center gap-1">
                <button onClick={() => {
                  if (editingImage && editImagePrompt !== (clip.image_prompt || '')) {
                    onSavePrompt(clip.index, 'image_prompt', editImagePrompt)
                  }
                  setEditingImage(!editingImage)
                  setEditImagePrompt(clip.image_prompt || '')
                }}
                  className={`p-0.5 rounded transition-colors ${editingImage ? 'text-accent-blue' : 'text-text-muted hover:text-text-secondary'}`}
                  title="Edit prompt">
                  <Pencil size={9} />
                </button>
                <button onClick={() => onRerunImage(clip.index, editingImage ? editImagePrompt : undefined)}
                  disabled={busy}
                  className="p-0.5 rounded text-text-muted hover:text-accent-blue transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  title="Re-generate start image">
                  <Camera size={10} />
                </button>
              </div>}
            </div>
            {editingImage ? (
              <textarea
                value={editImagePrompt}
                onChange={e => setEditImagePrompt(e.target.value)}
                className="w-full bg-bg-tertiary border border-accent-blue rounded px-1.5 py-1 text-[10px] text-text-primary resize-none focus:outline-none"
                rows={3}
              />
            ) : (
              <p className={`text-[10px] text-text-secondary ${expandImage ? '' : 'line-clamp-3'} cursor-pointer`}
                onClick={() => setExpandImage(!expandImage)}>
                {clip.image_prompt || <span className="italic text-text-muted">No image prompt</span>}
              </p>
            )}
          </div>
        </div>

        {/* Keyframes */}
        {(clip.keyframe_prompts?.length > 0 || clip.keyframe_filenames?.length > 0) && (
          <div>
            <div className="text-[9px] text-text-muted uppercase tracking-wider mb-0.5">
              Keyframes ({clip.keyframe_prompts?.length || clip.keyframe_filenames?.length || 0})
            </div>
            <div className="flex gap-1.5 overflow-x-auto">
              {clip.keyframe_filenames?.map((kf, ki) => (
                <div key={ki} className="shrink-0">
                  <img src={getFileUrl(kf)} alt={`KF ${ki + 1}`}
                    className="w-14 h-14 object-cover rounded border border-border" loading="lazy" />
                  {clip.keyframe_prompts?.[ki] && (
                    <p className="text-[8px] text-text-muted mt-0.5 w-14 truncate" title={safeStr(clip.keyframe_prompts[ki])}>
                      {safeStr(clip.keyframe_prompts[ki])}
                    </p>
                  )}
                </div>
              ))}
              {/* Show prompts without images if more prompts than files */}
              {clip.keyframe_prompts?.slice(clip.keyframe_filenames?.length || 0).map((kp, ki) => (
                <div key={`p${ki}`} className="shrink-0 w-14 h-14 rounded border border-dashed border-border flex items-center justify-center">
                  <p className="text-[7px] text-text-muted p-1 line-clamp-3">{safeStr(kp)}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Video prompt */}
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-[9px] text-text-muted uppercase tracking-wider">
              Video Prompt{clip.window_prompts?.length > 1 ? ` (${clip.window_prompts.length} windows)` : ''}
            </span>
            <div className="flex items-center gap-1">
              <button onClick={() => {
                if (editingVideo) {
                  // Multi-window: join windows with newline and save to video_prompt.
                  // Single-window: save the edited video prompt directly.
                  const newValue = editWindowPrompts.length > 1
                    ? editWindowPrompts.join('\n')
                    : editVideoPrompt
                  if (newValue !== (clip.video_prompt || '')) {
                    onSavePrompt(clip.index, 'video_prompt', newValue)
                  }
                }
                setEditingVideo(!editingVideo)
                setEditVideoPrompt(clip.video_prompt || '')
                setEditWindowPrompts(clip.window_prompts || [])
              }}
                className={`p-0.5 rounded transition-colors ${editingVideo ? 'text-accent-blue' : 'text-text-muted hover:text-text-secondary'}`}
                title="Edit prompt">
                <Pencil size={9} />
              </button>
              <button onClick={() => {
                if (editingVideo && editWindowPrompts.length > 1) {
                  onRerunVideo(clip.index, editWindowPrompts.join('\n'))
                } else {
                  onRerunVideo(clip.index, editingVideo ? editVideoPrompt : undefined)
                }
              }}
                disabled={busy || (requiresShotImage && !clip.start_image_filename)}
                className="p-0.5 rounded text-text-muted hover:text-indicator-success transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                title={busy
                  ? 'Wait for pipeline repair to finish'
                  : requiresShotImage && !clip.start_image_filename
                    ? 'Generate the start image first'
                    : 'Re-generate video clip'}>
                <Film size={10} />
              </button>
            </div>
          </div>
          {editingVideo ? (
            clip.window_prompts?.length > 1 ? (
              <div className="space-y-1.5">
                {editWindowPrompts.map((wp, wi) => (
                  <div key={wi}>
                    <div className="text-[8px] text-text-muted mb-0.5">Window {wi + 1}</div>
                    <textarea
                      value={wp}
                      onChange={e => {
                        const updated = [...editWindowPrompts]
                        updated[wi] = e.target.value
                        setEditWindowPrompts(updated)
                      }}
                      className="w-full bg-bg-tertiary border border-accent-blue/50 rounded px-1.5 py-1 text-[10px] text-text-primary resize-none focus:outline-none focus:border-accent-blue"
                      rows={3}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <textarea
                value={editVideoPrompt}
                onChange={e => setEditVideoPrompt(e.target.value)}
                className="w-full bg-bg-tertiary border border-accent-blue rounded px-1.5 py-1 text-[10px] text-text-primary resize-none focus:outline-none"
                rows={4}
              />
            )
          ) : (
            clip.window_prompts?.length > 1 ? (
              <div className="space-y-0.5">
                {clip.window_prompts.map((wp, wi) => (
                  <p key={wi} className={`text-[10px] text-text-secondary pl-2 border-l-2 ${wi === 0 ? 'border-accent-blue/40' : 'border-border'} ${expandVideo ? '' : 'line-clamp-2'} cursor-pointer`}
                    onClick={() => setExpandVideo(!expandVideo)}>
                    <span className="text-[8px] text-text-muted mr-1">W{wi + 1}</span>
                    {safeStr(wp)}
                  </p>
                ))}
              </div>
            ) : (
              <p className={`text-[10px] text-text-secondary ${expandVideo ? '' : 'line-clamp-3'} cursor-pointer`}
                onClick={() => setExpandVideo(!expandVideo)}>
                {clip.video_prompt || <span className="italic text-text-muted">No video prompt</span>}
              </p>
            )
          )}
        </div>

        {/* Prompt polish diff */}
        {hasPolish && (
          <div>
            <button onClick={() => setShowPolish(!showPolish)}
              className="flex items-center gap-1 text-[9px] text-accent-blue hover:underline">
              <Sparkles size={8} />
              {showPolish ? 'Hide' : 'Show'} prompt polish diff
            </button>
            {showPolish && (() => {
              // Compute change flags so the "no changes from polish"
              // message only shows when literally nothing was modified
              // by Pass 3 (across image, video, all windows, all keyframes).
              const imageChanged = !!(clip.image_prompt_pre_polish && clip.image_prompt_pre_polish !== clip.image_prompt)
              const videoChanged = !!(clip.video_prompt_pre_polish && clip.video_prompt_pre_polish !== clip.video_prompt)
              const wpsPre = clip.window_prompts_pre_polish || []
              const wpsPost = clip.window_prompts || []
              const windowDiffs = wpsPre
                .map((pre, i) => ({ pre, post: wpsPost[i] || '', i }))
                .filter(d => d.pre && d.post && d.pre !== d.post)
              const kfsPre = clip.keyframe_prompts_pre_polish || []
              const kfsPost = clip.keyframe_prompts || []
              const keyframeDiffs = kfsPre
                .map((pre, i) => ({ pre, post: kfsPost[i] || '', i }))
                .filter(d => d.pre && d.post && d.pre !== d.post)
              const anyChange = imageChanged || videoChanged || windowDiffs.length > 0 || keyframeDiffs.length > 0
              return (
                <div className="mt-1 space-y-1.5 bg-bg-tertiary rounded p-2">
                  {imageChanged && (
                    <div>
                      <div className="text-[8px] text-text-muted uppercase">Image — Before Polish</div>
                      <p className="text-[9px] text-red-400/70 line-through">{clip.image_prompt_pre_polish}</p>
                      <div className="text-[8px] text-text-muted uppercase mt-0.5">After</div>
                      <p className="text-[9px] text-indicator-success/80">{clip.image_prompt}</p>
                    </div>
                  )}
                  {videoChanged && (
                    <div>
                      <div className="text-[8px] text-text-muted uppercase">Video — Before Polish</div>
                      <p className="text-[9px] text-red-400/70 line-through">{clip.video_prompt_pre_polish}</p>
                      <div className="text-[8px] text-text-muted uppercase mt-0.5">After</div>
                      <p className="text-[9px] text-indicator-success/80">{clip.video_prompt}</p>
                    </div>
                  )}
                  {windowDiffs.map(({ pre, post, i }) => (
                    <div key={`wp${i}`}>
                      <div className="text-[8px] text-text-muted uppercase">Window {i + 1} — Before Polish</div>
                      <p className="text-[9px] text-red-400/70 line-through">{pre}</p>
                      <div className="text-[8px] text-text-muted uppercase mt-0.5">After</div>
                      <p className="text-[9px] text-indicator-success/80">{post}</p>
                    </div>
                  ))}
                  {keyframeDiffs.map(({ pre, post, i }) => (
                    <div key={`kf${i}`}>
                      <div className="text-[8px] text-text-muted uppercase">Keyframe {i + 1} — Before Polish</div>
                      <p className="text-[9px] text-red-400/70 line-through">{pre}</p>
                      <div className="text-[8px] text-text-muted uppercase mt-0.5">After</div>
                      <p className="text-[9px] text-indicator-success/80">{post}</p>
                    </div>
                  ))}
                  {!anyChange && (
                    <p className="text-[9px] text-text-muted italic">No changes from polish</p>
                  )}
                </div>
              )
            })()}
          </div>
        )}
      </div>
    </div>
  )
}

export function DirectorDashboard() {
  const open = useStore(s => s.dashboardOpen)

  if (!open) return null

  return (
    <DashboardErrorBoundary>
      <DirectorDashboardInner />
    </DashboardErrorBoundary>
  )
}

function DirectorDashboardInner() {
  const setOpen = useStore(s => s.setDashboardOpen)
  const pipelineList = useStore(s => s.dashboardPipelineList)
  const selectedPipeline = useStore(s => s.dashboardSelectedPipeline)
  const loading = useStore(s => s.dashboardLoading)
  const loadPipeline = useStore(s => s.loadSavedPipeline)
  const tagClip = useStore(s => s.tagClip)
  const updateClipPrompt = useStore(s => s.updateClipPrompt)
  const startPipelineRepair = useStore(s => s.startPipelineRepair)
  const cancelPipelineRepair = useStore(s => s.cancelPipelineRepair)
  const rerunClipImage = useStore(s => s.rerunClipImage)
  const rerunClipVideo = useStore(s => s.rerunClipVideo)
  const rejoinClips = useStore(s => s.rejoinPipelineClips)
  const resumePipeline = useStore(s => s.resumePipeline)
  const deletePipeline = useStore(s => s.deletePipeline)
  const loadDirectorFromPipeline = useStore(s => s.loadDirectorFromPipeline)
  const [resuming, setResuming] = useState(false)
  // Keyed by pipeline id — a bare boolean would let "arm on pipeline A,
  // switch to B, click once" delete B without a confirm.
  const [confirmDeletePid, setConfirmDeletePid] = useState<string | null>(null)
  const [deletingPipeline, setDeletingPipeline] = useState(false)
  const [repairStartingPid, setRepairStartingPid] = useState<string | null>(null)
  const [repairCancellingPid, setRepairCancellingPid] = useState<string | null>(null)
  const [regenErrors, setRegenErrors] = useState<Record<string, string>>({})
  const autoLoadAttemptedPid = useRef<string | null>(null)
  const selectedPid = selectedPipeline?.pipeline_id || null
  const repairStarting = repairStartingPid === selectedPid
  const repairCancelling = repairCancellingPid === selectedPid
  const regenError = selectedPid ? regenErrors[selectedPid] || null : null
  const setRegenError = (message: string | null) => {
    const pid = selectedPid
    if (!pid) return
    setRegenErrors(current => {
      const next = { ...current }
      if (message) next[pid] = message
      else delete next[pid]
      return next
    })
  }

  // Auto-load first pipeline when list loads
  useEffect(() => {
    if (selectedPipeline) {
      autoLoadAttemptedPid.current = null
      return
    }
    if (pipelineList.length > 0 && !selectedPipeline && !loading) {
      const active = pipelineList.find(p =>
        p.repair_status === 'queued'
        || p.repair_status === 'running'
        || p.repair_status === 'cancelling')
      const pid = (active || pipelineList[0]).id
      // A stale list entry (for example, a pipeline removed outside Maestro)
      // must not create an endless load/fail/render loop. Explicit selection
      // and closing/reopening the Dashboard still provide retry paths.
      if (autoLoadAttemptedPid.current === pid) return
      autoLoadAttemptedPid.current = pid
      void loadPipeline(pid)
    }
  }, [pipelineList, selectedPipeline, loading, loadPipeline])

  const goodCount = selectedPipeline?.clips.filter(c => c.tag === 'good').length || 0
  const needsWorkCount = selectedPipeline?.clips.filter(c => c.tag === 'needs_work').length || 0
  const totalClips = selectedPipeline?.clips.length || 0
  // Legacy projects predate this field and retain the original required-image
  // contract. New H3 prompt/direct-reference projects intentionally have no
  // image artifacts, so repair must judge only their videos.
  const requiresShotImages = !selectedPipeline?.shot_image_policy
    || selectedPipeline.shot_image_policy === 'generate'
  const missingImages = requiresShotImages
    ? selectedPipeline?.clips.filter(c => !c.start_image_filename).length || 0
    : 0
  // A video beside a missing/replaced start image is stale even if its file
  // still exists: it was generated from different visual conditioning.
  const missingVideos = selectedPipeline?.clips.filter(c =>
    !c.video_filename
    || c.video_stale
    || (requiresShotImages && !c.start_image_filename)).length || 0
  const incompleteClips = selectedPipeline?.clips.filter(c =>
    (requiresShotImages && !c.start_image_filename)
    || !c.video_filename
    || c.video_stale).length || 0
  const hasMissing = missingImages > 0 || missingVideos > 0
  const repair = selectedPipeline?.repair
  const repairActive = repair?.status === 'queued'
    || repair?.status === 'running'
    || repair?.status === 'cancelling'
  const repairRetryable = repair?.status === 'failed'
    || repair?.status === 'cancelled'
    || repair?.status === 'interrupted'
  const repairBusy = repairActive || repairStarting || repairCancelling
  const pipelineTerminal = !!selectedPipeline && [
    'completed', 'failed', 'crashed', 'cancelled',
  ].includes(selectedPipeline.status)
  const showRepairAction = hasMissing || repairActive || repairRetryable || pipelineTerminal

  const generateMissing = async () => {
    if (!selectedPipeline) return
    const pid = selectedPipeline.pipeline_id
    setRegenError(null)
    setRepairStartingPid(pid)
    try {
      await startPipelineRepair(pid)
    } catch (e) {
      setRegenError(String(e instanceof Error ? e.message : e))
    } finally {
      setRepairStartingPid(current => current === pid ? null : current)
    }
  }

  return (
    <div className="fixed inset-0 z-[60] flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-center gap-2 shrink-0">
        <h1 className="text-sm font-semibold text-text-primary shrink-0">Dashboard</h1>

        {/* Pipeline selector */}
        <select
          value={selectedPipeline?.pipeline_id || ''}
          onChange={e => { if (e.target.value) loadPipeline(e.target.value) }}
          className="flex-1 min-w-0 max-w-md bg-bg-tertiary border border-border rounded-lg px-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue truncate"
        >
          <option value="">Select pipeline...</option>
          {pipelineList.map(p => (
            <option key={p.id} value={p.id}>
              {p.repair_status ? `[repair: ${p.repair_status}] ` : ''}
              {formatDate(p.created_at)} — {p.pipeline_type} ({p.clip_count} clips) [{p.status}]
              {p.scene_description ? ` — ${p.scene_description}` : ''}
            </option>
          ))}
        </select>

        {/* Summary badges */}
        {selectedPipeline && (
          <div className="flex items-center gap-2 text-[10px] shrink-0">
            <span className="flex items-center gap-0.5 text-indicator-success">
              <Check size={10} /> {goodCount}
            </span>
            <span className="flex items-center gap-0.5 text-indicator-warning">
              <AlertTriangle size={10} /> {needsWorkCount}
            </span>
            <span className="text-text-muted">
              / {totalClips} clips
            </span>
            {(selectedPipeline.status === 'crashed' || selectedPipeline.status === 'failed') && (
              <button
                onClick={async () => {
                  if (!selectedPipeline) return
                  setResuming(true); setRegenError(null)
                  try {
                    await resumePipeline(selectedPipeline.pipeline_id)
                  } catch (e) {
                    setRegenError(String(e instanceof Error ? e.message : e))
                  } finally {
                    setResuming(false)
                  }
                }}
                disabled={resuming || loading || repairBusy}
                className="flex items-center gap-1 px-2 py-1 text-[10px] bg-green-500/10 border border-green-500/30 rounded text-indicator-success hover:bg-green-500/20 disabled:opacity-40 transition-colors"
                title="Re-run this pipeline from where it crashed — reuses the planning and start images that already completed"
              >
                <Play size={10} />
                {resuming ? 'Resuming…' : 'Resume'}
              </button>
            )}
            {repairActive ? (
              <>
                <span
                  className="flex items-center gap-1 px-2 py-1 text-[10px] bg-orange-500/10 border border-orange-500/30 rounded text-chip-orange"
                  title={repair?.message || 'Repair running'}
                >
                  <Loader2 size={10} className="animate-spin" />
                  {repair?.message || 'Repairing'}
                  {repair && repair.total > 0 ? ` (${repair.current}/${repair.total})` : ''}
                </span>
                <button
                  onClick={async () => {
                    if (!selectedPipeline) return
                    const pid = selectedPipeline.pipeline_id
                    setRepairCancellingPid(pid); setRegenError(null)
                    try {
                      await cancelPipelineRepair(pid)
                    } catch (e) {
                      setRegenError(String(e instanceof Error ? e.message : e))
                    } finally {
                      setRepairCancellingPid(current => current === pid ? null : current)
                    }
                  }}
                  disabled={repairCancelling || repair?.status === 'cancelling'}
                  className="flex items-center gap-1 px-2 py-1 text-[10px] bg-red-500/10 border border-red-500/30 rounded text-red-400 hover:bg-red-500/20 disabled:opacity-40 transition-colors"
                  title="Stop after aborting the current model step"
                >
                  {repairCancelling ? <Loader2 size={10} className="animate-spin" /> : <X size={10} />}
                  {repair?.status === 'cancelling' ? 'Cancelling...' : 'Stop'}
                </button>
              </>
            ) : showRepairAction ? (
              <button
                onClick={generateMissing}
                disabled={loading || repairBusy}
                className="flex items-center gap-1 px-2 py-1 text-[10px] bg-orange-500/10 border border-orange-500/30 rounded text-chip-orange hover:bg-orange-500/20 disabled:opacity-40 transition-colors"
                title={hasMissing
                  ? requiresShotImages
                    ? `Repair ${missingImages} missing images + ${missingVideos} missing or stale videos, then join when possible`
                    : `Repair ${missingVideos} missing or stale videos, then join when possible`
                  : 'Check saved clip files and repair anything missing or invalid, then join when possible'}
              >
                {repairStarting ? <Loader2 size={10} className="animate-spin" /> : <Play size={10} />}
                {repairRetryable && !hasMissing
                  ? 'Retry repair'
                  : missingImages > 0
                    ? `Repair ${incompleteClips} clip${incompleteClips === 1 ? '' : 's'}`
                    : missingVideos > 0
                      ? `Repair ${missingVideos} video${missingVideos === 1 ? '' : 's'}`
                      : 'Check + repair'}
              </button>
            ) : null}
            {repair?.status === 'completed' && (
              repair.result_filename ? (
                <a
                  href={getFileUrl(repair.result_filename)}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 px-2 py-1 text-[10px] bg-green-500/10 border border-green-500/30 rounded text-indicator-success hover:bg-green-500/20 transition-colors"
                  title="Open the repaired joined video"
                >
                  <Check size={10} /> Repaired + joined
                </a>
              ) : (
                <span className="flex items-center gap-1 px-2 py-1 text-[10px] text-indicator-success">
                  <Check size={10} /> Repair complete
                </span>
              )
            )}
            <button
              onClick={() => {
                if (selectedPipeline) {
                  void loadDirectorFromPipeline(selectedPipeline.pipeline_id)
                }
              }}
              disabled={loading || repairBusy}
              className="flex items-center gap-1 px-2 py-1 text-[10px] bg-accent-blue/10 border border-accent-blue/30 rounded text-accent-blue hover:bg-accent-blue/20 disabled:opacity-40 transition-colors"
              title="Restore this project’s script, prompts, references, models, LoRAs, and Director controls as an editable new revision"
            >
              <Pencil size={10} />
              Open &amp; Edit
            </button>
            <button
              onClick={() => {
                if (!selectedPipeline) return
                setRegenError(null)
                // Surface failures in the existing error slot — a bare
                // rejected promise here looked like the button doing nothing.
                rejoinClips(selectedPipeline.pipeline_id).catch(e =>
                  setRegenError(e instanceof Error ? e.message : 'Rejoin failed'))
              }}
              disabled={loading || repairBusy || totalClips < 2}
              className="flex items-center gap-1 px-2 py-1 text-[10px] bg-accent-blue/10 border border-accent-blue/30 rounded text-accent-blue hover:bg-accent-blue/20 disabled:opacity-40 transition-colors"
              title="Re-join all clips into a new video"
            >
              <Combine size={10} />
              Re-join
            </button>
            <button
              onClick={async () => {
                if (!selectedPipeline) return
                const pid = selectedPipeline.pipeline_id
                if (confirmDeletePid !== pid) {
                  setConfirmDeletePid(pid)
                  setTimeout(() => setConfirmDeletePid(c => (c === pid ? null : c)), 4000)
                  return
                }
                setConfirmDeletePid(null)
                setDeletingPipeline(true)
                setRegenError(null)
                try {
                  await deletePipeline(pid)
                } catch (e) {
                  setRegenError(e instanceof Error ? e.message : 'Delete failed')
                } finally {
                  setDeletingPipeline(false)
                }
              }}
              disabled={loading || deletingPipeline || repairBusy}
              className={`flex items-center gap-1 px-2 py-1 text-[10px] border rounded transition-colors disabled:opacity-40 ${
                confirmDeletePid === selectedPipeline.pipeline_id
                  ? 'bg-red-500/20 border-red-500/50 text-red-400'
                  : 'bg-red-500/10 border-red-500/30 text-red-400/80 hover:bg-red-500/20'
              }`}
              title={confirmDeletePid === selectedPipeline.pipeline_id
                ? `Click again to permanently delete this pipeline and its ${selectedPipeline.output_files?.length ?? 0} media files (they disappear from the gallery too)`
                : 'Delete this pipeline and ALL media it generated'}
            >
              {deletingPipeline ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
              {confirmDeletePid === selectedPipeline.pipeline_id ? 'Confirm?' : 'Delete'}
            </button>
            {(regenError || (repairRetryable ? repair?.error || repair?.message : null)) && (
              <span className="text-[9px] text-red-400 max-w-[200px] truncate" title={regenError || repair?.error || repair?.message || undefined}>
                {regenError || repair?.error || repair?.message}
              </span>
            )}
          </div>
        )}

        <button onClick={() => setOpen(false)}
          className="fixed top-3 right-4 z-[61] p-1.5 rounded-lg bg-bg-secondary hover:bg-bg-hover transition-colors shadow-md border border-border">
          <X size={16} className="text-text-muted" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {loading && (
          <div className="flex items-center justify-center py-12 text-text-muted">
            <Loader2 size={20} className="animate-spin mr-2" />
            Loading pipeline...
          </div>
        )}

        {!loading && !selectedPipeline && pipelineList.length === 0 && (
          <div className="text-center py-12 text-text-muted">
            <p className="text-sm">No saved pipelines yet</p>
            <p className="text-xs mt-1">Run a Director pipeline and it will appear here</p>
          </div>
        )}

        {selectedPipeline && (
          <>
            {/* Pipeline info */}
            <div className="bg-bg-secondary rounded-lg border border-border p-3 space-y-2">
              <div className="flex items-center gap-2 text-xs">
                <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                  selectedPipeline.status === 'completed' ? 'bg-green-500/20 text-indicator-success' :
                  selectedPipeline.status === 'failed' || selectedPipeline.status === 'crashed' ? 'bg-red-500/20 text-chip-red' :
                  'bg-blue-500/20 text-chip-blue'
                }`}>
                  {selectedPipeline.status === 'crashed' ? 'crashed (process died)' : selectedPipeline.status}
                </span>
                <span className="text-text-muted">{selectedPipeline.pipeline_type}</span>
                <span className="text-text-muted">|</span>
                <span className="text-text-muted">{selectedPipeline.image_model}</span>
                <span className="text-text-muted">+</span>
                <span className="text-text-muted">{selectedPipeline.video_model}</span>
              </div>
              {selectedPipeline.scene_description && (
                <p className="text-[11px] text-text-secondary">{selectedPipeline.scene_description}</p>
              )}
              <PipelineProgressBar pipeline={selectedPipeline} />
            </div>

            {/* LLM Log */}
            <div className="bg-bg-secondary rounded-lg border border-border p-3">
              <h3 className="text-[11px] text-text-secondary uppercase tracking-wider font-medium mb-2">LLM Planning Log</h3>
              <LlmLogPanel pipeline={selectedPipeline} />
            </div>

            {/* Clip Grid */}
            <div>
              <h3 className="text-[11px] text-text-secondary uppercase tracking-wider font-medium mb-2">
                Clips ({totalClips})
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {selectedPipeline.clips.map(clip => (
                  <ClipCard
                    key={clip.index}
                    clip={clip}
                    pipeline={selectedPipeline}
                    busy={repairBusy}
                    onTag={(tag) => tagClip(selectedPipeline.pipeline_id, clip.index, tag)}
                    onSavePrompt={(idx, field, value) => updateClipPrompt(selectedPipeline.pipeline_id, idx, field, value)}
                    onRerunImage={(idx, prompt) => { setRegenError(null); rerunClipImage(selectedPipeline.pipeline_id, idx, prompt).catch(e => setRegenError(String(e instanceof Error ? e.message : e))) }}
                    onRerunVideo={(idx, prompt) => { setRegenError(null); rerunClipVideo(selectedPipeline.pipeline_id, idx, prompt).catch(e => setRegenError(String(e instanceof Error ? e.message : e))) }}
                  />
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
