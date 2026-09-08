import { useEffect, useRef, useState } from 'react'
import { ImagePlus, Loader2, Video, WandSparkles, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { CharacterImagePickerButton } from '../Characters/CharacterImagePicker'
import { CharacterToolbarItem } from './SidebarPanels'
import { newViggleCharacter, vigglePreparationKey, viggleTimeline, VIGGLE_SWAP_PROMPT } from '../../lib/viggle'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import { characterDisplayName } from '../../lib/characters'
import type { GenerateParams, ViggleCharacterOptions } from '../../types'

const filename = (path: string) => path.replace(/\\/g, '/').split('/').pop() || path
const mediaUrl = (path: string) => /[/\\]uploads[/\\]/.test(path)
  ? api.getUploadUrl(filename(path)) : api.getFileUrl(filename(path))

export function ViggleControls() {
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const editFrame = useStore(s => s.sendFrameToImageMode)
  const generationError = useStore(s => s.promptEnhanceError)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [preparing, setPreparing] = useState(false)
  const [progress, setProgress] = useState('')
  const [previewJob, setPreviewJob] = useState('')
  const alive = useRef(true)
  useEffect(() => {alive.current = true; return () => {alive.current = false}}, [])
  const source = String(params.video_guide || '')
  const edited = String(params._viggle_edited_frame || '')
  const audio = String(params.audio_prompt_type || '')
  const character = params.viggle_character
  const timeline = viggleTimeline(params)
  const locked = busy || preparing
  const changeTimeline = (changes: Partial<GenerateParams>) => {
    useStore.setState(state => {
      if (state.params.video_guide !== source) return state
      const next = {...state.params, ...changes}
      // A frame is always an absolute timestamp in the original source.
      const selection = viggleTimeline(next)
      const frameChanged = selection.frame !== viggleTimeline(state.params).frame
      return {params: {...next, _viggle_trim_start: selection.start, _viggle_trim_end: selection.end,
        _viggle_frame_seconds: selection.frame,
        ...(next.viggle_character ? {viggle_character: {...next.viggle_character, frame_seconds: selection.frame},
          ...(frameChanged ? {_viggle_prepared: undefined, _viggle_edited_frame: undefined} : {})} : {})}}
    })
    const current = useStore.getState()
    const range = viggleTimeline(current.params)
    if (current.params.video_guide === source && range.length > 0
      && (current.params._duration_planning_mode ?? 'auto') === 'auto') current.setDurationSeconds(range.length)
  }
  const changeFrame = (seconds: number) => changeTimeline({
    _viggle_frame_seconds: seconds,
    ...(character ? {viggle_character: {...character, frame_seconds: seconds}} : {}),
  })
  const clearPrepared = () => {
    setParam('_viggle_prepared', undefined)
    setParam('_viggle_edited_frame', undefined)
  }
  const changeCharacter = (changes: Partial<ViggleCharacterOptions>) => {
    setParam('viggle_character', {...newViggleCharacter(), frame_seconds: timeline.frame, ...character, ...changes})
    clearPrepared()
  }
  const upload = async (file: File | undefined, kind: 'video' | 'image' | 'audio' | 'character') => {
    if (!file || locked) return
    setBusy(true); setError('')
    try {
      const result = await api.uploadImage(file)
      if (kind === 'video') {
        setParam('video_guide', result.path)
      } else if (kind === 'image') {
        setParam('_viggle_edited_frame', result.path)
        setParam('viggle_character', undefined)
        setParam('_viggle_prepared', undefined)
      } else if (kind === 'character') changeCharacter({reference_path: result.path,
        reference_url: result.url, character_name: file.name, character_id: undefined, view_id: undefined})
      else setParam('audio_guide', result.path)
    } catch (e) { setError(e instanceof Error ? e.message : 'Upload failed') }
    finally { setBusy(false) }
  }
  const input = (kind: 'video' | 'image' | 'audio' | 'character', label: string) => (
    <label onDragOver={event => event.preventDefault()} onDrop={event => {
      event.preventDefault()
      const file = event.dataTransfer.files[0]
      if (file && file.type.startsWith(`${kind === 'character' ? 'image' : kind}/`)) void upload(file, kind)
    }} className={`flex min-h-24 cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-bg-tertiary/30 px-3 py-3 text-xs text-text-secondary hover:border-accent-blue ${locked ? 'pointer-events-none opacity-50' : ''}`}>
      {kind === 'video' ? <Video size={17} /> : kind !== 'audio' ? <ImagePlus size={17} /> : null}
      <span>{label}</span>
      <input className="sr-only" type="file" accept={`${kind === 'character' ? 'image' : kind}/*`} disabled={locked}
        aria-label={label} onChange={event => {
          void upload(event.target.files?.[0], kind)
          event.currentTarget.value = ''
        }} />
    </label>
  )
  const preparePreview = async () => {
    if (!source || !character?.reference_path || locked) return
    const key = vigglePreparationKey(source, character, params.seed)
    setPreparing(true); setProgress('Queuing character preparation…'); setError('')
    try {
      // Submit a normal, cancellable job. It keeps running if the browser closes.
      const submitted = await api.submitGeneration({model_type: 'viggle_animate', prompt: 'Viggle Animate',
        generation_mode: 'video', image_mode: 0, resolution: params.resolution, video_length: 124,
        video_guide: source, seed: params.seed, viggle_character: character,
        _viggle_trim_start: params._viggle_trim_start, _viggle_trim_end: params._viggle_trim_end,
        _viggle_source_seconds: params._viggle_source_seconds, _viggle_frame_seconds: timeline.frame,
        _viggle_prepared: params._viggle_prepared, _viggle_prepare_only: true,
        audio_prompt_type: '', workspace: useStore.getState().activeWorkspace})
      if (alive.current) setPreviewJob(submitted.job_id)
      while (true) {
        const status = await api.fetchJobStatus(submitted.job_id)
        if (status.status === 'failed') throw new Error(status.error || 'Character preparation failed.')
        if (status.status === 'cancelled') throw new Error('Character preparation cancelled.')
        if (status.status === 'completed') {
          if (!status.viggle_preparation) throw new Error('No prepared frame was returned.')
          const current = useStore.getState().params
          if (current.model_type === 'viggle_animate' && key === vigglePreparationKey(current.video_guide, current.viggle_character, current.seed)) {
            setParam('_viggle_prepared', status.viggle_preparation)
            setParam('_viggle_edited_frame', status.viggle_preparation.image_path)
          }
          break
        }
        if (alive.current) setProgress(`${status.message || status.phase || 'Preparing character…'}${status.total_steps ? ` · ${status.step}/${status.total_steps}` : ''}`)
        // Unmounting stops UI polling; the managed job remains cancellable in Queue.
        if (!alive.current) break
        await new Promise(resolve => setTimeout(resolve, 1000))
      }
    } catch (e) {if (alive.current) setError(e instanceof Error ? e.message : 'Could not prepare the character.')}
    finally {if (alive.current) {setPreparing(false); setPreviewJob(''); setProgress('')}}
  }
  return (
    <div className="space-y-3 min-w-0" aria-label="Viggle Animate inputs">
      <CharacterToolbarItem><CharacterImagePickerButton maxImages={1} disabled={locked} label="Characters"
        onSelect={async (selected, images) => {
          const file = await api.characterImageFile(selected, images[0])
          const uploaded = await api.uploadImage(file)
          changeCharacter({reference_path: uploaded.path, reference_url: uploaded.url,
            character_id: selected.id, character_name: selected.name, view_id: images[0].id})
        }}/></CharacterToolbarItem>
      <p className="text-[11px] leading-relaxed text-text-secondary">Replace a person or object in one frame, then animate that edit through the source video.</p>
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-text-primary"><span>1 · Control video</span>
          {source && <button type="button" disabled={locked} aria-label="Remove control video" onClick={() => {
            setParam('video_guide', undefined)
          }} className="p-2 text-text-muted"><X size={14} /></button>}
        </div>
        {source ? <>
          <VideoTimelineSelector key={source} videoUrl={mediaUrl(source)} duration={timeline.duration}
            startTime={timeline.start} endTime={timeline.end} playheadTime={timeline.frame} disabled={locked}
            onStartChange={seconds => changeTimeline({_viggle_trim_start: seconds})}
            onEndChange={seconds => changeTimeline({_viggle_trim_end: seconds})}
            onPlayheadChange={changeFrame} onMetadata={seconds => {
              if (Number.isFinite(seconds) && seconds > 0) changeTimeline({_viggle_source_seconds: seconds})
            }}/>
          <div className="grid grid-cols-3 gap-2">
            {([
              ['Start', 'Trim start seconds', timeline.start, 0, Math.max(0, timeline.end - 0.1),
                (seconds: number) => changeTimeline({_viggle_trim_start: seconds})],
              ['End', 'Trim end seconds', timeline.end, timeline.start + 0.1, timeline.duration,
                (seconds: number) => changeTimeline({_viggle_trim_end: seconds})],
              ['Frame to edit', 'Source frame seconds', timeline.frame, timeline.start, timeline.lastFrame, changeFrame],
            ] as const).map(([label, ariaLabel, value, min, max, change]) => <label key={label}
              className="min-w-0 text-[10px] text-text-muted">{label}
              <input type="number" aria-label={ariaLabel} min={min} max={max} step={0.01}
                disabled={locked || !timeline.duration} value={Math.round(value * 100) / 100}
                onChange={event => change(Number(event.target.value) || 0)}
                className="mt-1 min-h-10 w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-2 text-base sm:text-xs text-text-primary"/>
            </label>)}
          </div>
          <p className="text-[10px] leading-relaxed text-text-muted">Drag the end handles to trim. Move the white playhead to choose the frame to edit. Times refer to the original video.</p>
        </> : input('video', 'Upload control video')}
      </div>
      <div className="space-y-2">
        <div className="text-xs text-text-primary">2 · Character replacement</div>
        <div className="grid grid-cols-2 gap-1 rounded-xl bg-bg-tertiary p-1">
          {['Use a character', 'Use an edited frame'].map((label, index) => <button key={label} type="button" disabled={locked}
            aria-pressed={Boolean(character) === (index === 0)} onClick={() => {
              if (Boolean(character) === (index === 0)) return
              setParam('_viggle_frame_seconds', timeline.frame)
              setParam('viggle_character', index === 0 ? {...newViggleCharacter(), frame_seconds: timeline.frame} : undefined)
              clearPrepared(); setError('')
            }} className={`min-h-10 rounded-lg px-2 text-xs ${Boolean(character) === (index === 0) ? 'bg-bg-primary text-text-primary shadow-sm' : 'text-text-muted'}`}>{label}</button>)}
        </div>
        {character ? <>
          {character.reference_path && <div className="flex items-center gap-3 rounded-xl border border-border p-2">
            <img src={character.reference_url || mediaUrl(character.reference_path)} alt="Character reference" className="h-20 w-20 shrink-0 rounded-lg object-contain bg-black"/>
            <span className="min-w-0 flex-1 break-words text-xs text-text-primary">{character.character_name ? characterDisplayName(character.character_name) : 'Character image'}</span>
          </div>}
          <div>
            {input('character', 'Upload image')}
          </div>
          <label htmlFor="viggle-appearance" className="block text-[11px] text-text-muted">Appearance (optional)</label>
          <textarea id="viggle-appearance" disabled={locked} value={character.appearance_prompt} rows={3} maxLength={8000}
            onChange={event => changeCharacter({appearance_prompt: event.target.value})}
            placeholder="For example: Blaine wearing a dark leather jacket and blue jeans."
            className="w-full rounded-xl border border-border bg-bg-tertiary p-3 text-base sm:text-xs text-text-primary placeholder:text-text-muted"/>
          <details className="rounded-xl border border-border p-3 text-xs text-text-secondary">
            <summary className="cursor-pointer">Preparation settings</summary>
            <label className="mt-3 block text-[11px]">Image model
              <select aria-label="Preparation image model" disabled={locked} value={character.image_model}
                onChange={event => changeCharacter({image_model: event.target.value as ViggleCharacterOptions['image_model']})}
                className="mt-1 min-h-10 w-full rounded-lg border border-border bg-bg-primary px-2 text-text-primary">
                <option value="flux2_klein_9b">Flux 2 Klein 9B</option><option value="flux2_klein_4b">Flux 2 Klein 4B</option>
              </select>
            </label>
            <label className="mt-3 block text-[11px]">Replacement instructions
              <textarea aria-label="Replacement instructions" disabled={locked} rows={5} maxLength={8000}
                value={character.swap_prompt} onChange={event => changeCharacter({swap_prompt: event.target.value})}
                className="mt-1 w-full rounded-lg border border-border bg-bg-primary p-2 text-base sm:text-xs text-text-primary"/>
            </label>
            <button type="button" disabled={locked} onClick={() => changeCharacter({swap_prompt: VIGGLE_SWAP_PROMPT})} className="mt-2 text-accent-blue">Reset instructions</button>
          </details>
          <button type="button" disabled={!source || !character.reference_path || locked} onClick={() => void preparePreview()}
            className="flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-border bg-bg-tertiary px-3 text-xs text-text-primary disabled:opacity-40">
            <WandSparkles size={15}/>Preview replacement frame
          </button>
          <p className="text-[10px] leading-relaxed text-text-muted">Generate prepares the frame with Klein, then animates it with Viggle. Preview is optional. Appearance instructions apply to the replacement image.</p>
          {params._viggle_prepared && <figure className="space-y-1">
            <img src={params._viggle_prepared.image_url} alt="Prepared character frame" className="max-h-56 w-full rounded-xl object-contain bg-black"/>
            <figcaption className="text-[10px] text-text-muted">Ready to animate · {params._viggle_prepared.width} × {params._viggle_prepared.height}
              <a href={params._viggle_prepared.image_url} download className="ml-2 text-accent-blue">Download frame</a>
            </figcaption>
          </figure>}
        </> : <>
        <div className="flex items-center justify-between text-xs text-text-primary"><span>Edited frame</span>
          {edited && <button type="button" aria-label="Remove edited frame" onClick={() => setParam('_viggle_edited_frame', undefined)} className="p-2 text-text-muted"><X size={14} /></button>}
        </div>
        {edited && <img src={mediaUrl(edited)} alt="Edited reference frame" className="max-h-48 w-full rounded-xl object-contain bg-black" />}
        {input('image', edited ? 'Replace edited frame' : 'Upload edited frame')}
        <button type="button" disabled={!source || locked} onClick={async () => {
          setBusy(true); setError('')
          try {
            await editFrame('animate')
          } catch (e) { setError(e instanceof Error ? e.message : 'Could not open the frame editor') }
          finally { setBusy(false) }
        }} className="flex w-full min-h-10 items-center justify-center gap-2 rounded-xl border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary disabled:opacity-40 hover:border-accent-blue">
          <WandSparkles size={15} />Edit current video frame in Maestro
        </button>
        <p className="text-[10px] leading-relaxed text-text-muted">Select a clear frame with the white playhead. Change the subject, keeping its pose, background, framing and image dimensions.</p>
        </>}
      </div>
      <div className="space-y-2">
        <label htmlFor="viggle-audio" className="text-[11px] text-text-muted">Audio</label>
        <select id="viggle-audio" value={['', 'K', 'A'].includes(audio) ? audio : ''}
          onChange={event => setParam('audio_prompt_type', event.target.value)}
          className="min-h-10 w-full rounded-xl border border-border bg-bg-tertiary px-3 text-xs text-text-primary">
          <option value="">No input audio</option><option value="K">Use control video audio</option><option value="A">Use custom audio</option>
        </select>
        {audio === 'A' && input('audio', params.audio_guide ? filename(params.audio_guide) : 'Upload custom audio')}
        <p className="text-[10px] text-text-muted">{audio === ''
          ? 'The default recipe requests silence. Choose the control soundtrack or custom audio to retain sound.'
          : 'Audio guidance is experimental. Use a track synchronized to the original source; the same trim is applied to its audio.'}</p>
      </div>
      <div className="rounded-xl border border-border bg-bg-tertiary/40 p-2.5 text-[10px] leading-relaxed text-text-muted">3 steps · 5.2s windows · 24 fps. Longer footage continues automatically. The edited image supplies the instructions; this model uses a fixed prompt.</div>
      {preparing && <div className="space-y-2 rounded-xl border border-accent-blue/30 p-3">
        <p className="flex items-center gap-2 text-xs text-text-secondary" role="status"><Loader2 size={15} className="shrink-0 animate-spin"/>{progress}</p>
        {previewJob && <button type="button" onClick={() => void api.cancelJob(previewJob).catch(e => setError(String(e)))} className="text-xs text-text-muted">Cancel preparation</button>}
      </div>}
      {busy && <p className="text-xs text-text-muted" role="status">Preparing media…</p>}
      {(error || generationError) && <p className="text-xs text-red-400" role="alert">{error || generationError}</p>}
    </div>
  )
}
