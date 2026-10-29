import { useEffect, useRef, useState } from 'react'
import { planAnimatedTitle } from '../api/client'
import { animatedTitlePrompt, applyTitleCreativePlan, directTitleDescription, titleBackground, titleBrief, titleFps, titleFrames, titlePlanIsCurrent, titlePlanningKey, titleSeconds, TITLE_EFFECTS } from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'

function keyOf(settings: AnimatedTitleSettings) {
  try { return titlePlanningKey(settings) } catch { return JSON.stringify([settings.brief, settings.seconds, settings.sequenceFrames, settings.model, settings.background, settings.backgroundColor]) }
}

export function useTitleComposer(settings: AnimatedTitleSettings, patch: (value: Partial<AnimatedTitleSettings>) => void, scope: string) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const inFlight = useRef(false)
  const mounted = useRef(false)
  const latest = useRef({ key: keyOf(settings), scope })
  useEffect(() => { latest.current = { key: keyOf(settings), scope } }, [settings, scope])
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])

  const enhance = async (submit?: (value: AnimatedTitleSettings) => Promise<void>) => {
    if (inFlight.current) return
    inFlight.current = true; setBusy(true); setError(''); setMessage('')
    const captured = { ...settings, phrases: settings.phrases?.map(phrase => ({ ...phrase })) }
    const capturedKey = keyOf(captured), capturedScope = scope
    const stillHere = () => mounted.current && latest.current.key === capturedKey && latest.current.scope === capturedScope
    try {
      const brief = titleBrief(captured).trim()
      if (!brief || brief.length > 4000) throw new Error('Describe your title animation in 1–4,000 characters.')
      const preset = captured.stylePreset ?? 'custom'
      const resolvedBackground = titleBackground(captured)
      const model = captured.model
      const response = await planAnimatedTitle({ brief, style_hint: preset === 'custom' ? '' : TITLE_EFFECTS[preset],
        frames: titleFrames(titleSeconds(captured), captured), background: resolvedBackground.plannerToken,
        fps: titleFps(captured), native_h3: model?.nativeH3 ?? true })
      if (!stillHere()) return
      const diagnostics = { rawTitleDraft: response.raw_draft.slice(0, 64000), titleDraftHistoryId: response.draft_history_id,
        plannerWarnings: response.warnings, titleReviewRequired: response.review_required }
      patch(diagnostics)
      if (!response.plan) {
        throw new Error(response.warnings.join(' ') || 'The AI draft needs review. Your description was retained; you can edit the plan manually in Advanced.')
      }
      const planned = { ...applyTitleCreativePlan(captured, response.plan), ...diagnostics }
      if (titleFrames(titleSeconds(planned), planned) !== titleFrames(titleSeconds(captured), captured)) {
        planned.plannerWarnings = [...(planned.plannerWarnings ?? []), 'The clip length was increased to fit at least one second per phrase.']
        planned.titleReviewRequired = true
      }
      patch(planned)
      if (planned.titleReviewRequired) setMessage('Review the detected title text and order below. Generate will use this plan.')
      else if (submit) await submit(planned)
      else setMessage('Title plan ready. Review it below or generate when you are ready.')
    } catch (reason) {
      if (mounted.current) {
        setError(reason instanceof Error ? reason.message : 'Title planning failed. Your description was retained.')
        const historyId = (reason as { draft_history_id?: string })?.draft_history_id
        if (historyId && stillHere()) patch({ titleDraftHistoryId: historyId })
      }
    } finally {
      inFlight.current = false
      if (mounted.current) setBusy(false)
    }
  }

  const generate = async (submit: (value: AnimatedTitleSettings) => Promise<void>, options: { enhance?: boolean } = {}) => {
    if (inFlight.current) return
    try {
      if (settings.manualPlanEdited) throw new Error('Choose Use manual title plan to keep your Advanced edits, or Enhance to create a new AI plan.')
      if (titlePlanIsCurrent(settings) && !(settings.directDescription && options.enhance !== false)) {
        const accepted = { ...settings, titleReviewRequired: false }
        animatedTitlePrompt(accepted)
        patch({ titleReviewRequired: false }); await submit(accepted)
      } else if (options.enhance === false) {
        const direct = directTitleDescription(settings)
        patch(direct); setError(''); setMessage('Using your description directly, with AI enhancement off.')
        await submit(direct)
      } else await enhance(submit)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Review the title plan before generating.') }
  }

  const acceptManual = () => {
    try {
      const accepted = { ...settings, brief: titleBrief(settings), directDescription: false, manualPlanEdited: false, titleReviewRequired: false, plannerWarnings: [] }
      accepted.plannedKey = titlePlanningKey(accepted)
      animatedTitlePrompt(accepted); patch(accepted); setError(''); setMessage('Manual title plan ready.')
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Check the manual title settings.') }
  }
  const clearFeedback = () => { setError(''); setMessage('') }
  return { busy, error, message, enhance: () => enhance(), generate, acceptManual, clearFeedback }
}

export type TitleComposerController = ReturnType<typeof useTitleComposer>
