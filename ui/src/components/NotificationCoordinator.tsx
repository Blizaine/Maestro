import { useEffect } from 'react'
import { useStore } from '../stores/useStore'
import {
  announceMaestroEvent,
  getDeviceNotificationPreferences,
  prepareDeviceNotificationAudio,
  syncBackgroundPush,
} from '../lib/notifications'

const ACTIVE_JOB_STATUSES = new Set(['held', 'queued', 'running'])
const ACTIVE_PIPELINE_STATUSES = new Set(['running', 'paused'])
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled'])

type StoreState = ReturnType<typeof useStore.getState>
type TerminalStatus = 'completed' | 'failed' | 'cancelled'

interface QueueSession {
  id: number
  armed: boolean
  maxActive: number
  completed: number
  failed: number
  cancelled: number
  terminalIds: Set<string>
}

function newQueueSession(): QueueSession {
  return {
    id: Date.now(),
    armed: false,
    maxActive: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
    terminalIds: new Set(),
  }
}

function activeQueueCount(state: StoreState): number {
  const studio = state.jobs.filter(job => ACTIVE_JOB_STATUSES.has(job.status)).length
  const directorEntries = state.directorQueue?.entries || []
  const director = directorEntries.filter(entry => ACTIVE_JOB_STATUSES.has(entry.status)).length
  const pipelineActive = Boolean(
    state.pipelineId
    && (
      state.pipelinePolling
      || (state.pipelineStatus && ACTIVE_PIPELINE_STATUSES.has(state.pipelineStatus.status))
    ),
  )
  const pipelineRepresentedInQueue = Boolean(
    pipelineActive
    && directorEntries.some(entry => (
      entry.status === 'running'
      && (!entry.pipeline_id || entry.pipeline_id === state.pipelineId)
    )),
  )
  return studio + director + (pipelineActive && !pipelineRepresentedInQueue ? 1 : 0)
}

function queueSummaryBody(session: QueueSession): string {
  const parts: string[] = []
  if (session.completed > 0) parts.push(`${session.completed} finished`)
  if (session.failed > 0) parts.push(`${session.failed} failed`)
  if (session.cancelled > 0) parts.push(`${session.cancelled} cancelled`)
  return parts.length > 0 ? parts.join(', ') : 'All queued work has finished.'
}

/**
 * Observes terminal transitions from the one Zustand store shared by Studio,
 * Director, and the universal queue. Zustand subscribers receive every state
 * write synchronously, including the brief `completed` state before Studio
 * removes its placeholder card, so no polling path needs its own alert code.
 */
export function NotificationCoordinator() {
  useEffect(() => {
    let session = newQueueSession()

    // Refresh the server-side copy of this browser's subscription and event
    // preferences after a restart or an app update. Permission was already
    // granted by an earlier explicit user gesture, so this never prompts.
    const savedPreferences = getDeviceNotificationPreferences()
    if (
      savedPreferences.browserNotifications
      && 'Notification' in window
      && Notification.permission === 'granted'
    ) {
      void syncBackgroundPush(savedPreferences).catch(() => {
        // The local in-app notification path remains available when the host
        // has not installed the optional Web Push dependency yet.
      })
    }

    // Browser audio policies require Web Audio to be resumed by a user
    // gesture. Prime it silently on the first interaction when a saved chime
    // preference is already enabled; enabling the toggle also primes it.
    const primeAudio = () => {
      if (getDeviceNotificationPreferences().deviceSound) {
        void prepareDeviceNotificationAudio()
      }
      window.removeEventListener('pointerdown', primeAudio)
      window.removeEventListener('keydown', primeAudio)
    }
    window.addEventListener('pointerdown', primeAudio, { once: true })
    window.addEventListener('keydown', primeAudio, { once: true })

    const publishTerminal = (
      identity: string,
      status: TerminalStatus,
      title: string,
      body: string,
    ) => {
      if (!session.terminalIds.has(identity)) {
        session.terminalIds.add(identity)
        if (status === 'completed') session.completed += 1
        if (status === 'failed') session.failed += 1
        if (status === 'cancelled') session.cancelled += 1
      }

      // Multi-item queues receive one useful summary instead of a chime and
      // OS notification for every completed item. Failures still interrupt
      // immediately; intermediate completions remain visible as quiet toasts.
      const batchCompletion = status === 'completed'
        && session.armed
        && session.maxActive > 1

      announceMaestroEvent({
        key: `${identity}:${status}`,
        category: status === 'failed'
          ? 'failure'
          : status === 'cancelled'
            ? 'cancelled'
            : 'completion',
        title,
        body,
        system: !batchCompletion && status !== 'cancelled',
        sound: !batchCompletion && status !== 'cancelled',
      })
    }

    const unsubscribe = useStore.subscribe((state, previous) => {
      const previousActive = activeQueueCount(previous)
      const nextActive = activeQueueCount(state)

      if (!session.armed && nextActive > 0) {
        session.armed = true
        session.id = Date.now()
      }
      if (session.armed) {
        session.maxActive = Math.max(
          session.maxActive,
          previousActive,
          nextActive,
        )
      }

      const previousJobs = new Map(previous.jobs.map(job => [job.id, job]))
      for (const job of state.jobs) {
        if (!TERMINAL_STATUSES.has(job.status)) continue
        const before = previousJobs.get(job.id)
        if (before?.status === job.status) continue
        const status = job.status as TerminalStatus
        publishTerminal(
          `studio:${job.id || 'pending'}`,
          status,
          status === 'completed'
            ? 'Generation complete'
            : status === 'failed'
              ? 'Generation failed'
              : 'Generation cancelled',
          status === 'completed'
            ? job.outputFiles.length > 0
              ? `${job.outputFiles.length} ${job.outputFiles.length === 1 ? 'output is' : 'outputs are'} ready.`
              : 'Your output is ready in the gallery.'
            : status === 'failed'
              ? (job.error || job.message || 'The generation could not be completed.')
              : 'The generation was stopped.',
        )
      }

      const pipelineStatus = state.pipelineStatus?.status
      const previousPipelineStatus = previous.pipelineStatus?.status
      if (
        state.pipelineId
        && pipelineStatus
        && TERMINAL_STATUSES.has(pipelineStatus)
        && pipelineStatus !== previousPipelineStatus
      ) {
        const status = pipelineStatus as TerminalStatus
        publishTerminal(
          `director:${state.pipelineId}`,
          status,
          status === 'completed'
            ? 'Director project complete'
            : status === 'failed'
              ? 'Director project failed'
              : 'Director project cancelled',
          status === 'completed'
            ? 'The completed project is ready in the gallery and Director Dashboard.'
            : status === 'failed'
              ? (state.pipelineStatus?.error || 'The Director project could not be completed.')
              : 'The Director project was stopped.',
        )
      }

      // The first queue payload is hydration and may contain months of saved
      // terminal entries. Establish it as the baseline instead of announcing
      // old work every time Maestro opens.
      if (previous.directorQueue !== null && state.directorQueue !== null) {
        const previousEntries = new Map(
          previous.directorQueue.entries.map(entry => [entry.id, entry]),
        )
        for (const entry of state.directorQueue.entries) {
          if (!TERMINAL_STATUSES.has(entry.status)) continue
          const before = previousEntries.get(entry.id)
          if (before?.status === entry.status) continue
          const status = entry.status as TerminalStatus
          const identity = entry.pipeline_id
            ? `director:${entry.pipeline_id}`
            : `director-queue:${entry.id}`
          publishTerminal(
            identity,
            status,
            status === 'completed'
              ? 'Director project complete'
              : status === 'failed'
                ? 'Director project failed'
                : 'Director project cancelled',
            status === 'completed'
              ? (entry.scene_description || 'The queued Director project is ready.')
              : status === 'failed'
                ? (entry.error || entry.message || 'The queued Director project failed.')
                : 'The queued Director project was stopped.',
          )
        }
      }

      if (session.armed && previousActive > 0 && nextActive === 0) {
        const terminalCount = session.completed + session.failed + session.cancelled
        if (session.maxActive > 1 && terminalCount > 0) {
          announceMaestroEvent({
            key: `queue:${session.id}:finished`,
            category: 'queue',
            title: session.failed > 0 ? 'Queue finished with errors' : 'Queue complete',
            body: queueSummaryBody(session),
          })
        }
        session = newQueueSession()
      }
    })

    return () => {
      unsubscribe()
      window.removeEventListener('pointerdown', primeAudio)
      window.removeEventListener('keydown', primeAudio)
    }
  }, [])

  return null
}
