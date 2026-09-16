import { useStore } from '../stores/useStore'
import * as api from '../api/client'
import type { GenerationJob } from '../types'

export const dlssSpatialOptions = [1, 1.5, 1.724, 2, 3].map(scale => ({
  value: `dlss5*${scale}`, label: `DLSS 5 ${scale}x${scale === 1 ? ' (native refinement)' : ''}`,
}))

export function trackMediaFlowJobs(ids: string[]) {
  const jobs: GenerationJob[] = ids.map(id => ({ id, status: 'queued', progress: 0,
    step: 0, totalSteps: 0, phase: '', message: 'Queued (Media Flow)',
    outputFiles: [], error: null, oomInfo: null }))
  useStore.setState(state => ({ jobs: [...jobs, ...state.jobs], isGenerating: true }))
  const pending = new Set(ids)
  let polling = false
  const timer = setInterval(async () => {
    if (polling) return
    polling = true
    try {
      await Promise.all([...pending].map(async id => {
        try {
          const status = await api.fetchJobStatus(id)
          const terminal = ['completed', 'cancelled', 'failed'].includes(status.status)
          useStore.setState(state => {
            const updated = state.jobs.map(job => job.id === id ? { ...job, status: status.status,
              progress: status.progress / 100, step: status.step, totalSteps: status.total_steps,
              message: status.message, phase: status.phase, outputFiles: status.output_files,
              error: status.error, oomInfo: status.oom_info ?? null } : job)
            return { jobs: updated, isGenerating: updated.some(job => ['running', 'queued'].includes(job.status)) }
          })
          if (terminal) {
            pending.delete(id)
            if (status.status === 'completed') void useStore.getState().loadOutputs()
          }
        } catch { /* A transient connection failure does not cancel server jobs. */ }
      }))
    } finally {
      polling = false
      if (!pending.size) clearInterval(timer)
    }
  }, 2000)
}
