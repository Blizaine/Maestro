import type { GenerationJob } from '../types'

/**
 * Prompt enhancement happens before Studio submits a generation to the server.
 * Represent that front-end-only work in the same surfaces as real jobs so an
 * idle queue never makes a Generate click look like it was ignored.
 */
export const PROMPT_ENHANCEMENT_ACTIVITY: GenerationJob = {
  id: 'studio-prompt-enhancement',
  kind: 'prompt_enhancement',
  status: 'running',
  progress: 0,
  step: 0,
  totalSteps: 0,
  phase: 'Enhancing prompt with AI...',
  message: 'Loading the prompt LLM, thinking, and writing before generation',
  outputFiles: [],
  error: null,
  oomInfo: null,
}
