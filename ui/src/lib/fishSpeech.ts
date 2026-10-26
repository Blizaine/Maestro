/** Fish runtimes share speech controls; their model loaders remain separate. */
export function isFishSpeechArchitecture(architecture?: string | null): boolean {
  return architecture === 'fish_drama3' || architecture === 'fish_drama3_small'
}
