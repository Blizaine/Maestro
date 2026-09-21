export const validAutoSteps = (voice: number, style: number) =>
  [voice, style].every(n => Number.isInteger(n) && n >= 1 && n <= 1600)
