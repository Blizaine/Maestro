export const EDITOR_FONT_OPTIONS = [
  { value: 'Arial', label: 'Arial', stack: 'Arial, Helvetica, sans-serif' },
  { value: 'Arial Black', label: 'Arial Black', stack: '"Arial Black", Arial, sans-serif' },
  { value: 'Georgia', label: 'Georgia', stack: 'Georgia, "Times New Roman", serif' },
  { value: 'Times New Roman', label: 'Times New Roman', stack: '"Times New Roman", Times, serif' },
  { value: 'Verdana', label: 'Verdana', stack: 'Verdana, Geneva, sans-serif' },
  { value: 'Trebuchet MS', label: 'Trebuchet MS', stack: '"Trebuchet MS", Arial, sans-serif' },
  { value: 'Courier New', label: 'Courier New', stack: '"Courier New", Courier, monospace' },
  { value: 'Impact', label: 'Impact', stack: 'Impact, Haettenschweiler, "Arial Narrow Bold", sans-serif' },
] as const

export const DEFAULT_EDITOR_FONT = EDITOR_FONT_OPTIONS[0].value

export function editorFontStack(fontFamily?: string): string {
  return EDITOR_FONT_OPTIONS.find(option => option.value === fontFamily)?.stack
    || EDITOR_FONT_OPTIONS[0].stack
}
