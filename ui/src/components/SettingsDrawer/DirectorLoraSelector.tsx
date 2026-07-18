import { useState, useEffect, useCallback } from 'react'
import { Search, X, Loader2, FolderOpen, Globe, Sparkles, BookOpen } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import { generateLoraGuide, fetchLoraGuide, fetchLoraDetails } from '../../api/client'
import { LoraGuideTooltip, LoraAgeChip, LoraSortToggle, sortLoraNames } from './LoraSelector'
import type { LoraDates } from './LoraSelector'
import type { LoraRecommendedWeights } from '../../types'

/**
 * Compact preset picker for Director mode LoRA sections.
 */
function DirectorPresetPicker({ mode, modelType }: { mode: 'image' | 'video'; modelType: string }) {
  const presets = useStore(s => s.presets)
  const loadPresets = useStore(s => s.loadPresets)
  const directorSetLora = useStore(s => s.directorSetLora)
  const savedLora = useStore(s => s.savedLoraPerMode[mode])

  useEffect(() => { loadPresets() }, [loadPresets])

  const modePresets = presets.filter(p => p.mode === mode && p.model_type === modelType && p.activated_loras.length > 0)

  if (modePresets.length === 0) return null

  const applyPreset = (preset: typeof modePresets[0]) => {
    directorSetLora(
      mode,
      preset.activated_loras,
      preset.loras_multipliers,
      preset.lora_weights || {},
      savedLora?.availableLoras || [],
    )
  }

  return (
    <div className="mb-2">
      <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Presets</label>
      <div className="flex flex-wrap gap-1">
        {modePresets.map(p => (
          <button
            key={p.id}
            onClick={() => applyPreset(p)}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px] border border-border text-text-secondary hover:bg-bg-hover hover:text-text-primary hover:border-accent-blue transition-colors"
            title={`${p.activated_loras.length} LoRA(s): ${p.activated_loras.map(l => l.replace(/\.(safetensors|sft)$/i, '')).join(', ')}`}
          >
            <FolderOpen size={9} className="shrink-0" />
            <span className="truncate max-w-[100px]">{p.name}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/**
 * Standalone LoRA selector for Director mode with recommended weight zones,
 * guide indicators, CivitAI browser trigger, and auto-apply defaults.
 */
export function DirectorLoraSelector({ mode, modelType }: {
  mode: 'image' | 'video'
  modelType: string
}) {
  const savedLora = useStore(s => s.savedLoraPerMode[mode])
  const directorSetLora = useStore(s => s.directorSetLora)
  const openBrowser = useStore(s => s.setLoraBrowserOpen)

  const [availableLoras, setAvailableLoras] = useState<string[]>(savedLora?.availableLoras || [])
  const [activatedLoras, setActivatedLoras] = useState<string[]>(savedLora?.activated_loras || [])
  const [loraWeights, setLoraWeights] = useState<Record<string, number[]>>(savedLora?.loraWeights || {})
  const [phases, setPhases] = useState(1)
  const [loading, setLoading] = useState(false)
  const [search, setSearch] = useState('')
  const [loraWeightRecs, setLoraWeightRecs] = useState<Record<string, LoraRecommendedWeights>>({})
  const [guideStatus, setGuideStatus] = useState<Record<string, 'none' | 'exists' | 'generating' | 'done'>>({})
  const [guideTexts, setGuideTexts] = useState<Record<string, string>>({})
  const [loraDates, setLoraDates] = useState<Record<string, LoraDates>>({})
  // Sticky list order shared with the Studio picker via the store.
  const sortMode = useStore(s => s.loraPickerSort)
  const setSortSticky = useStore(s => s.setLoraPickerSort)

  // Load available LoRAs when model changes
  useEffect(() => {
    if (!modelType) return
    let cancelled = false
    setLoading(true)
    api.fetchLoras(modelType).then(data => {
      if (cancelled) return
      const newPhases = data.guidance_max_phases ?? 1
      setAvailableLoras(data.loras)
      setPhases(newPhases)
      setActivatedLoras(prev => {
        const valid = prev.filter(l => data.loras.includes(l))
        const adjustedWeights: Record<string, number[]> = {}
        valid.forEach(l => {
          const existing = loraWeights[l] || Array(newPhases).fill(1.0)
          if (existing.length < newPhases) {
            adjustedWeights[l] = [...existing, ...Array(newPhases - existing.length).fill(1.0)]
          } else {
            adjustedWeights[l] = existing.slice(0, newPhases)
          }
        })
        if (valid.length !== prev.length || newPhases !== (loraWeights[valid[0]]?.length ?? 1)) {
          const multipliers = serializeMultipliers(valid, adjustedWeights)
          directorSetLora(mode, valid, multipliers, adjustedWeights, data.loras)
        }
        setLoraWeights(adjustedWeights)
        return valid
      })
      setLoading(false)
    }).catch(() => {
      if (!cancelled) { setAvailableLoras([]); setLoading(false) }
    })
    return () => { cancelled = true }
  }, [modelType]) // eslint-disable-line react-hooks/exhaustive-deps

  // Load weight recommendations and guide status
  useEffect(() => {
    if (!modelType) return
    fetchLoraDetails(modelType).then(r => {
      const recs: Record<string, LoraRecommendedWeights> = {}
      const guides: Record<string, string> = {}
      const statuses: Record<string, 'exists' | 'none'> = {}
      const dates: Record<string, LoraDates> = {}
      for (const info of r.loras) {
        if (info.recommended_weights) recs[info.filename] = info.recommended_weights
        if (info.guide) { guides[info.filename] = info.guide; statuses[info.filename] = 'exists' }
        else if (info.has_guide) statuses[info.filename] = 'exists'
        if (info.released_at || info.downloaded_at) {
          dates[info.filename] = { released: info.released_at, downloaded: info.downloaded_at }
        }
      }
      setLoraWeightRecs(recs)
      setGuideTexts(prev => ({ ...prev, ...guides }))
      setGuideStatus(prev => ({ ...prev, ...statuses }))
      setLoraDates(dates)

      // Auto-apply recommended defaults to newly activated LoRAs at 1.0 fill
      for (const lora of activatedLoras) {
        const rec = recs[lora]
        if (!rec) continue
        const currentWeights = loraWeights[lora]
        if (!currentWeights || !currentWeights.every(w => w === 1.0)) continue
        const newWeights = currentWeights.map((_, i) => {
          const phaseRec = rec.phases?.find(p => p.phase === i + 1)
          const d = phaseRec?.default ?? rec.default
          const min = phaseRec?.min ?? rec.min
          const max = phaseRec?.max ?? rec.max
          if (d != null && d >= min && d <= max) return d
          if (min != null && max != null) return Math.round(((min + max) / 2) * 20) / 20
          return d ?? 0.8
        })
        for (let i = 0; i < newWeights.length; i++) {
          updateWeight(lora, i, newWeights[i])
        }
      }
    }).catch(() => {})

    // Check guide status for activated LoRAs
    for (const lora of activatedLoras) {
      if (guideStatus[lora]) continue
      fetchLoraGuide(modelType, lora).then(r => {
        setGuideStatus(s => ({ ...s, [lora]: r.guide ? 'exists' : 'none' }))
      }).catch(() => {})
    }
  }, [modelType, activatedLoras]) // eslint-disable-line react-hooks/exhaustive-deps

  // Sync from store when savedLora changes externally
  useEffect(() => {
    if (savedLora) {
      setActivatedLoras(savedLora.activated_loras || [])
      setLoraWeights(savedLora.loraWeights || {})
      if (savedLora.availableLoras?.length) setAvailableLoras(savedLora.availableLoras)
    }
  }, [savedLora])

  const serializeMultipliers = (loras: string[], weights: Record<string, number[]>) => {
    return loras.map(name => {
      const w = weights[name] || [1.0]
      return w.map(v => v.toFixed(2)).join(';')
    }).join(' ')
  }

  const persist = useCallback((newLoras: string[], newWeights: Record<string, number[]>) => {
    const multipliers = serializeMultipliers(newLoras, newWeights)
    directorSetLora(mode, newLoras, multipliers, newWeights, availableLoras)
  }, [mode, availableLoras, directorSetLora])

  const toggleLora = useCallback((filename: string) => {
    setActivatedLoras(prev => {
      const idx = prev.indexOf(filename)
      const newWeights = { ...loraWeights }
      let next: string[]
      if (idx >= 0) {
        next = prev.filter((_, i) => i !== idx)
        delete newWeights[filename]
      } else {
        next = [...prev, filename]
        // Apply recommended default or fallback
        const rec = loraWeightRecs[filename]
        const defaultWeight = rec?.default ?? 0.8
        newWeights[filename] = Array(phases).fill(defaultWeight)
        if (rec?.phases) {
          newWeights[filename] = newWeights[filename].map((_, i) => {
            const pr = rec.phases?.find(p => p.phase === i + 1)
            return pr?.default ?? rec.default ?? 0.8
          })
        }
      }
      setLoraWeights(newWeights)
      persist(next, newWeights)
      return next
    })
  }, [loraWeights, phases, persist, loraWeightRecs])

  const updateWeight = useCallback((filename: string, phaseIndex: number, value: number) => {
    setLoraWeights(prev => {
      const next = { ...prev }
      if (!next[filename]) return prev
      next[filename] = [...next[filename]]
      next[filename][phaseIndex] = value
      persist(activatedLoras, next)
      return next
    })
  }, [activatedLoras, persist])

  const handleGenerateGuide = async (filename: string) => {
    if (!modelType) return
    setGuideStatus(s => ({ ...s, [filename]: 'generating' }))
    try {
      await generateLoraGuide(modelType, filename)
      setGuideStatus(s => ({ ...s, [filename]: 'done' }))
    } catch (e) {
      console.error('Guide generation failed:', e)
      setGuideStatus(s => ({ ...s, [filename]: 'none' }))
    }
  }

  const clearAll = () => {
    setActivatedLoras([])
    setLoraWeights({})
    persist([], {})
  }

  const displayName = (filename: string) =>
    filename.replace(/\.(safetensors|sft)$/i, '')

  const filtered = sortLoraNames(
    availableLoras.filter(name =>
      displayName(name).toLowerCase().includes(search.toLowerCase())
    ),
    sortMode,
    loraDates,
  )

  if (loading) {
    return (
      <div className="text-xs text-text-muted bg-bg-tertiary border border-border rounded-lg px-3 py-3 text-center flex items-center justify-center gap-2">
        <Loader2 size={12} className="animate-spin" />
        Loading LoRAs...
      </div>
    )
  }

  if (availableLoras.length === 0) {
    return (
      <div className="flex items-center justify-between">
        <div className="text-xs text-text-muted">No LoRAs found</div>
        <button
          onClick={() => openBrowser(true, modelType)}
          className="text-[10px] text-accent-blue hover:text-accent-blue-hover flex items-center gap-0.5"
        >
          <Globe size={10} /> Browse
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Preset picker */}
      <DirectorPresetPicker mode={mode} modelType={modelType} />

      {/* Header with Browse */}
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-[10px] text-text-muted uppercase tracking-wider">LoRAs</label>
        <div className="flex items-center gap-2">
          <LoraSortToggle sort={sortMode} onChange={setSortSticky} />
          <button
            onClick={() => openBrowser(true, modelType)}
            className="text-[10px] text-accent-blue hover:text-accent-blue-hover flex items-center gap-0.5 transition-colors"
            title="Browse CivitAI"
          >
            <Globe size={10} /> Browse
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="relative mb-2">
        <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search LoRAs..."
          className="w-full bg-bg-tertiary border border-border rounded-lg pl-7 pr-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
        />
      </div>

      {/* Available LoRAs list */}
      <div className="max-h-[120px] overflow-y-auto border border-border rounded-lg bg-bg-tertiary">
        {filtered.map(filename => {
          const isActive = activatedLoras.includes(filename)
          return (
            <button
              key={filename}
              onClick={() => toggleLora(filename)}
              className={`w-full text-left px-2.5 py-1.5 text-xs flex items-center gap-2 hover:bg-bg-hover transition-colors ${
                isActive ? 'text-accent-blue' : 'text-text-secondary'
              }`}
            >
              <div className={`w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 ${
                isActive ? 'bg-accent-blue border-accent-blue' : 'border-border'
              }`}>
                {isActive && (
                  <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                    <path d="M1.5 4L3 5.5L6.5 2" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </div>
              <span className="truncate flex-1">{displayName(filename)}</span>
              {loraDates[filename] && (
                <LoraAgeChip
                  released={loraDates[filename].released}
                  downloaded={loraDates[filename].downloaded}
                />
              )}
              {guideTexts[filename] && (
                <span onClick={e => e.stopPropagation()}>
                  <LoraGuideTooltip guide={guideTexts[filename]} />
                </span>
              )}
              {loraWeightRecs[filename] && (
                <span
                  // Functional indicator tokens, not accent-green: Golden
                  // Hour remaps accent-green to amber, which made every
                  // CivitAI dot (and weight zone) look like the fallback
                  // orange. Mirrors LoraSelector.
                  className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                    loraWeightRecs[filename].source === 'civitai' ? 'bg-indicator-success' : 'bg-indicator-warning'
                  }`}
                  title={loraWeightRecs[filename].source === 'civitai' ? 'CivitAI recommended settings' : 'Default settings'}
                />
              )}
            </button>
          )
        })}
        {filtered.length === 0 && (
          <div className="px-3 py-2 text-xs text-text-muted text-center">No matches</div>
        )}
      </div>

      {/* Selected LoRAs with weight sliders */}
      {activatedLoras.length > 0 && (
        <div className="mt-2 space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="text-[10px] text-text-muted uppercase tracking-wider">
              Selected ({activatedLoras.length})
            </div>
            <button
              onClick={clearAll}
              className="text-[10px] text-text-muted hover:text-red-400 transition-colors"
            >
              Clear all
            </button>
          </div>
          {activatedLoras.map(filename => {
            const weights = loraWeights[filename] || Array(phases).fill(1.0)
            return (
              <div key={filename} className="bg-bg-tertiary border border-border rounded-lg px-2.5 py-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-text-primary truncate flex-1 mr-2">
                    {displayName(filename)}
                  </span>
                  <div className="flex items-center gap-0.5 shrink-0">
                    {guideStatus[filename] === 'exists' || guideStatus[filename] === 'done' ? (
                      <span className="p-0.5 text-indicator-success" title="LoRA guide available">
                        <BookOpen size={11} />
                      </span>
                    ) : guideStatus[filename] === 'generating' ? (
                      <span className="p-0.5 text-accent-blue">
                        <Loader2 size={11} className="animate-spin" />
                      </span>
                    ) : (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleGenerateGuide(filename) }}
                        className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-accent-blue transition-colors"
                        title="Generate AI guide for this LoRA"
                      >
                        <Sparkles size={11} />
                      </button>
                    )}
                    <button
                      onClick={() => toggleLora(filename)}
                      className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
                {weights.map((w, i) => {
                  const rec = loraWeightRecs[filename]
                  const phaseRec = rec?.phases?.find(p => p.phase === i + 1)
                  const fallbackMin = 0.6, fallbackMax = 1.0
                  const recMin = phaseRec?.min ?? rec?.min ?? fallbackMin
                  const recMax = phaseRec?.max ?? rec?.max ?? fallbackMax
                  const isCivitai = rec?.source === 'civitai' || (rec != null && rec.source !== 'default')
                  const sliderMax = 2
                  const zoneLeft = (recMin / sliderMax) * 100
                  const zoneWidth = ((recMax - recMin) / sliderMax) * 100
                  const inZone = w >= recMin && w <= recMax
                  const zoneColor = isCivitai
                    ? 'bg-indicator-success/20 border-indicator-success/30'
                    : 'bg-indicator-warning/15 border-indicator-warning/25'
                  const valueColor = inZone
                    ? (isCivitai ? 'text-indicator-success' : 'text-indicator-warning')
                    : 'text-text-muted'

                  return (
                    <div key={i} className="flex items-center gap-2">
                      {phases > 1 && (
                        <span className="text-[10px] text-text-muted w-12 shrink-0" title={phaseRec?.label || ''}>
                          Phase {i + 1}
                        </span>
                      )}
                      <div className="flex-1 relative">
                        <div
                          className={`absolute top-1/2 -translate-y-1/2 h-2 rounded-full ${zoneColor} pointer-events-none`}
                          style={{ left: `${zoneLeft}%`, width: `${zoneWidth}%` }}
                          title={`${isCivitai ? 'CivitAI' : 'Default'}: ${recMin}-${recMax}`}
                        />
                        <input
                          type="range"
                          min={0}
                          max={sliderMax}
                          step={0.05}
                          value={w}
                          onChange={e => updateWeight(filename, i, parseFloat(e.target.value))}
                          className="w-full relative z-10"
                        />
                      </div>
                      <span className={`text-[10px] w-8 text-right shrink-0 ${valueColor}`}>
                        {w.toFixed(2)}
                      </span>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
