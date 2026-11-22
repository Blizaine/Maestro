import { useState } from 'react'
import { X, BookMarked, Trash2, Upload, Play, Loader2, AlertTriangle, Download, ExternalLink, Layers } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { Recipe, RecipeCard, RecipeLora } from '../../api/client'
import * as api from '../../api/client'

/**
 * RecipesOverlay — a reviewable library of bundled and user-saved recipes.
 * People can inspect the model, settings, requirements and workflow before
 * applying the preset to the active Studio mode.
 */
export function RecipesOverlay() {
  const open = useStore(s => s.recipesOpen)
  const setOpen = useStore(s => s.setRecipesOpen)
  const recipes = useStore(s => s.recipes)
  const loading = useStore(s => s.recipesLoading)
  const applyRecipe = useStore(s => s.applyRecipe)
  const deleteRecipe = useStore(s => s.deleteRecipe)
  const loadRecipes = useStore(s => s.loadRecipes)
  const civitaiKeySet = useStore(s => s.servicesConfig?.civitai_api_key_set ?? false)
  const setSettingsOpen = useStore(s => s.setSettingsOpen)
  const setSettingsTab = useStore(s => s.setSettingsTab)

  const [applying, setApplying] = useState<string | null>(null)
  const [missing, setMissing] = useState<{ modelType: string; loras: RecipeLora[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [selectedPrompt, setSelectedPrompt] = useState('')
  const [detailLoading, setDetailLoading] = useState(false)

  if (!open) return null

  const handleApply = async (card: RecipeCard, promptOverride?: string) => {
    setApplying(card.id); setError(null); setMissing(null)
    try {
      const { missing: missingLoras } = await applyRecipe(card.id, promptOverride)
      if (missingLoras.length > 0) {
        // Applied, but some LoRAs aren't installed — keep the overlay open
        // and surface them so the user can fetch them before generating
        // (rather than hitting a cryptic "Loras missing" failure at gen time).
        setMissing({ modelType: card.model_type, loras: missingLoras })
        setSelectedRecipe(null)
      } else {
        setOpen(false)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to apply recipe')
    } finally {
      setApplying(null)
    }
  }

  const openDetails = async (card: RecipeCard) => {
    setSelectedRecipe(null)
    setSelectedPrompt(card.prompt_example || '')
    setDetailLoading(true)
    setError(null)
    try {
      const recipe = await api.fetchRecipe(card.id)
      setSelectedRecipe(recipe)
      setSelectedPrompt(recipe.prompt_example || '')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load recipe details')
    } finally {
      setDetailLoading(false)
    }
  }

  const openCivitaiKeySettings = () => {
    setMissing(null); setOpen(false)
    setSettingsTab('integrations'); setSettingsOpen(true)
  }

  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'; input.accept = '.json'
    input.onchange = async () => {
      const file = input.files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        await api.importRecipe(JSON.parse(text))
        loadRecipes()
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Import failed')
      }
    }
    input.click()
  }

  return (
    <div className="fixed inset-0 z-[80] flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center gap-2 shrink-0">
        <BookMarked size={16} className="text-accent-blue shrink-0" />
        <h1 className="text-sm font-semibold text-text-primary">Recipes</h1>
        <span className="text-[11px] text-text-muted">Review the setup, adjust the prompt, then apply a look</span>
        <div className="flex-1" />
        <button
          onClick={handleImport}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] bg-bg-tertiary border border-border rounded-lg text-text-secondary hover:text-text-primary hover:border-border-light transition-colors"
        >
          <Upload size={12} /> Import
        </button>
        <button onClick={() => setOpen(false)}
          className="p-1.5 rounded-lg bg-bg-secondary hover:bg-bg-hover transition-colors border border-border">
          <X size={16} />
        </button>
      </div>

      {/* Missing-LoRA notice */}
      {missing && (
        <div className="px-4 py-2.5 bg-amber-500/10 border-b border-amber-500/30">
          <div className="flex items-start gap-2">
            <AlertTriangle size={14} className="text-indicator-warning shrink-0 mt-0.5" />
            <div className="flex-1 text-[11px] text-text-primary">
              <div className="font-medium mb-1">
                Applied — but this recipe uses {missing.loras.length} LoRA
                {missing.loras.length > 1 ? 's' : ''} you don't have installed:
              </div>
              <div className="space-y-1">
                {missing.loras.map(l => (
                  <MissingLoraRow key={l.filename} lora={l} modelType={missing.modelType} civitaiKeySet={civitaiKeySet} />
                ))}
              </div>
              {!civitaiKeySet && missing.loras.some(l => l.source_url) && (
                <div className="mt-1.5 text-[10px] text-text-secondary leading-snug">
                  Auto-download needs a free CivitAI API key.{' '}
                  <button onClick={openCivitaiKeySettings} className="underline hover:text-text-primary">Add one in Settings</button>
                  {' '}— then click Download. Or use each “Open source” link to grab it manually.
                </div>
              )}
              <div className="mt-1.5 text-[10px] text-text-secondary">
                The recipe is applied and ready — you just need the LoRA before you Generate.
              </div>
            </div>
            <button onClick={() => { setMissing(null); setOpen(false) }}
              className="text-[10px] text-text-secondary hover:text-text-primary shrink-0">Dismiss</button>
          </div>
        </div>
      )}
      {error && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/30 text-[11px] text-chip-red">{error}</div>
      )}

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center min-h-[300px] text-text-muted">
            <Loader2 size={22} className="animate-spin" />
          </div>
        ) : recipes.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[300px] gap-3 text-text-muted text-center">
            <BookMarked size={28} />
            <p className="text-sm max-w-xs">No recipes yet. Generate something you like, then use
              “Save as Recipe” on it — or Import a recipe file.</p>
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3">
            {recipes.map(card => (
              <RecipeGridCard
                key={card.id}
                card={card}
                applying={applying === card.id}
                onApply={() => handleApply(card)}
                onDetails={() => openDetails(card)}
                onDelete={card.source === 'user' ? () => deleteRecipe(card.id) : undefined}
              />
            ))}
          </div>
        )}
      </div>

      {(detailLoading || selectedRecipe) && (
        <RecipeDetails
          recipe={selectedRecipe}
          loading={detailLoading}
          applying={selectedRecipe ? applying === selectedRecipe.id : false}
          prompt={selectedPrompt}
          error={error}
          onPromptChange={setSelectedPrompt}
          onClose={() => { setSelectedRecipe(null); setDetailLoading(false); setError(null) }}
          onApply={selectedRecipe ? () => handleApply(selectedRecipe, selectedPrompt) : undefined}
        />
      )}
    </div>
  )
}

function RecipeGridCard({ card, applying, onApply, onDetails, onDelete }: {
  card: RecipeCard; applying: boolean; onApply: () => void; onDetails: () => void; onDelete?: () => void
}) {
  return (
    <div className="group relative rounded-xl border border-border bg-bg-secondary overflow-hidden hover:border-accent-blue/60 transition-colors flex flex-col">
      {/* Thumbnail */}
      <button onClick={onDetails} className="block aspect-video bg-bg-tertiary relative">
        {card.thumbnail_url ? (
          <img src={card.thumbnail_url} alt={card.name} className="absolute inset-0 w-full h-full object-cover" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-text-muted">
            <BookMarked size={28} />
          </div>
        )}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
          {applying
            ? <Loader2 size={22} className="text-white animate-spin" />
            : <span className="text-[11px] font-medium text-white opacity-0 group-hover:opacity-100 transition-opacity">Review recipe</span>}
        </div>
        {card.nsfw && (
          <span className="absolute top-1.5 left-1.5 text-[8px] uppercase tracking-wide bg-red-500/80 text-white rounded px-1 py-0.5">Mature</span>
        )}
      </button>

      {/* Body */}
      <div className="p-2.5 flex-1 flex flex-col gap-1">
        <div className="flex items-start justify-between gap-1.5">
          <div className="text-xs font-medium text-text-primary leading-tight">{card.name}</div>
          {onDelete && (
            <button onClick={onDelete} title="Delete recipe"
              className="shrink-0 p-0.5 rounded text-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100">
              <Trash2 size={12} />
            </button>
          )}
        </div>
        {card.description && (
          <div className="text-[10px] text-text-muted leading-snug line-clamp-2">{card.description}</div>
        )}
        <div className="mt-auto pt-1 flex items-center gap-2 text-[9px] text-text-muted">
          <span className="capitalize">{card.mode}</span>
          {card.lora_count > 0 && (
            <span className="flex items-center gap-0.5"><Layers size={9} /> {card.lora_count}</span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-1.5 pt-1.5">
          <button onClick={onDetails}
            className="px-2 py-1.5 text-[10px] text-text-secondary border border-border rounded-md hover:text-text-primary hover:border-border-light transition-colors">
            Review
          </button>
          <button onClick={onApply} disabled={applying}
            className="px-2 py-1.5 text-[10px] text-white bg-accent-blue rounded-md hover:bg-accent-blue-hover transition-colors disabled:opacity-50 flex items-center justify-center gap-1">
            {applying ? <Loader2 size={11} className="animate-spin" /> : <Play size={10} />}
            Apply
          </button>
        </div>
      </div>
    </div>
  )
}

function RecipeDetails({ recipe, loading, applying, prompt, error, onPromptChange, onClose, onApply }: {
  recipe: Recipe | null
  loading: boolean
  applying: boolean
  prompt: string
  error: string | null
  onPromptChange: (value: string) => void
  onClose: () => void
  onApply?: () => void
}) {
  const settings = Object.entries(recipe?.params || {})
  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <section className="w-full max-w-4xl max-h-[92vh] overflow-hidden bg-bg-primary border border-border rounded-xl shadow-2xl flex flex-col"
        onClick={event => event.stopPropagation()}>
        <header className="px-4 py-3 border-b border-border flex items-center gap-2 shrink-0">
          <BookMarked size={15} className="text-accent-blue" />
          <h2 className="text-sm font-semibold text-text-primary flex-1">{recipe?.name || 'Recipe details'}</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg border border-border hover:bg-bg-hover" aria-label="Close recipe details">
            <X size={15} />
          </button>
        </header>
        {loading ? (
          <div className="flex-1 min-h-[240px] flex items-center justify-center text-text-muted"><Loader2 size={22} className="animate-spin" /></div>
        ) : recipe ? (
          <>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {recipe.preview_url ? (
                <video src={recipe.preview_url} autoPlay muted loop playsInline controls preload="metadata"
                  className="w-full max-h-[320px] rounded-lg bg-black object-contain" />
              ) : recipe.thumbnail_url ? (
                <img src={recipe.thumbnail_url} alt={recipe.name} className="w-full max-h-[320px] rounded-lg bg-black object-contain" />
              ) : null}
              <div>
                <p className="text-sm text-text-primary">{recipe.description || 'No description provided.'}</p>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-muted">
                  <span>Mode: <b className="text-text-secondary capitalize">{recipe.mode}</b></span>
                  <span>Model: <b className="text-text-secondary">{recipe.model_type || 'Not specified'}</b></span>
                  <span>Source: <b className="text-text-secondary capitalize">{recipe.source}</b></span>
                </div>
                {recipe.model_guidance && <p className="mt-2 text-[11px] text-text-secondary leading-relaxed">{recipe.model_guidance}</p>}
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <section className="rounded-lg border border-border bg-bg-secondary p-3">
                  <h3 className="text-[11px] uppercase tracking-wide text-text-muted mb-2">Requirements</h3>
                  {recipe.requirements?.length ? (
                    <ul className="space-y-1 text-xs text-text-secondary list-disc pl-4">
                      {recipe.requirements.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
                    </ul>
                  ) : <p className="text-xs text-text-muted">No extra requirements.</p>}
                  {!!recipe.loras?.length && (
                    <div className="mt-2 pt-2 border-t border-border">
                      <div className="text-[10px] text-text-muted mb-1">LoRAs</div>
                      <ul className="space-y-1 text-[11px] text-text-secondary">
                        {recipe.loras.map(lora => <li key={lora.filename} className="font-mono break-all">{lora.filename} <span className="font-sans">× {lora.multiplier}</span></li>)}
                      </ul>
                    </div>
                  )}
                </section>
                <section className="rounded-lg border border-border bg-bg-secondary p-3">
                  <h3 className="text-[11px] uppercase tracking-wide text-text-muted mb-2">Settings</h3>
                  {settings.length ? (
                    <dl className="grid grid-cols-[minmax(100px,0.8fr)_minmax(0,1.2fr)] gap-x-3 gap-y-1 text-[11px]">
                      {settings.map(([key, value]) => (
                        <div key={key} className="contents">
                          <dt className="text-text-muted break-words">{key.replaceAll('_', ' ')}</dt>
                          <dd className="text-text-secondary font-mono break-all">{formatRecipeSetting(value)}</dd>
                        </div>
                      ))}
                    </dl>
                  ) : <p className="text-xs text-text-muted">No saved generation settings.</p>}
                </section>
              </div>

              <section>
                <label className="text-[11px] uppercase tracking-wide text-text-muted block mb-1.5" htmlFor="recipe-prompt">Prompt example</label>
                {recipe.prompt_guidance && <p className="text-[11px] text-text-secondary mb-2 leading-relaxed">{recipe.prompt_guidance}</p>}
                <textarea id="recipe-prompt" value={prompt} onChange={event => onPromptChange(event.target.value)} rows={5}
                  className="w-full bg-bg-secondary border border-border rounded-lg p-3 text-xs text-text-primary leading-relaxed focus:outline-none focus:border-accent-blue resize-y" />
              </section>

              {(recipe.workflow_guidance || recipe.workflow_steps?.length > 0) && (
                <section className="rounded-lg border border-border bg-bg-secondary p-3">
                  <h3 className="text-[11px] uppercase tracking-wide text-text-muted mb-2">Workflow guide</h3>
                  {recipe.workflow_guidance && <p className="text-xs text-text-secondary leading-relaxed mb-3">{recipe.workflow_guidance}</p>}
                  {!!recipe.workflow_steps?.length && (
                    <ol className="space-y-2 list-decimal pl-5">
                      {recipe.workflow_steps.map((step, index) => (
                        <li key={`${step.type}-${index}`} className="text-xs text-text-primary">
                          <span className="font-medium">{step.title}</span>
                          <span className="block text-text-secondary mt-0.5">{step.instruction}</span>
                          {step.setting && <span className="block text-[10px] text-text-muted mt-0.5">{step.setting}{step.value !== undefined ? ` = ${formatRecipeSetting(step.value)}` : ''}</span>}
                        </li>
                      ))}
                    </ol>
                  )}
                </section>
              )}
              {error && <div className="text-xs text-chip-red">{error}</div>}
            </div>
            <footer className="px-4 py-3 border-t border-border flex items-center justify-end gap-2 shrink-0">
              <button onClick={onClose} className="px-3 py-2 text-xs text-text-secondary border border-border rounded-lg hover:text-text-primary">Close</button>
              <button onClick={onApply} disabled={applying}
                className="px-4 py-2 text-xs bg-accent-blue text-white rounded-lg hover:bg-accent-blue-hover disabled:opacity-50 flex items-center gap-1.5">
                {applying ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                Apply recipe
              </button>
            </footer>
          </>
        ) : (
          <div className="p-6 text-sm text-chip-red">{error || 'Unable to load recipe details.'}</div>
        )}
      </section>
    </div>
  )
}

function formatRecipeSetting(value: unknown): string {
  if (typeof value === 'string') return value || '—'
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function MissingLoraRow({ lora, modelType, civitaiKeySet }: { lora: RecipeLora; modelType: string; civitaiKeySet: boolean }) {
  const downloadRecipeLora = useStore(s => s.downloadRecipeLora)
  const [state, setState] = useState<'idle' | 'downloading' | 'done' | 'error'>('idle')

  const handleDownload = async () => {
    setState('downloading')
    try {
      await downloadRecipeLora(lora, modelType)
      setState('done')
    } catch {
      setState('error')
    }
  }

  // The "Open source" link — always a valid fallback (opens the CivitAI page/
  // file directly, no key needed to view it).
  const sourceLink = lora.source_url ? (
    <a href={lora.source_url} target="_blank" rel="noreferrer"
      className="flex items-center gap-0.5 text-accent-blue hover:text-accent-blue-hover shrink-0">
      <ExternalLink size={10} /> Open source
    </a>
  ) : (
    <span className="text-text-secondary shrink-0">install manually</span>
  )

  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-text-secondary truncate">{lora.filename}</span>
      {lora.size_mb ? <span className="text-text-secondary shrink-0">~{Math.round(lora.size_mb)} MB</span> : null}
      {/* In-app auto-download needs a CivitAI key. With a key → offer Download;
          without → skip the button (it would just fail) and show the source
          link so the user can grab it manually. */}
      {lora.source_url && civitaiKeySet && state === 'idle' && (
        <button onClick={handleDownload}
          className="flex items-center gap-0.5 text-accent-blue hover:text-accent-blue-hover shrink-0">
          <Download size={10} /> Download
        </button>
      )}
      {lora.source_url && !civitaiKeySet && state === 'idle' && sourceLink}
      {state === 'downloading' && <Loader2 size={10} className="animate-spin text-indicator-warning shrink-0" />}
      {state === 'done' && <span className="text-indicator-success shrink-0">started ↓ (see download bar)</span>}
      {state === 'error' && sourceLink}
      {!lora.source_url && state === 'idle' && sourceLink}
    </div>
  )
}
