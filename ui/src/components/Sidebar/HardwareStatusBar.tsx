import { useEffect, useState } from 'react'
import { ChevronUp, ChevronDown, Cpu, MemoryStick, Zap } from 'lucide-react'
import { useStore } from '../../stores/useStore'

// Color a "fullness" bar (VRAM / RAM) by how close to full it is —
// green well below, amber as it tightens, red near the ceiling. This is
// the at-a-glance OOM-risk read that matters most in this app.
function fullnessColor(pct: number): string {
  if (pct >= 90) return 'bg-red-500'
  if (pct >= 75) return 'bg-indicator-warning'
  return 'bg-emerald-500'
}

// Same thresholds, applied to TEXT (used by the collapsed chips, which
// have no bars). Low load stays neutral so only pressure stands out.
function fullnessText(pct: number): string {
  if (pct >= 90) return 'text-chip-red'
  if (pct >= 75) return 'text-indicator-warning'
  return 'text-text-secondary'
}

function Gauge({
  label,
  percent,
  value,
  fill,
  title,
}: {
  label: string
  percent: number
  value: string
  fill: string
  title?: string
}) {
  const w = Math.max(0, Math.min(100, percent))
  return (
    <div className="flex items-center gap-2" title={title}>
      <span className="w-10 shrink-0 text-[10px] text-text-muted uppercase tracking-wide">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-bg-tertiary overflow-hidden">
        <div
          className={`h-full rounded-full ${fill} transition-[width] duration-500`}
          style={{ width: `${w}%` }}
        />
      </div>
      <span className="w-[92px] shrink-0 text-right text-[10px] text-text-secondary tabular-nums">{value}</span>
    </div>
  )
}

const COLLAPSE_KEY = 'hwbar_collapsed'

/**
 * Live hardware status indicators docked at the bottom of the sidebar.
 * Two views, toggled by the chevron and remembered in localStorage:
 *   - Expanded: labeled mini-gauges (GPU util + VRAM, CPU, RAM) plus the
 *     resident model and loaded LLM.
 *   - Collapsed: a single one-line row of tiny status chips (~the height
 *     of the model line), for users who want the readout but not the bulk.
 * Polls GET /api/v1/system-stats every ~2s while mounted (both views);
 * pauses when the tab is hidden.
 */
export function HardwareStatusBar() {
  const stats = useStore(s => s.systemStats)
  const loadSystemStats = useStore(s => s.loadSystemStats)
  const llmStatus = useStore(s => s.llmStatus)

  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try { return localStorage.getItem(COLLAPSE_KEY) === '1' } catch { return false }
  })
  const toggle = () => setCollapsed(c => {
    const next = !c
    try { localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0') } catch { /* ignore */ }
    return next
  })

  useEffect(() => {
    const tick = () => {
      if (typeof document !== 'undefined' && document.hidden) return
      loadSystemStats()
    }
    tick() // populate immediately, don't wait for the first interval
    const id = setInterval(tick, 2000)
    const onVis = () => { if (!document.hidden) loadSystemStats() }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [loadSystemStats])

  const gpu = stats?.gpu
  const ram = stats?.ram
  const cpu = stats?.cpu
  const model = stats?.model
  // Only treat a model as "current" when it is actually resident in VRAM.
  // On a fresh restart `transformer_type` is seeded from the config's
  // last_model_type (the model from the previous session), so without
  // this gate the bar would show a stale name that isn't loaded.
  const modelLoaded = !!model?.loaded

  const fmtGb = (used?: number, total?: number) =>
    used == null || total == null ? '—' : `${used.toFixed(1)} / ${total.toFixed(0)} GB`
  const fmtG = (v?: number) => (v == null ? '—' : `${v.toFixed(1)}G`)

  // ---- Collapsed: one row of tiny status chips ----------------------
  if (collapsed) {
    return (
      <button
        onClick={toggle}
        title="Show hardware status"
        className="w-full flex items-center gap-2.5 px-3 py-1.5 border-t border-border bg-bg-secondary hover:bg-bg-hover transition-colors text-[10px] shrink-0"
      >
        {gpu?.available && (
          <span
            className="flex items-center gap-1 shrink-0 text-text-secondary"
            title={`GPU ${gpu.percent.toFixed(0)}% (3D engine)${gpu.compute_percent != null ? ` · compute ${gpu.compute_percent.toFixed(0)}%` : ''} · VRAM ${fmtGb(gpu.vram_used_gb, gpu.vram_total_gb)}`}
          >
            <Zap size={11} className="text-text-muted" />
            <span className="tabular-nums">{gpu.percent.toFixed(0)}%</span>
            <span className={`tabular-nums ${fullnessText(gpu.vram_percent)}`}>{fmtG(gpu.vram_used_gb)}</span>
          </span>
        )}
        <span className="flex items-center gap-1 shrink-0 text-text-secondary" title={`CPU ${(cpu?.percent ?? 0).toFixed(0)}%`}>
          <Cpu size={11} className="text-text-muted" />
          <span className="tabular-nums">{(cpu?.percent ?? 0).toFixed(0)}%</span>
        </span>
        <span className="flex items-center gap-1 shrink-0 text-text-secondary" title={`RAM ${fmtGb(ram?.used_gb, ram?.total_gb)}`}>
          <MemoryStick size={11} className="text-text-muted" />
          <span className={`tabular-nums ${fullnessText(ram?.percent ?? 0)}`}>{fmtG(ram?.used_gb)}</span>
        </span>
        <span
          className="flex items-center gap-1 min-w-0 ml-auto"
          title={modelLoaded ? `${model?.name || 'Unknown model'} — loaded` : 'No model loaded'}
        >
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${modelLoaded ? 'bg-emerald-500' : 'bg-text-muted/40'}`} />
          <span className="truncate text-text-muted">{modelLoaded ? (model?.name || '—') : 'No model'}</span>
        </span>
        <ChevronUp size={13} className="shrink-0 text-text-muted" />
      </button>
    )
  }

  // ---- Expanded: full gauges ----------------------------------------
  return (
    <div className="px-3 py-2 border-t border-border bg-bg-secondary shrink-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[9px] uppercase tracking-wider text-text-muted">System</span>
        <button
          onClick={toggle}
          title="Collapse"
          className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-text-secondary transition-colors"
        >
          <ChevronDown size={13} />
        </button>
      </div>
      <div className="flex flex-col gap-1">
        {gpu?.available ? (
          <>
            <Gauge label="GPU" percent={gpu.percent} value={`${gpu.percent.toFixed(0)}%`} fill="bg-accent-blue"
              title={gpu.compute_percent != null ? `3D engine (matches Task Manager) · compute (nvidia-smi): ${gpu.compute_percent.toFixed(0)}%` : undefined} />
            <Gauge
              label="VRAM"
              percent={gpu.vram_percent}
              value={fmtGb(gpu.vram_used_gb, gpu.vram_total_gb)}
              fill={fullnessColor(gpu.vram_percent)}
            />
          </>
        ) : (
          <div className="text-[10px] text-text-muted">No NVIDIA GPU detected</div>
        )}
        <Gauge label="CPU" percent={cpu?.percent ?? 0} value={`${(cpu?.percent ?? 0).toFixed(0)}%`} fill="bg-accent-blue" />
        <Gauge label="RAM" percent={ram?.percent ?? 0} value={fmtGb(ram?.used_gb, ram?.total_gb)} fill={fullnessColor(ram?.percent ?? 0)} />
      </div>

      {/* Currently-loaded model(s) */}
      <div className="mt-1.5 pt-1.5 border-t border-border/50 flex flex-col gap-0.5">
        <div
          className="flex items-center gap-1.5 min-w-0"
          title={modelLoaded ? `${model?.name || 'Unknown model'} — resident in VRAM` : 'No generation model loaded'}
        >
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${modelLoaded ? 'bg-emerald-500' : 'bg-text-muted/40'}`} />
          <span className="text-[11px] text-text-secondary truncate">
            {modelLoaded ? (model?.name || 'Unknown model') : 'No model loaded'}
          </span>
        </div>
        {llmStatus?.loaded && llmStatus.model_id && (
          <div className="flex items-center gap-1.5 min-w-0" title="LLM (Director / prompt enhancer)">
            <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-accent-blue" />
            <span className="text-[10px] text-text-muted truncate">LLM · {llmStatus.model_id}</span>
          </div>
        )}
      </div>
    </div>
  )
}
