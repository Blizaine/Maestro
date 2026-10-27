import { useId, useState } from 'react'
import { useStore } from '../../stores/useStore'
import type { SystemConfig } from '../../types'
import * as api from '../../api/client'

const selectClass = 'w-full bg-bg-primary border border-border rounded-md px-2 py-1.5 text-xs text-text-primary'
const labelClass = 'block text-[11px] text-text-muted mb-1'
const helpClass = 'text-[10px] text-text-muted leading-relaxed mt-1'
const allocatorNames = { default: 'PyTorch default', vmm: 'MMGP optimized', vmm_spill: 'MMGP optimized with RAM spilling' }

export function MemorySettingsPanel() {
  const config = useStore(s => s.systemConfig)
  const update = useStore(s => s.updateSystemConfig)
  const loadSystemConfig = useStore(s => s.loadSystemConfig)
  const id = useId()
  const [runtimeSaving, setRuntimeSaving] = useState(false)
  const [runtimeError, setRuntimeError] = useState<string | null>(null)
  if (!config || config.vram_allocator === undefined) return null
  const allocator = config.vram_allocator
  const active = config.vram_allocator_active ?? 'default'
  const savedRuntime = config.saved_runtime ?? config.memory_runtime ?? 'legacy'
  const runtimeExperimental = savedRuntime === 'wangp9905'
    || config.active_runtime === 'wangp9905'
  const saveRuntime = async (next: 'legacy' | 'wangp9905') => {
    setRuntimeSaving(true)
    setRuntimeError(null)
    try {
      await api.updateSystemConfig({ memory_runtime: next })
      await loadSystemConfig()
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error))
    } finally {
      setRuntimeSaving(false)
    }
  }
  return (
    <section className="space-y-4 border-t border-border pt-4" aria-labelledby={`${id}-title`}>
      <h3 id={`${id}-title`} className="text-[11px] text-text-secondary uppercase tracking-wider font-medium">RAM / VRAM Management</h3>
      <p className={helpClass}>Auto-tune can size H3 transformer residency for each generation while preload is at Profile default. Changing a managed control here switches to manual mode.</p>
      <div>
        <label className={labelClass} htmlFor={`${id}-runtime`}>Memory Runtime</label>
        <select id={`${id}-runtime`} data-testid="memory-runtime-select" className={selectClass}
          value={savedRuntime} disabled={runtimeSaving}
          onChange={event => void saveRuntime(event.target.value as 'legacy' | 'wangp9905')}>
          <option value="legacy">Maestro MMGP 4.0 (default)</option>
          <option value="wangp9905" disabled={!config.experimental_runtime_selectable}>
            {config.experimental_runtime_selectable
              ? 'Experimental Profile0 (WanGP9905)'
              : 'Experimental Profile0 (WanGP9905) — VMM asset unavailable'}
          </option>
        </select>
        <p className={helpClass}>Changing the memory runtime requires a full Maestro restart. Your profiles and manual settings are preserved.</p>
        <p className={helpClass}>Active: {config.active_label ?? 'MMGP source not verified'}.</p>
        {!config.experimental_runtime_selectable && <p className={helpClass}>Experimental runtime unavailable: {config.experimental_runtime_capability_reason ?? config.profile0_capability_reason ?? 'allocator support is unavailable on this installation'}.</p>}
        {config.restart_required && <p role="status" className="text-[11px] text-indicator-warning mt-1">Restart Maestro to apply the saved memory runtime.</p>}
        {runtimeExperimental && <p className="text-[10px] text-indicator-warning mt-1">Fish Audio is unsupported on this runtime. Profiles 1 and 3 also place model weights differently than the Maestro default.</p>}
        {runtimeExperimental && !config.profile0_native_available && <p className={helpClass}>Profile 0 is unavailable: {config.profile0_capability_reason ?? 'the pinned VMM binary has not passed availability checks'}.</p>}
        {runtimeExperimental && config.ram_allocator === 'mmgp' && !config.profile0_ram_native_available && <p className="text-[10px] text-indicator-warning mt-1">The saved MMGP RAM allocator has no companion binary in this runtime. Change RAM Allocator to PyTorch default explicitly before selecting this runtime.</p>}
        {runtimeError && <p role="alert" className="text-[10px] text-indicator-warning mt-1">Could not save runtime: {runtimeError}</p>}
      </div>
      <div>
        <label className={labelClass} htmlFor={`${id}-allocator`}>VRAM Allocator</label>
        <select id={`${id}-allocator`} className={selectClass} value={allocator}
          onChange={event => void update({ vram_allocator: event.target.value as SystemConfig['vram_allocator'] })}>
          <option value="default">PyTorch default</option>
          <option value="vmm">MMGP optimized</option>
          <option value="vmm_spill">MMGP optimized with RAM spilling</option>
        </select>
        <p className={helpClass}>MMGP recycles freed VRAM for longer videos and larger images. RAM spilling can finish a job that slightly exceeds VRAM, with a speed cost.</p>
        <p className={helpClass}>Active: {allocatorNames[active]}.</p>
        {config.vram_allocator_cli_override && <p className={helpClass}>The launch argument overrides this saved choice. Remove --vram-allocator when restarting to use the saved setting.</p>}
        {config.vram_allocator_restart_required && <p role="status" className="text-[11px] text-indicator-warning mt-1">Restart Maestro to apply this allocator.</p>}
        {config.vram_allocator_fallback_reason && <p className="text-[10px] text-indicator-warning mt-1">Using PyTorch: {config.vram_allocator_fallback_reason}</p>}
      </div>
      <div>
        <label className={labelClass} htmlFor={`${id}-ram-allocator`}>RAM Allocator</label>
        <select id={`${id}-ram-allocator`} className={selectClass} value={config.ram_allocator ?? 'default'}
          onChange={event => void update({ ram_allocator: event.target.value as SystemConfig['ram_allocator'] })}>
          <option value="default">PyTorch default</option>
          <option value="mmgp" disabled={runtimeExperimental && !config.profile0_ram_native_available && config.ram_allocator !== 'mmgp'}>MMGP optimized</option>
        </select>
        <p className={helpClass}>MMGP reuses freed CPU tensor buffers and returns its cache when RAM is under pressure. This can reduce allocation overhead; live model weights still need RAM.</p>
        <p className={helpClass}>Active: {config.ram_allocator_active === 'mmgp' ? 'MMGP optimized' : 'PyTorch default'}.</p>
        {config.ram_allocator_cli_override && <p className={helpClass}>The launch argument overrides this saved choice. Remove --ram-allocator when restarting to use the saved setting.</p>}
        {config.ram_allocator_restart_required && <p role="status" className="text-[11px] text-indicator-warning mt-1">Restart Maestro to apply the RAM allocator.</p>}
        {config.ram_allocator_fallback_reason && <p className="text-[10px] text-indicator-warning mt-1">Using PyTorch: {config.ram_allocator_fallback_reason}</p>}
      </div>
      <label className="flex gap-2 items-start text-xs text-text-secondary">
        <input type="checkbox" className="mt-0.5" checked={config.smart_memory_pinning ?? true}
          onChange={event => void update({ smart_memory_pinning: event.target.checked })} />
        <span>Smart Memory Pinning<span className={`block ${helpClass}`}>{runtimeExperimental ? 'Profile 0 uses a shared pin pool to stage weights; this preference does not override that pool. Its saved value is preserved.' : 'Uses a small staging buffer in reserved RAM to speed up weight transfers, especially with Profile 5 and image models.'}</span></span>
      </label>
      <label className="flex gap-2 items-start text-xs text-text-secondary">
        <input type="checkbox" className="mt-0.5" checked={config.read_ahead ?? false}
          disabled={runtimeExperimental && !(config.read_ahead ?? false)}
          onChange={event => void update({ read_ahead: event.target.checked })} />
        <span>Read Ahead (Windows)<span className={`block ${helpClass}`}>{runtimeExperimental ? 'Unavailable with the Experimental Profile0 runtime. Turn it off explicitly before switching runtimes; this saved setting is never reset automatically.' : 'Reads checkpoint files ahead of use for faster loading. Windows may retain more file data in its RAM cache.'}</span></span>
      </label>
      {(['video', 'image', 'audio'] as const).map(kind => {
        const profile = config[`${kind}_profile`]
        const resident = profile === 1
        const mode = config[`${kind}_preload_mode`] ?? 'default'
        return <div key={kind}>
          <label className={labelClass} htmlFor={`${id}-${kind}`}>{kind.charAt(0).toUpperCase() + kind.slice(1)} VRAM Preload</label>
          <select id={`${id}-${kind}`} className={selectClass} value={mode}
            onChange={event => void update({ [`${kind}_preload_mode`]: event.target.value } as Partial<SystemConfig>)}>
            <option value="default">Profile default</option>
            <option value="dynamic">Dynamic</option>
            <option value="manual">Manual</option>
          </select>
          {mode === 'manual' && <label className={`${labelClass} mt-2`}>
            Preload per model (MB)
            <input type="number" min={0} max={40000} step={100} className={`${selectClass} mt-1`}
              value={config[`${kind}_preload_in_VRAM`] ?? 0}
              onChange={event => {
                const value = Number(event.target.value)
                if (Number.isInteger(value) && value >= 0 && value <= 40000) void update({ [`${kind}_preload_in_VRAM`]: value })
              }} />
          </label>}
          {resident ? <p className={helpClass}>Profile {profile} keeps models resident. This preload preference applies when using a streaming profile.</p>
            : profile === 4.5 ? <p className={helpClass}>Profile 4.5 sends one part at a time; Dynamic preload cannot apply.</p>
              : mode === 'dynamic' ? <p className={helpClass}>Keeps weights in spare VRAM and releases them when needed. Requires an active MMGP allocator; otherwise the profile default applies.</p>
                : [3, 3.5].includes(profile) && <p className={helpClass}>Profile {profile} uses its 70% VRAM budget. Dynamic preload can adapt it; a manual preload does not change this profile.</p>}
        </div>
      })}
      <div>
        <label className={labelClass} htmlFor={`${id}-ram`}>Reserved RAM ceiling: {config.perc_reserved_mem_max || 'Auto'}{config.perc_reserved_mem_max ? '%' : ''}</label>
        <input id={`${id}-ram`} type="range" min={0} max={80} step={5} className="w-full"
          value={config.perc_reserved_mem_max ?? 0} onChange={event => void update({ perc_reserved_mem_max: Number(event.target.value) })} />
        <p className={helpClass}>Sets the maximum share of system RAM that pinning can reserve. Zero uses automatic sizing. This is a ceiling, not an immediate allocation.</p>
      </div>
      <div>
        <label className={labelClass} htmlFor={`${id}-heads`}>Attention Head Split</label>
        <select id={`${id}-heads`} className={selectClass} value={config.attention_head_split ?? 0}
          onChange={event => void update({ attention_head_split: Number(event.target.value) })}>
          <option value={0}>Off</option><option value={1}>Low</option><option value={2}>Medium — balanced</option><option value={3}>High — lowest VRAM</option>
        </select>
        <p className={helpClass}>Computes supported long attention sequences in groups to reduce peak VRAM, with a speed cost. Unsupported weights or attention paths keep their normal behavior. Sage2 enables additional attention memory savings. Auto-tune can choose Off, Low or Medium for each H3 video on a 12 GB GPU with at least 30 GB RAM. This selector shows the saved setting; changing it selects manual mode.</p>
      </div>
      <div>
        <label className={labelClass} htmlFor={`${id}-int8`}>INT8 Kernel Backend</label>
        <select id={`${id}-int8`} className={selectClass} value={config.int8_kernels ?? 'triton'}
          onChange={event => void update({ int8_kernels: event.target.value as SystemConfig['int8_kernels'] })}>
          <option value="disabled">PyTorch</option><option value="auto">Auto</option>
          <option value="triton">Triton</option><option value="kitchen">Comfy Kitchen</option>
          <option value="wgpkernels">WGPKernels</option>
        </select>
        <p className={helpClass}>Auto tries WGPKernels, Comfy Kitchen, Triton, then PyTorch. WGPKernels uses higher-precision INT8 math on supported RTX 30 series or newer GPUs. Run Pinokio Update to install the matching optional package. The terminal reports the active backend.</p>
      </div>
      <p className={helpClass}>Head splitting applies to the next generation. Pinning, read ahead, preload, and INT8 backend changes reload models on the next generation. Allocator changes require a restart.</p>
    </section>
  )
}
