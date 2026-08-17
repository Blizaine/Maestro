"use strict"

// Keep launcher-side hardware routing in one place. Pinokio exposes both a
// normalized GPU model and a CUDA architecture target before Python/PyTorch
// exists, so this works for fresh installs as well as upgrades.
const isRtx50 = (kernel = {}) => {
  const target = String(kernel.gpu_target || "").toLowerCase()
  const model = String(kernel.gpu_model || "").toLowerCase()
  return kernel.gpu === "nvidia" && (
    target === "sm_120" || /(?:geforce\s+)?rtx\s*50\d{2}/i.test(model)
  )
}

const isRtx40 = (kernel = {}) => {
  const target = String(kernel.gpu_target || "").toLowerCase()
  const model = String(kernel.gpu_model || "").toLowerCase()
  return kernel.gpu === "nvidia" && (
    target === "sm_89" || /(?:geforce\s+)?rtx\s*40\d{2}/i.test(model)
  )
}

const isSolCapable = (kernel = {}) => {
  const target = String(kernel.gpu_target || "").toLowerCase()
  return kernel.gpu === "nvidia" && (
    ["sm_89", "sm_90", "sm_100", "sm_120"].includes(target)
    || isRtx40(kernel)
    || isRtx50(kernel)
  )
}

const needsCuda13DriverUpdate = (kernel = {}) => {
  if (kernel.gpu !== "nvidia" || !kernel.gpu_driver) return false
  const driver = Number.parseFloat(String(kernel.gpu_driver))
  return Number.isFinite(driver) && driver < 580
}

const legacyRuntimeProfile = (kernel = {}) => {
  if (isRtx50(kernel)) {
    return {
      env: "env-rtx50",
      python: "3.11",
      // v2 pins Triton 3.6 for the integrated H3 Sol Engine path. The marker
      // bump makes v1.7.5 Update migrate existing RTX 50 environments once.
      marker: "app/env-rtx50/.maestro_torch_rtx50_v2.installed",
      flashMarker: "app/env-rtx50/.maestro_flash_2_8_3_v1.installed",
      flashSupported: true,
      label: "RTX 50 / CUDA 13",
    }
  }
  const target = String(kernel.gpu_target || "").toLowerCase()
  const legacyWindowsFlashSupported = (
    kernel.platform !== "win32"
    || target === "sm_89"
    || isRtx40(kernel)
  )
  return {
    env: "env",
    python: "3.10",
    marker: "app/env/.maestro_torch_v1.installed",
    // The pinned Windows 2.7.4 wheel contains only sm_89 cubins. Bump the
    // marker for older GPUs so Update removes the incompatible package once.
    flashMarker: legacyWindowsFlashSupported
      ? "app/env/.maestro_flash_2_7_4_v1.installed"
      : "app/env/.maestro_flash_disabled_v2.installed",
    flashSupported: legacyWindowsFlashSupported,
    label: "CUDA 12.8 legacy",
  }
}

const solRuntimeProfile = (kernel = {}) => {
  if (isRtx50(kernel)) return legacyRuntimeProfile(kernel)
  return {
    env: "env-sol",
    python: "3.11",
    marker: "app/env-sol/.maestro_sol_runtime_v1.installed",
    flashMarker: "app/env-sol/.maestro_sol_flash_2_8_3_v1.installed",
    flashSupported: true,
    label: "H3 Sol Engine / CUDA 13",
  }
}

// The tested CUDA 13 / Python 3.11 environment is now Maestro's preferred
// runtime on GPUs supported by H3 Sol Engine. Existing RTX 40 installations
// retain app/env as a recovery path; start.js falls back to it automatically
// until the normal Update flow finishes this side-by-side migration.
const runtimeProfile = (kernel = {}) => (
  isSolCapable(kernel) && !needsCuda13DriverUpdate(kernel)
    ? solRuntimeProfile(kernel)
    : legacyRuntimeProfile(kernel)
)

module.exports = {
  isRtx40,
  isRtx50,
  isSolCapable,
  needsCuda13DriverUpdate,
  legacyRuntimeProfile,
  runtimeProfile,
  solRuntimeProfile,
}
