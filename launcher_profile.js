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

const runtimeProfile = (kernel = {}) => {
  if (isRtx50(kernel)) {
    return {
      env: "env-rtx50",
      python: "3.11",
      marker: "app/env-rtx50/.maestro_torch_rtx50_v1.installed",
      flashMarker: "app/env-rtx50/.maestro_flash_2_8_3_v1.installed",
      label: "RTX 50 / CUDA 13",
    }
  }
  return {
    env: "env",
    python: "3.10",
    marker: "app/env/.maestro_torch_v1.installed",
    flashMarker: "app/env/.maestro_flash_2_7_4_v1.installed",
    label: "CUDA 12.8 legacy",
  }
}

const solRuntimeProfile = (kernel = {}) => {
  if (isRtx50(kernel)) return runtimeProfile(kernel)
  return {
    env: "env-sol",
    python: "3.11",
    marker: "app/env-sol/.maestro_sol_runtime_v1.installed",
    flashMarker: "app/env-sol/.maestro_sol_flash_2_8_3_v1.installed",
    label: "H3 Sol Engine / CUDA 13",
  }
}

module.exports = {
  isRtx40,
  isRtx50,
  isSolCapable,
  runtimeProfile,
  solRuntimeProfile,
}
