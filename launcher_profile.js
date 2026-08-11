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

module.exports = { isRtx50, runtimeProfile }
