const {
  isSolCapable,
  needsCuda13DriverUpdate,
  runtimeProfile,
} = require("./launcher_profile")
module.exports = {
  version: "8.0",
  title: "Maestro",
  description: "An all-in-one, 100% local AI video, image & music studio. Its Director mode turns a single prompt into a full music video or short film — LLM-planned, shot by shot. Built on the WanGP pipeline (Wan 2.1/2.2, LTX-2.3, Qwen, Hunyuan Video, Flux). Requires an NVIDIA GPU (6GB+ VRAM).",
  icon: "maestro_simplified_icon_alpha.png",
  menu: async (kernel, info) => {
    const runtime = runtimeProfile(kernel)
    const solCapable = isSolCapable(kernel)
    const cuda13DriverUpdateRequired = solCapable && needsCuda13DriverUpdate(kernel)
    // Do not gate this menu on kernel.gpu. Pinokio can render an app menu
    // before its hardware inventory has populated that property, which would
    // hide Start from supported systems. install.js retains the documented
    // execution-time NVIDIA check for fresh installations.
    let installed = info.exists("app/env") || info.exists("app/env-rtx50") || info.exists("app/env-sol")
    let runtimeReady = info.exists(runtime.marker) && info.exists(runtime.flashMarker)
    let running = {
      install: info.running("install.js"),
      sol_install: info.running("sol_install.js"),
      start: info.running("start.js"),
      start_sol: info.running("start_sol.js"),
      start_classic: info.running("start_classic.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install || running.sol_install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: running.sol_install
          ? "Installing H3 Sol Engine Runtime"
          : "Installing",
        href: running.sol_install
          ? "sol_install.js"
          : "install.js",
      }]
    } else if (running.update) {
      return [{
        default: true,
        icon: 'fa-solid fa-terminal',
        text: "Updating",
        href: "update.js",
      }]
    } else if (installed && solCapable && !cuda13DriverUpdateRequired && !runtimeReady) {
      return [{
        default: true,
        icon: "fa-solid fa-bolt",
        text: "Finish H3 Performance Runtime Upgrade",
        href: "update.js",
      }, ...(info.exists("app/env") ? [{
        icon: "fa-solid fa-shield-halved",
        text: "Start with Compatibility Runtime",
        href: "start.js",
      }] : []), {
        icon: "fa-regular fa-circle-xmark",
        text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
        href: "reset.js",
        confirm: "Are you sure you wish to reset the app?"
      }]
    } else if (installed && solCapable && cuda13DriverUpdateRequired && !runtimeReady) {
      return [...(info.exists("app/env") ? [{
        default: true,
        icon: "fa-solid fa-shield-halved",
        text: "Start with Compatibility Runtime",
        href: "start.js",
      }] : []), {
        icon: "fa-solid fa-triangle-exclamation",
        text: `NVIDIA Driver 580+ Required (found ${kernel.gpu_driver})`,
        href: "https://www.nvidia.com/Download/index.aspx",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
        href: "reset.js",
        confirm: "Are you sure you wish to reset the app?"
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-rocket",
            text: "Open Classic UI",
            href: local.url + "/classic",
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.start_sol) {
        let local = info.local("start_sol.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-bolt",
            text: "Open Web UI (Sol Runtime)",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Sol Runtime Terminal",
            href: "start_sol.js",
          }]
        }
        return [{
          icon: 'fa-solid fa-terminal',
          text: "Sol Runtime Terminal",
          href: "start_sol.js",
        }]
      } else if (running.start_classic) {
        let local = info.local("start_classic.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Classic UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start_classic.js",
          }]
        } else {
          return [{
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start_classic.js",
          }]
        }
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }, {
          icon: "fa-solid fa-display",
          text: "Start (Classic UI)",
          href: "start_classic.js",
        }, {
          icon: "fa-solid fa-power-off",
          text: "Advanced",
          menu: [{
            icon: "fa-solid fa-power-off",
            text: "Compiled (Faster but may not work)",
            href: "start.js",
            params: {
              compile: true
            }
          }, {
            icon: "fa-solid fa-power-off",
            text: "Classic Compiled",
            href: "start_classic.js",
            params: {
              compile: true
            }
          }, ...(solCapable && !cuda13DriverUpdateRequired ? [{
            icon: "fa-solid fa-bolt",
            text: "Repair H3 Performance Runtime",
            href: "torch.js",
            params: {
              venv: runtime.env,
              path: "app",
              xformers: true
            }
          }] : [])]
        }, {
          icon: "fa-regular fa-folder-open",
          text: "T2V Loras (save lora files here)",
          href: "app/loras",
          fs: true
        }, {
          icon: "fa-regular fa-folder-open",
          text: "I2V Loras (save lora files here)",
          href: "app/loras_i2v",
          fs: true
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          // Install / re-install the SAM 3.1 segmentation service
          // (separate Python 3.12 conda env, takes ~5 min). Only
          // needed for the experimental Inpaint feature in Edit
          // mode — most users never need it, which is why install.js
          // no longer runs sam_install.js automatically. Label flips
          // to "Update Inpaint Support" once installed so users can
          // refresh SAM independently of the main app update.
          icon: "fa-solid fa-vector-square",
          text: info.exists("app/services/sam/env")
            ? "Update Inpaint Support (SAM 3.1)"
            : "Install Inpaint Support (SAM 3.1)",
          href: "sam_install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js",
          confirm: "Are you sure you wish to reset the app?"
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
