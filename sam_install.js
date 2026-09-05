// SAM 3.1 Segmentation Service — Install Script
// Creates a separate Python 3.12 conda env and installs SAM 3.1 + dependencies.
// Called directly from the Pinokio menu and by update.js for existing installs.
module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // A previous successful marker must not survive a failed update. The menu
    // treats SAM as installed only after the dependency and import checks below.
    {
      when: "{{exists('app/services/sam/env/.maestro-sam-ready')}}",
      method: "fs.rm",
      params: {
        path: "app/services/sam/env/.maestro-sam-ready"
      }
    },
    // Step 1: Clone SAM 3 repo if not already present
    {
      when: "{{!exists('app/services/sam/sam3')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/facebookresearch/sam3.git app/services/sam/sam3"
      }
    },
    // Step 1b: Pull latest if repo already exists
    {
      when: "{{exists('app/services/sam/sam3')}}",
      method: "shell.run",
      params: {
        path: "app/services/sam/sam3",
        message: "git pull"
      }
    },
    // Step 2: Install PyTorch (CUDA 12.8) in a Python 3.12 conda env
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/sam/env",
          python: "3.12"
        },
        message: [
          "python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ]
      }
    },
    // Step 3: Resolve SAM3 and Maestro's runtime dependencies together. Installing
    // them in separate transactions allowed a later OpenCV/SciPy pass to replace
    // SAM3's required NumPy 1.x with NumPy 2.x while pip still exited successfully.
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/sam/env",
          python: "3.12"
        },
        message: [
          "python -m pip install -r app/services/sam/requirements.txt app/services/sam/sam3",
          "python -m pip check",
          "python -c \"import numpy as np, cv2, scipy; from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor; from sam3.model.sam3_image_processor import Sam3Processor; major = int(np.__version__.split('.')[0]); assert major == 1, f'Expected NumPy 1.x, found {np.__version__}'; print(f'SAM dependency check passed: numpy={np.__version__}, opencv={cv2.__version__}, scipy={scipy.__version__}')\""
        ]
      }
    },
    // Written only after every command above succeeds. Existing incomplete or
    // incompatible environments therefore remain visibly repairable in Pinokio.
    {
      method: "fs.write",
      params: {
        path: "app/services/sam/env/.maestro-sam-ready",
        text: "SAM 3.1 dependency health check passed.\n"
      }
    },
    // Note: SAM 3.1 model checkpoints (~1.7GB each for base + multiplex) are downloaded
    // automatically on first use. The service tries the official facebook/sam3 repo first,
    // and falls back to ungated mirrors (jetjodh/sam3, jetjodh/sam3.1) if gated access
    // is not available.
  ]
}
