// SAM 3.1 Segmentation Service — Install Script
// Creates a separate Python 3.12 conda env and installs SAM 3.1 + dependencies.
// Called by install.js and update.js.
module.exports = {
  run: [
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
          "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ]
      }
    },
    // Step 3: Install SAM 3 package + all microservice deps from requirements.txt
    {
      method: "shell.run",
      params: {
        conda: {
          path: "app/services/sam/env",
          python: "3.12"
        },
        message: [
          "pip install app/services/sam/sam3",
          "pip install -r app/services/sam/requirements.txt"
        ]
      }
    },
    // Note: SAM 3.1 model checkpoints (~1.7GB each for base + multiplex) are downloaded
    // automatically on first use. The service tries the official facebook/sam3 repo first,
    // and falls back to ungated mirrors (jetjodh/sam3, jetjodh/sam3.1) if gated access
    // is not available.
  ]
}
