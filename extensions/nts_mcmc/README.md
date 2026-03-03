# Neural Texture Splatting: Expressive 3D Gaussian Splatting for View Synthesis, Geometry, and Dynamic Reconstruction

<div align="center">

### 🚀 3DGS-MCMC + NTS Integration

**SIGGRAPH Asia 2025 (Conference Track)**

[Yiming Wang](https://19reborn.github.io/), [Shaofei Wang](https://taconite.github.io/), [Marko Mihajlovic](https://markomih.github.io/), [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)

ETH Zürich


[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://19reborn.github.io/nts/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18873-b31b1b.svg)](https://arxiv.org/abs/2511.18873)
</div>

**Neural Texture Splatting (NTS)** extends 3D Gaussian Splatting by introducing a local neural RGBA field per primitive. This codebase is built on [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc).


## Setup
To get started, please follow the environment and data preparation guides in the [3DGS-MCMC repository](https://github.com/ubc-vision/3dgs-mcmc).

Once the base environment is ready, install our custom NTS rasterizer:
```bash
# Install the Neural Texture Rasterizer
pip install submodules/diff-gaussian-rasterization_3dgs_3dtex
```

## Training 

```bash
# Standard training on NeRF-Synthetic
python scripts/run_nerf_synthetic.py

# Training with Mip-NeRF
python scripts/run_mipnerf.py
```

To accommodate GPUs with 24 GB VRAM, the `run_mipnerf.py` script is configured to use a reduced density (0.1×) of Gaussian points. If you have more VRAM available, you can scale this factor up in the configuration.

## Citation
```
@misc{wang2025neuraltexturesplattingexpressive,
      title={Neural Texture Splatting: Expressive 3D Gaussian Splatting for View Synthesis, Geometry, and Dynamic Reconstruction}, 
      author={Yiming Wang and Shaofei Wang and Marko Mihajlovic and Siyu Tang},
      year={2025},
      eprint={2511.18873},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18873}, 
}
```