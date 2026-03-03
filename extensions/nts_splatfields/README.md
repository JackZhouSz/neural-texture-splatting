# Neural Texture Splatting: Expressive 3D Gaussian Splatting for View Synthesis, Geometry, and Dynamic Reconstruction

<div align="center">

### 🚀 SplatFields + NTS Integration

**SIGGRAPH Asia 2025 (Conference Track)**

[Yiming Wang](https://19reborn.github.io/), [Shaofei Wang](https://taconite.github.io/), [Marko Mihajlovic](https://markomih.github.io/), [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)

ETH Zürich

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://19reborn.github.io/nts/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18873-b31b1b.svg)](https://arxiv.org/abs/2511.18873)
</div>

**Neural Texture Splatting (NTS)** extends 3D Gaussian Splatting by introducing a local neural RGBA field per primitive. This codebase is built on [SplatFields](https://github.com/markomih/SplatFields), showcasing NTS’s ability to improve sparse-view and dynamic reconstruction.


## Setup
To get started, please follow the environment and data preparation guides in the [SplatFields repository](https://github.com/markomih/SplatFields).

Once the base environment is ready, install our custom NTS rasterizer:
```bash
# Install the Neural Texture Rasterizer
pip install submodules/diff-gaussian-rasterization_3dgs_3dtex
```

## Training 
Training from scratch on Owlii matches our paper’s results (which used Splatfields checkpoints). We also provide a resume script for sparse-view static scenes.

```bash
# Run training on Owlii, training from scratch.
python scripts/run_owlii.py

# Run training on the NeRF synthetic dataset, resuming from a Splatfields checkpoint.
python scripts/run_nerf.py
```

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