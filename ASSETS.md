# DEF-GHOSTFWL — Asset Manifest

## Paper
- Title: Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal
- ArXiv: 2603.28224
- URL: https://arxiv.org/abs/2603.28224
- Project page: https://keio-csg.github.io/Ghost-FWL/
- Authors: Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako, Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, Kentaro Yoshioka

## Status: ALMOST

This module has the paper PDF, a vendored reference repo in `repositories/Ghost-FWL/`, and concrete training settings from the paper and appendix. The public dataset page exists, but the dataset archive URL and pretrained checkpoints are not exposed in the vendored docs, so assets are not yet fully resolvable.

## Local Reference Assets
| Asset | Type | Local Path | Source | Status |
|---|---|---|---|---|
| Paper PDF | paper | `papers/2603.28224_Ghost-FWL.pdf` | arXiv | DONE |
| Reference repo | code | `repositories/Ghost-FWL/` | https://github.com/Keio-CSG/Ghost-FWL | DONE |
| Project scaffold | code | `src/anima_shinigami/` | local scaffold | STALE |

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---|---|---|---|
| `fwl_mae_pretrain` encoder checkpoint | 6-layer ViT encoder, `Dencoder=768`, `Ddecoder=384` | Project page / future release | `/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl/fwl_mae_pretrain.ckpt` | MISSING |
| `fwl_mae` classifier checkpoint | frozen encoder + 2-layer classifier head | Project page / future release | `/Volumes/AIFlowDev/RobotFlowLabs/models/ghost_fwl/fwl_mae_classifier.ckpt` | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---|---|---|---|---|
| Ghost-FWL supervised set | 24,412 annotated static frames, 7.5B peak labels, 10 scenes | train/val/test | https://keio-csg.github.io/Ghost-FWL/ | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/ghost_dataset/` | MISSING |
| Ghost-FWL mobile unlabeled set | 8,933 mobile-trajectory frames | pretrain | https://keio-csg.github.io/Ghost-FWL/ | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/mae_dataset/` | MISSING |
| Scene-level GT maps | per-scene Mid-360 SLAM maps + glass/reflection regions | annotation dependency | built from appendix C.2 pipeline | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ghost_fwl/gt_maps/` | MISSING |

## Dataset Structure Facts
| Asset | Shape / Structure | Source |
|---|---|---|
| Raw FWL voxel file | `(400, 512, 700)` on disk | `repositories/Ghost-FWL/docs/README_dataset.md` |
| Peak metadata file | `(204800, 3)` for `400*512` rays | `repositories/Ghost-FWL/docs/README_dataset.md` |
| Model preprocessing target | `(H, W, T) = (128, 128, 256)` | paper §5.1, appendix D.1 |
| Model tensor layout | `Tensor[B, 1, 256, 128, 128]` after reorder to `[C, T, H, W]` | inferred from paper + `repositories/Ghost-FWL/src/models/FWLMAE.py` |
| Classes | `noise=0, object=1, glass=2, ghost=3` | `repositories/Ghost-FWL/src/config/constants.py` |

## Supervised Split
| Split | Frames | Scenes | Source |
|---|---|---|---|
| train | 13,853 | 001, 003, 004, 005, 006, 008, 010 | appendix D.2 |
| val | 2,994 | 001, 003, 004, 005, 006, 008, 010 | appendix D.2 |
| test | 1,427 | 002, 007, 009 | appendix D.2 |

## Scene Inventory
| Scene | Frames | Location | Source |
|---|---|---|---|
| 001 | 2500 | Indoor | appendix Table 5 |
| 002 | 2500 | Indoor | appendix Table 5 |
| 003 | 2749 | Indoor | appendix Table 5 |
| 004 | 1853 | Indoor | appendix Table 5 |
| 005 | 2500 | Outdoor | appendix Table 5 |
| 006 | 2445 | Outdoor | appendix Table 5 |
| 007 | 2461 | Outdoor | appendix Table 5 |
| 008 | 2300 | Outdoor | appendix Table 5 |
| 009 | 2354 | Outdoor | appendix Table 5 |
| 010 | 2750 | Outdoor | appendix Table 5 |

## Annotation Parameters
| Param | Value | Source |
|---|---|---|
| Classes | Object, Glass, Ghost, Noise | paper §3.3 |
| Nearest-neighbor threshold `tau` | `0.5 m` for all 10 scenes | appendix Table 6 |
| Accumulation per viewpoint | about 50 frames | paper §3.2, appendix C.2/C.3 |
| Viewpoints per scene | 37-55 | paper §3.2, appendix C.2 |
| Glass region count | scene-specific: 4-48 | appendix Table 6 |

## Hyperparameters From Paper
| Param | Value | Paper Section |
|---|---|---|
| pretrain optimizer | AdamW | appendix D.1 |
| finetune optimizer | AdamW | appendix D.2 |
| `beta1`, `beta2`, `eps` | `0.9`, `0.999`, `1e-8` | appendix D.1/D.2 |
| weight decay | `1e-2` | appendix D.1/D.2 |
| learning rate | `1e-3` | appendix D.1/D.2 |
| batch size | `32` | appendix D.1/D.2 |
| epochs | `100` | appendix D.1/D.2 |
| pretrain mask ratio | `0.70` | appendix D.1 |
| patch size | `(16, 16, 256)` in `(Hpatch, Wpatch, Tpatch)` | appendix D.1 |
| encoder depth / heads | `6 / 6` | paper §4.1, appendix Table 7 |
| decoder depth / heads | `6 / 6` | appendix Table 7 |
| `Dencoder`, `Ddecoder` | `768`, `384` | appendix D.1 |
| peak head `K` | `4` peaks per patch | appendix D.1 |
| pretrain loss weights | `lambda_p=1.0`, `lambda_a=1.0`, `lambda_w=0.5` | appendix D.1 |
| focal alpha | `[glass=0.25, ghost=0.7, object=0.05, noise=0.0001]` | appendix D.2 |
| focal gamma | `2.0` | appendix D.2 |
| preprocessing crop | top 90, bottom 90, front 25 bins removed | paper §5.1, appendix D.1/D.2 |

## Compute Requirements
| Resource | Value | Source |
|---|---|---|
| paper training box | Intel Xeon w5-3535X + single RTX 6000 Ada | appendix D |
| original sensor histogram | `512 x 400 x 700`, about `1 ns` time resolution, `105 m` max range | paper §3.1 |
| ANIMA local dev target | Apple Silicon MLX okay for inference experiments, CUDA preferred for reproduction | local project docs + paper compute profile |

## Expected Metrics From Paper
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---|---|
| ghost denoising | Recall | `0.751` | `>=0.73` |
| ghost denoising | Ghost Removal Rate | `0.918` | `>=0.90` |
| SLAM | ATE | `0.245 +/- 0.138 m` | `<=0.30 m` |
| SLAM | RTE | `0.245 +/- 0.131 m` | `<=0.30 m` |
| object detection | Ghost FP Rate | `1.34%` | `<=2.0%` |

## External Downloads
| Asset | URL | Notes |
|---|---|---|
| Paper | https://arxiv.org/pdf/2603.28224.pdf | mirrored locally |
| Project page | https://keio-csg.github.io/Ghost-FWL/ | primary release page |
| Reference repo | https://github.com/Keio-CSG/Ghost-FWL | mirrored locally in `repositories/Ghost-FWL/` |
| Dataset archive | pending public URL on project page | exact direct link not published in vendored docs |

## Immediate Gaps
1. Resolve the public Ghost-FWL dataset archive URL and expected checksums.
2. Confirm whether official pretrained weights will be released.
3. Replace the stale `SHINIGAMI` scaffold namespace with `DEF-GHOSTFWL` in package, config, and PRD surfaces.
