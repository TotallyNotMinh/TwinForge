# TwinForge: Comprehensive Multi-Task Run & Architecture Benchmark

This document provides an exhaustive, comparative summary of all 17 experimental runs completed in the TwinForge project across its development branches. It details the model architectures, lineage from parent branches, parameter and compute profiles (FLOPs/MACs), loss formulations, training dynamics, and empirical performance across depth estimation, semantic segmentation, and boundary detection on the NYUv2 dataset.

---

## 1. Master Benchmark & Computational Overview

| # | Run Name | Git Branch | Input Res | Encoder Backbone | Decoder Architecture | Tasks | Total Params | Trainable Params | Mult-Adds (GMac) | Best Depth $\delta_1$ ($\uparrow$) | Best Depth RMSE [m] ($\downarrow$) | Eigen Protocol Depth $\delta_1$ / RMSE [m] | Best Seg mIoU ($\uparrow$) | Best Bound F1 ($\uparrow$) | Min Val Loss | Epochs (Trained) |
|---|:---|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **`clamped_kendall`** | `origin/main` | $288 \times 384$ | ResNet34 (Unfrozen) | Shared FPN + Late Heads | Depth, Seg, Bound | 25.48M | 25.48M (100%) | 30.08 | 0.7353 (Ep 120) | 0.6268 (Ep 120) | 0.5104 / 0.9347m | 0.2731 (Ep 117) | 0.5429 (Ep 111) | 0.4703 (Ep 43) | 146 (0–145) |
| 2 | **`isolated_depth`** | `depth-only-baseline` | $288 \times 384$ | ResNet34 (Unfrozen) | FPN + Depth Head Only | Depth Only | 25.48M | 25.48M (100%) | 30.08 | 0.7248 (Ep 139) | 0.6470 (Ep 116) | *Checkpoint not available* | N/A | N/A | 0.4812 (Ep 139) | 151 (0–150) |
| 3 | **`isolated_seg`** | `segment-only-baseline` | $288 \times 384$ | ResNet34 (Unfrozen) | FPN + Seg Head Only | Seg Only | 25.48M | 25.48M (100%) | 30.08 | N/A | N/A | *Checkpoint not available* | 0.2754 (Ep 118) | N/A | 1.2111 (Ep 18) | 144 (0–143) |
| 4 | **`drop_head`** | `drop-boundary-head` | $288 \times 384$ | ResNet34 (Unfrozen) | FPN + CrossTask Gate | Depth, Seg | 25.75M | 25.75M (100%) | 31.04 | 0.7171 (Ep 145) | 0.6600 (Ep 145) | *Checkpoint not available* | 0.2705 (Ep 135) | N/A | 0.8736 (Ep 41) | 151 (0–150) |
| 5 | **`earlier_split`** | `earlier-head-branching` | $288 \times 384$ | ResNet34 (Unfrozen) | Early Split at `up3` + SE-Gates | Depth, Seg, Bound | 26.08M | 26.08M (100%) | 40.29 | 0.7213 (Ep 82) | 0.6653 (Ep 79) | 0.5101 / 0.9261m | 0.2618 (Ep 97) | 0.5472 (Ep 116) | 0.4707 (Ep 40) | 140 (0–139) |
| 6 | **`feature_share`** | `cross-task-interaction` | $288 \times 384$ | ResNet34 (Unfrozen) | CrossTask SE-Gate Interaction | Depth, Seg, Bound | 26.09M | 26.09M (100%) | 40.29 | 0.7285 (Ep 120) | 0.6536 (Ep 107) | 0.5027 / 0.9594m | 0.2489 (Ep 108) | **0.5523** (Ep 87) | 0.4733 (Ep 52) | 146 (0–145) |
| 7 | **`feature_share_with_segmentation_factor`** | `cross-task-interaction` | $288 \times 384$ | ResNet34 (Unfrozen) | CrossTask SE-Gate + $2.0\times$ Seg Loss | Depth, Seg, Bound | 26.09M | 26.09M (100%) | 40.29 | 0.7163 (Ep 119) | 0.6692 (Ep 119) | 0.4923 / 0.9685m | 0.2887 (Ep 126) | 0.5493 (Ep 80) | 0.7851 (Ep 39) | 151 (0–150) |
| 8 | **`resnet50_clamped_kendall`** | `resnet50_clamped_kendall` | $288 \times 384$ | ResNet50 (Unfrozen) | Shared FPN + Late Heads | Depth, Seg, Bound | 33.56M | 33.56M (100%) | 33.69 | 0.7186 (Ep 128) | 0.6613 (Ep 116) | *Checkpoint not available* | 0.2929 (Ep 131) | 0.5473 (Ep 111) | **0.4599** (Ep 31) | 151 (0–150) |
| 9 | **`vit`** | `vit-approach` | $384 \times 512$ | ResNet50 (Unfrozen) | ViT Patch + Cross-Attn + All-MLP | Depth, Seg | 45.42M | 45.42M (100%) | 38.52 | 0.6696 (Ep 148) | 0.7498 (Ep 104) | *Checkpoint not available* | 0.3220 (Ep 76) | N/A | 2.1266 (Ep 16) | 151 (0–150) |
| 10 | **`vit_fixed_weight`** | `vit-fixed-weighting` | $384 \times 512$ | ResNet50 (Unfrozen) | ViT Patch + Cross-Attn + All-MLP | Depth, Seg | 45.42M | 45.42M (100%) | 38.52 | 0.6702 (Ep 120) | 0.7457 (Ep 141) | *Checkpoint not available* | 0.3142 (Ep 73) | N/A | 5.2342 (Ep 20) | 146 (0–145, Early Stop) |
| 11 | **`vit-orientation-fixed`** | `vit-fixed-weighting` | $384 \times 512$ | ResNet50 (Unfrozen) | ViT Patch + Cross-Attn (Fixed Orientation) | Depth, Seg | 46.04M | 46.04M (100%) | 38.52 | 0.7596 (Ep 121) | 0.6108 (Ep 121) | 0.7593 / 0.5478m | 0.4005 (Ep 70) | N/A | 4.3281 (Ep 25) | 141 (0–140) |
| 12 | **`vit-isolated-depth`** | `vit-isolated-depth` | $384 \times 512$ | ResNet50 (Unfrozen) | ViT Patch + Cross-Attn (Depth Only) | Depth Only | 46.04M | 46.04M (100%) | 38.52 | 0.7572 (Ep 119) | 0.6136 (Ep 119) | *Checkpoint not available* | N/A | N/A | 2.2341 (Ep 119) | 145 (0–144) |
| 13 | **`vit-full-res`** | `main` | $480 \times 640$ | ResNet50 (Unfrozen) | ViT Full Res + Spatial Dropout + All-MLP | Depth, Seg | 46.04M | 46.04M (100%) | 60.20 | 0.7477 (Ep 128) | 0.6312 (Ep 107) | *Checkpoint not available* | 0.4081 (Ep 136) | N/A | 4.7163 (Ep 128) | 151 (0–150) |
| 14 | **`vit-increase-regularization`** | `main` | $384 \times 512$ | ResNet50 (Frozen stem..L2) | ViT Patch + Multi-scale Regs + Strong Aug | Depth, Seg | 45.68M | 44.24M (96.8%) | 38.52 | 0.7496 (Ep 139) | 0.6282 (Ep 106) | *Checkpoint not available* | 0.4126 (Ep 131) | N/A | 4.6680 (Ep 128) | 151 (0–150) |
| 15 | **`dinov2-backbone`** | `dinov2-vit-s-backbone` | $392 \times 518$ | Frozen DINOv2 / Depth Anything V2 ViT-S | ViT Patch + Cross-Attn + All-MLP | Depth, Seg | 45.75M | 23.69M (51.8%) | 24.91 | 0.9096 (Ep 88) | 0.3755 (Ep 88) | **0.9096** / 0.3755m | 0.5384 (Ep 131) | N/A | 3.3814 (Ep 89) | 151 (0–150) |
| 16 | **`dino-v2-backbone-early-freeze`** | `dinov2-vit-s-backbone` | $392 \times 518$ | DINOv2 ViT-S (Early Freeze, Blocks 6–11 Fine-Tuned, Warm-start) | ViT Patch + Cross-Attn + All-MLP | Depth, Seg | 45.74M | 34.34M (75.1%) | 24.91 | 0.9084 (Ep 90) | 0.3778 (Ep 90) | 0.9082 / 0.3766m | 0.5652 (Ep 138) | N/A | 3.3394 (Ep 94) | 151 (0–150) |
| 17 | **`dinov2-backbone-unfreeze`** | `dinov2-vit-s-backbone` | $392 \times 518$ | DINOv2 ViT-S (Early Freeze, Blocks 6–11 Fine-Tuned, Scratch) | ViT Patch + Cross-Attn + All-MLP | Depth, Seg | 45.74M | 34.34M (75.1%) | 24.91 | **0.9103** (Ep 102) | **0.3683** (Ep 102) | 0.9095 / **0.3681m** | **0.5761** (Ep 133) | N/A | **3.3068** (Ep 87) | 151 (0–150) |

*Note: Runs 1–13 used full end-to-end training (`freeze=False`). Run 14 froze the stem and layers 1–2 of ResNet50 to control memorization. Run 15 froze the Depth Anything V2 ViT-S encoder (`freeze=True`), training only the multi-scale projection layers and MultiHeadDecoder. Run 16 applied early freezing (blocks 6–11 fine-tuned) with warm-starting from Run 15 (`freeze_early=True`). Run 17 trained with early freezing (blocks 6–11 fine-tuned) from scratch (`freeze_early=True`) at learning rate $1\times 10^{-5}$ without checkpoint warm-starting. Mult-Adds (Multiply-Accumulate operations) were benchmarked via `torchinfo` with batch size 1 at the respective training input resolutions.*
*Eigen Protocol Benchmark: Standard evaluation using Eigen crop [45:471, 41:601] against raw 480x640 ground truth depth on the 654 validation images. Evaluated on all checkpoints physically present on disk; checkpoints for runs 2–4, 8–10, and 12–14 were not retained in local storage.*

---

## 2. Architecture Evolution & Branch Lineage

```mermaid
graph TD
    Main["origin/main <br/> <b>clamped_kendall</b> <br/> (ResNet34 + Late FPN, 3 Tasks)"]
    
    Main --> IsoDepth["depth-only-baseline <br/> <b>isolated_depth</b> <br/> (Single-task Depth)"]
    Main --> IsoSeg["segment-only-baseline <br/> <b>isolated_seg</b> <br/> (Single-task Seg)"]
    Main --> DropBound["drop-boundary-head <br/> <b>drop_head</b> <br/> (2 Tasks: Depth + Seg)"]
    Main --> EarlySplit["earlier-head-branching <br/> <b>earlier_split</b> <br/> (Branch at up3, 3 Tasks)"]
    Main --> CrossTask["cross-task-interaction <br/> <b>feature_share</b> <br/> (CrossTask SE-Gating)"]
    CrossTask --> CrossTaskScale["cross-task-interaction <br/> <b>feature_share_with_segmentation_factor</b> <br/> (2.0x Seg Loss Scaling)"]
    Main --> ResNet50["resnet50_clamped_kendall <br/> <b>resnet50_clamped_kendall</b> <br/> (ResNet50 Backbone)"]
    
    ResNet50 --> ViTApp["vit-approach <br/> <b>vit</b> <br/> (ViT Cross-Attention + Kendall Loss, 384x512)"]
    ViTApp --> ViTFix["vit-fixed-weighting <br/> <b>vit_fixed_weight</b> <br/> (Fixed 1:1 Weight + Pairwise Boundary Mask)"]
    ViTFix --> ViTOri["vit-fixed-weighting (commit 73009ac) <br/> <b>vit-orientation-fixed</b> <br/> (Fixed Transposed MATLAB Loading Bug)"]
    ViTOri --> ViTIso["vit-isolated-depth <br/> <b>vit-isolated-depth</b> <br/> (Single-task Depth ViT Baseline, 384x512)"]
    ViTOri --> ViTFull["main (commits 886ecb0..b7f43a6) <br/> <b>vit-full-res</b> <br/> (Full Res 480x640 + Regs + Grad Accum)"]
    ViTFull --> ViTReg["main (commits 4f55505..83418a2) <br/> <b>vit-increase-regularization</b> <br/> (384x512, Early Freeze, Heavy Aug, Regs)"]
    ViTReg --> DinoV2["dinov2-vit-s-backbone (commits cb10a5b..00607bd) <br/> <b>dinov2-backbone</b> <br/> (Depth Anything V2 ViT-S, 392x518, Frozen)"]
    DinoV2 --> DinoEarly["dinov2-vit-s-backbone (commit 84bfafb) <br/> <b>dino-v2-backbone-early-freeze</b> <br/> (Blocks 6-11 Fine-tuned, Warm-started)"]
    DinoV2 --> DinoUnfreeze["dinov2-vit-s-backbone (commit 2d8946c) <br/> <b>dinov2-backbone-unfreeze</b> <br/> (Early Freeze, Blocks 6-11 Fine-Tuned, Scratch)"]

    classDef baseline fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef vit fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef vitbest fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.5px;
    classDef dinobest fill:#e0f2f1,stroke:#00796b,stroke-width:3px;
    classDef abl fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    
    class Main,IsoDepth,IsoSeg baseline;
    class DropBound,EarlySplit,CrossTask,CrossTaskScale,ResNet50 abl;
    class ViTApp,ViTFix,ViTIso vit;
    class ViTOri,ViTFull,ViTReg vitbest;
    class DinoV2,DinoEarly,DinoUnfreeze dinobest;
```

---

## 3. Detailed Per-Run Analysis

### Run 1: `clamped_kendall` (Baseline Multi-Task Model)
* **Branch:** [`origin/main`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/clamped_kendall)
* **Architecture:**
  * **Encoder:** ImageNet-pretrained ResNet34 (`weights=ResNet34_Weights.DEFAULT`), **fully unfrozen** (`freeze=False`, learning rate $1\times 10^{-4}$). Feature stages extracted: `f1` (64ch), `f2` (64ch), `f3` (128ch), `f4` (256ch), `f5` (512ch).
  * **Decoder:** Top-down Feature Pyramid Network (`MultiHeadDecoder`) with bilinear interpolation and concatenative skip connections (learning rate $1\times 10^{-3}$):
    * `bot` (512 $\rightarrow$ 256) at bottleneck `f5`.
    * `dec1` (512 $\rightarrow$ 256) fusing `f4`.
    * `dec2` (384 $\rightarrow$ 128) fusing `f3`.
    * `dec3` (192 $\rightarrow$ 64) fusing `f2`.
    * `dec4` (128 $\rightarrow$ 64) fusing `f1`.
  * **Heads:** Late-branching heads off `up5` (resolution $288 \times 384$):
    * `DepthHead`: 2-layer conv ($64 \rightarrow 64 \rightarrow 1$) with ReLU, Dropout2d(0.1), clamped to $[10^{-3}, 10.0]$.
    * `SegmentHead`: 2-layer conv ($64 \rightarrow 64 \rightarrow 41$) with ReLU, Dropout2d(0.1).
    * `BoundaryHead`: 2-layer conv ($64 \rightarrow 64 \rightarrow 1$) with ReLU, Dropout2d(0.1).
* **Losses:** Bounded Kendall Uncertainty Loss (`KendallMultiTaskLoss` with 3 tasks):
  $$L = \sum_{i=1}^3 \left( \frac{1}{2}\exp(-\log \sigma_i^2) L_i + \frac{1}{2}\log \sigma_i^2 \right)$$
  with $\log \sigma_i^2$ clamped to $[-2.0, 2.0]$.
* **Parameters & MACs:** 25.48M Total | **25.48M Trainable (100%)** | 30.08 GMac ($288 \times 384$).
* **Training Dynamics:**
  * Ran for 146 epochs. Stable convergence with Cosine Annealing LR ($1\times 10^{-4} \rightarrow 1\times 10^{-6}$).
  * Min val loss: 0.4703 at Epoch 43.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7353$ (Ep 120), $\delta_2 = 0.9372$ (Ep 110), $\delta_3 = 0.9861$ (Ep 95), $\text{RMSE} = 0.6268$m (Ep 120), $\text{AbsRel} = 0.1765$ (Ep 112).
  * **Seg:** $\text{mIoU} = 0.2731$ (Ep 117), $\text{Dice} = 0.3836$ (Ep 117), $\text{Pixel Acc} = 0.6361$ (Ep 145).
  * **Bound:** $\text{F1} = 0.5429$ (Ep 111), $\text{Prec} = 0.4510$, $\text{Recall} = 0.7900$ (Ep 17).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.5104$, $\text{RMSE} = 0.9347\text{m}$, $\text{AbsRel} = 0.2858$, $\text{mIoU} = 0.1561$.
* **Key Takeaway:** Provided strong depth estimation baselines for ResNet models ($\delta_1 = 0.7353$, $\text{RMSE} = 0.6268$m). Joint boundary supervision served as an essential structural prior.

---

### Run 2: `isolated_depth` (Single-Task Depth Baseline)
* **Branch:** [`origin/depth-only-baseline`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/isolated_depth)
* **What Changed vs Baseline:** Disabled segmentation and boundary loss backpropagation. Only the depth head received gradients. Encoder was unfrozen (`freeze=False`).
* **Architecture:** Identical to Run 1.
* **Parameters & MACs:** 25.48M Total | **25.48M Trainable (100%)** | 30.08 GMac.
* **Training Dynamics:** Smooth monotonically decreasing depth loss. Best val loss reached at Epoch 139 (0.4812).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7248$ (Ep 139), $\text{RMSE} = 0.6470$m (Ep 116), $\text{AbsRel} = 0.1811$ (Ep 115).
* **Key Takeaway:** Achieved $0.7248$ $\delta_1$, which is lower than the multi-task baseline ($0.7353$). Demonstrated positive transfer in the 3-task setup (+0.0105 $\delta_1$).

---

### Run 3: `isolated_seg` (Single-Task Segmentation Baseline)
* **Branch:** [`origin/segment-only-baseline`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/isolated_seg)
* **What Changed vs Baseline:** Disabled depth and boundary loss backpropagation. Only the semantic segmentation head was trained. Encoder was unfrozen (`freeze=False`).
* **Architecture:** Identical to Run 1.
* **Parameters & MACs:** 25.48M Total | **25.48M Trainable (100%)** | 30.08 GMac.
* **Training Dynamics:** Quickest validation loss minimum: Epoch 18 (1.2111). After Epoch 20, severe overfitting on cross-entropy occurred, with validation loss climbing while mIoU slowly crept up to Epoch 118.
* **Best Scores:**
  * **Seg:** $\text{mIoU} = 0.2754$ (Ep 118), $\text{Dice} = 0.3875$ (Ep 103), $\text{Pixel Acc} = 0.6744$ (Ep 123).
* **Key Takeaway:** Marginally outperformed the 3-task baseline on mIoU ($0.2754$ vs $0.2731$), confirming slight task competition in the shared CNN decoder.

---

### Run 4: `drop_head` (Ablating the Boundary Head)
* **Branch:** [`origin/drop-boundary-head`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/drop_head)
* **What Changed vs Baseline:**
  * Removed boundary detection head (reduced to 2 tasks: Depth + Seg).
  * Incorporated `CrossTaskSEGate` between depth and segmentation decoder channels.
* **Parameters & MACs:** 25.75M Total | **25.75M Trainable (100%)** | 31.04 GMac.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7171$ (Ep 145), $\text{RMSE} = 0.6600$m (Ep 145), $\text{AbsRel} = 0.1855$ (Ep 86).
  * **Seg:** $\text{mIoU} = 0.2705$ (Ep 135), $\text{Dice} = 0.3808$ (Ep 135), $\text{Pixel Acc} = 0.6641$ (Ep 133).
* **Key Takeaway:** Dropping boundary supervision degraded both tasks simultaneously (Depth $\delta_1$ dropped from $0.7353 \rightarrow 0.7171$; Seg mIoU dropped from $0.2731 \rightarrow 0.2705$).

---

### Run 5: `earlier_split` (Early Head Branching)
* **Branch:** [`origin/earlier-head-branching`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/earlier_split)
* **What Changed vs Baseline:** Shared decoder decoupled earlier at `up3`. `dec3`, `dec4`, and SE-gates were duplicated independently inside each task head.
* **Parameters & MACs:** 26.08M Total | **26.08M Trainable (100%)** | 40.29 GMac (+33.9%).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7213$ (Ep 82), $\text{RMSE} = 0.6653$m (Ep 79).
  * **Seg:** $\text{mIoU} = 0.2618$ (Ep 97), $\text{Dice} = 0.3735$ (Ep 95).
  * **Bound:** $\text{F1} = 0.5472$ (Ep 116).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.5101$, $\text{RMSE} = 0.9261\text{m}$, $\text{AbsRel} = 0.2825$, $\text{mIoU} = 0.1520$.
* **Key Takeaway:** Early branching was counterproductive. Seg mIoU dropped to $0.2618$ while compute surged by +10.2 GMac.

---

### Run 6: `feature_share` (Cross-Task Feature Sharing via SE-Gating)
* **Branch:** [`origin/cross-task-interaction`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/feature_share)
* **What Changed vs Baseline:** Bidirectional channel interaction via `CrossTaskSEGate` across depth, seg, and boundary decoders.
* **Parameters & MACs:** 26.09M Total | **26.09M Trainable (100%)** | 40.29 GMac.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7285$ (Ep 120), $\text{RMSE} = 0.6536$m (Ep 107).
  * **Seg:** $\text{mIoU} = 0.2489$ (Ep 108), $\text{Dice} = 0.3561$ (Ep 108).
  * **Bound:** $\text{F1} = \mathbf{0.5523}$ (Ep 87).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.5027$, $\text{RMSE} = 0.9594\text{m}$, $\text{AbsRel} = 0.2936$, $\text{mIoU} = 0.1568$.
* **Key Takeaway:** Boundary and depth gradients dominated the gates, suppressing segmentation gradients and dropping mIoU to $0.2489$.

---

### Run 7: `feature_share_with_segmentation_factor` (Rebalancing Interactive Multi-Tasking)
* **Branch:** [`origin/cross-task-interaction`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/feature_share_with_segmentation_factor)
* **What Changed vs Run 6:** Scaled segmentation loss by static factor of **$2.0\times$** before entering Kendall loss to restore gradient parity.
* **Parameters & MACs:** 26.09M Total | **26.09M Trainable (100%)** | 40.29 GMac.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7163$ (Ep 119), $\text{RMSE} = 0.6692$m (Ep 119).
  * **Seg:** $\text{mIoU} = 0.2887$ (Ep 126), $\text{Dice} = 0.4000$ (Ep 126), $\text{Pixel Acc} = 0.6899$ (Ep 146).
  * **Bound:** $\text{F1} = 0.5493$ (Ep 80).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.4923$, $\text{RMSE} = 0.9685\text{m}$, $\text{AbsRel} = 0.3032$, $\text{mIoU} = 0.1658$.
* **Key Takeaway:** $2\times$ segmentation loss scaling recovered segmentation performance ($0.2489 \rightarrow 0.2887$, +16.0% relative).

---

### Run 8: `resnet50_clamped_kendall` (Encoder Scaling to ResNet50)
* **Branch:** [`origin/resnet50_clamped_kendall`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/resnet50_clamped_kendall)
* **What Changed vs Baseline:** Replaced ResNet34 with unfrozen **ResNet50** backbone.
* **Parameters & MACs:** 33.56M Total | **33.56M Trainable (100%)** | 33.69 GMac.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7186$ (Ep 128), $\text{RMSE} = 0.6613$m (Ep 116), $\text{AbsRel} = 0.1851$ (Ep 101).
  * **Seg:** $\text{mIoU} = 0.2929$ (Ep 131), $\text{Dice} = 0.4059$ (Ep 131), $\text{Pixel Acc} = \mathbf{0.6923}$ (Ep 131).
  * **Bound:** $\text{F1} = 0.5473$ (Ep 111).
  * **Min Val Loss:** **0.4599** (Ep 31).
* **Key Takeaway:** Peak performance for the ResNet CNN family. Bottleneck residual stages unlocked superior semantic representations ($0.2929$ mIoU).

---

### Run 9: `vit` (Vision Transformer Cross-Attention Architecture)
* **Branch:** [`origin/vit-approach`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/vit)
* **What Changed vs Run 8:** Transitioned to hybrid ViT architecture with ResNet50 + MultiHead Cross-Attention Transformer Decoder:
  * Input resolution increased to **$384 \times 512$**.
  * Replaced CNN decoder with `PatchEmbeder`, `CrossTaskRefinementBlock` (8 heads), and 1/4 resolution All-MLP fusion.
  * Pruned boundary head; introduced SILog depth loss + Lovasz-Softmax.
  * Used smooth bounded Kendall uncertainty loss.
* **Parameters & MACs:** 45.42M Total | **45.42M Trainable (100%)** | 38.52 GMac ($384 \times 512$).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.6696$ (Ep 148), $\text{RMSE} = 0.7498$m (Ep 104), $\text{AbsRel} = 0.2067$ (Ep 103).
  * **Seg:** $\text{mIoU} = 0.3220$ (Ep 76), $\text{Dice} = 0.4640$ (Ep 84), $\text{Pixel Acc} = 0.6372$ (Ep 141).
* **Key Takeaway:** Kendall uncertainty drifted to heavily favor segmentation (weight ratio 2.94 : 1.00), giving $0.3220$ mIoU. However, depth accuracy lagged behind ResNet ($0.6696$ vs $0.7353$). Unbeknownst at the time, data loading orientation was inverted.

---

### Run 10: `vit_fixed_weight` (Fixed Task Weighting & Pairwise Boundary Mask Fix)
* **Branch:** [`origin/vit-fixed-weighting`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/vit_fixed_weight)
* **What Changed vs Run 9:**
  * Replaced Kendall loss with **fixed 1:1 weighting**: $L_{total} = L_{seg} + 1.0 \times L_{depth}$.
  * Fixed depth boundary mask logic in [`losses/depth_loss.py`](file:///home/totallynotminh/Documents/TwinForge/losses/depth_loss.py#L46-L51) to prevent invalid edge gradients.
* **Parameters & MACs:** 45.42M Total | **45.42M Trainable (100%)** | 38.52 GMac ($384 \times 512$).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.6702$ (Ep 120), $\text{RMSE} = 0.7457$m (Ep 141), $\text{AbsRel} = 0.2083$ (Ep 110).
  * **Seg:** $\text{mIoU} = 0.3142$ (Ep 73), $\text{Dice} = 0.4564$ (Ep 73), $\text{Pixel Acc} = 0.6280$ (Ep 141).
* **Key Takeaway:** Fixed 1:1 weighting improved depth stability and reached peak depth 28 epochs earlier. Terminated cleanly via Early Stopping at Epoch 145.

---

### Run 11: `vit-orientation-fixed` (The Orientation Alignment Breakthrough)
* **Branch:** [`origin/vit-fixed-weighting`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/vit-orientation-fixed) (Commit `73009ac`)
* **What Changed vs Run 10:**
  * **Critical Bug Fix:** In [`data/nyuv2.py`](file:///home/totallynotminh/Documents/TwinForge/data/nyuv2.py), raw MATLAB `.mat` tensors had inverted spatial axes ($W \times H \times C$ instead of $H \times W \times C$). Images were previously loaded sideways/transposed relative to depth and segmentation ground truths.
  * Corrected transposed tensor indexing and removed redundant bilinear resizing interpolation.
  * Retained identical $384 \times 512$ resolution, fixed 1:1 task weighting, and All-MLP fusion.
* **Parameters & MACs:** 46.04M Total | **46.04M Trainable (100%)** | 38.52 GMac ($384 \times 512$).
* **Training Dynamics:**
  * Ran 141 epochs (terminating at Epoch 140).
  * Reached best validation loss at **Epoch 25** ($4.3281$).
  * Overfitting set in thereafter: validation loss gradually climbed to $4.9918$ at Epoch 140, while training loss fell from $8.74 \to 1.00$ (divergence gap: $\approx 3.99$).
* **Best Scores:**
  * **Depth:** $\delta_1 = \mathbf{0.7596}$ (Ep 121), $\delta_2 = \mathbf{0.9507}$ (Ep 121), $\delta_3 = \mathbf{0.9899}$ (Ep 121), $\text{RMSE} = \mathbf{0.6108}$m (Ep 121), $\text{AbsRel} = \mathbf{0.1655}$ (Ep 139).
  * **Seg:** $\text{mIoU} = 0.4005$ (Ep 70), $\text{Dice} = 0.5497$ (Ep 70), $\text{Pixel Acc} = 0.6867$ (Ep 129).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = \mathbf{0.7593}$ (Ep 121), $\text{RMSE} = \mathbf{0.5478}\text{m}$ (Ep 121), $\text{AbsRel} = 0.1688$, $\text{mIoU} = 0.3957$ (Ep 121) / $\mathbf{0.4005}$ (Ep 70).
* **Key Takeaway:** **The single most impactful update in the repository.** Correcting spatial orientation caused metrics to leap across the board:
  * Depth $\delta_1$ jumped from $0.6702 \rightarrow \mathbf{0.7596}$ (+0.0894), setting the **all-time project record for Depth Estimation**.
  * Depth RMSE dropped from $0.7457\text{m} \rightarrow \mathbf{0.6108}\text{m}$ (a 13.5 cm improvement).
  * Segmentation mIoU leaped from $0.3142 \rightarrow \mathbf{0.4005}$ (+0.0863 / +27.5% relative).
  Proved that transformer cross-attention had been attempting to align misaligned coordinate systems in Runs 9–10.

---

### Run 12: `vit-isolated-depth` (ViT Single-Task Depth Baseline)
* **Branch:** [`origin/vit-isolated-depth`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/vit-isolated-depth) (Commit `54c0032`)
* **What Changed vs Run 11:**
  * Single-task ablation: Disabled segmentation loss backpropagation entirely ($L_{total} = L_{depth}$).
  * Model architecture, input resolution ($384 \times 512$), and fixed orientation remained identical.
* **Parameters & MACs:** 46.04M Total | **46.04M Trainable (100%)** | 38.52 GMac.
* **Training Dynamics:**
  * Smooth convergence over 145 epochs.
  * Best validation loss reached at **Epoch 119** ($2.2341$).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7572$ (Ep 119), $\delta_2 = 0.9494$ (Ep 119), $\delta_3 = 0.9891$ (Ep 119), $\text{RMSE} = 0.6136$m (Ep 119), $\text{AbsRel} = 0.1666$ (Ep 81).
* **Key Takeaway:** ViT single-task depth ($\delta_1 = 0.7572$, $\text{RMSE} = 0.6136$m) performed comparably to multi-task `vit-orientation-fixed` ($\delta_1 = 0.7596$, $\text{RMSE} = 0.6108$m). Confirmed that ViT patch cross-attention with ResNet50 provides outstanding depth representation on its own, with multi-task cross-attention providing a modest positive transfer (+0.0024 $\delta_1$).

---

### Run 13: `vit-full-res` (Full Native Resolution & Regularization Attempt)
* **Branch:** `main` (Commits `886ecb0`, `48e6dfa`, `a291e42`, `b7f43a6`)
* **What Changed vs Run 11:**
  * **Full Resolution:** Increased input resolution from $384 \times 512$ to full native NYUv2 **$480 \times 640$**.
  * **Positional Embeddings:** Scaled fixed grid parameter tables in [`models/multihead_decoder.py`](file:///home/totallynotminh/Documents/TwinForge/models/multihead_decoder.py#L111-L120) to match $480 \times 640$ feature maps (`pos_embed1`: $40 \times 53$, `pos_embed2`: $30 \times 40$, `pos_embed3`: $15 \times 20$, `pos_embed4`: $10 \times 13$, `pos_embed5`: $5 \times 6$).
  * **Added Regularization:**
    * Positional embedding dropout (`Dropout(0.1)`).
    * Multi-scale spatial dropout (`Dropout2d(0.2)`) on `shared_proj4`, `shared_proj5`, and `fuse`.
    * Increased head dropout in `SegmentDecoder` (`Dropout2d(0.3)`).
    * Label smoothing ($0.05$) on Cross-Entropy loss in [`losses/segment_loss.py`](file:///home/totallynotminh/Documents/TwinForge/losses/segment_loss.py#L30).
    * Increased encoder weight decay from $1\times 10^{-4} \to 5\times 10^{-3}$.
  * **Gradient Accumulation:** Reduced batch size to 12 with `grad_accum_steps=2` (effective batch size 24) to accommodate increased activation memory.
* **Parameters & MACs:** 46.04M Total | **46.04M Trainable (100%)** | **60.20 GMac** (+56.3% increase in GMacs).
* **Training Dynamics:**
  * Ran all 151 epochs (0–150).
  * **Persistent Overfitting & Divergence:** Validation loss flatlined at $\sim 4.75$ after epoch 20 (reaching an absolute low of $4.7163$ at Epoch 128), while training loss plunged to $2.1935$. Generalization gap ended at **$\approx 2.57$**. The added dropouts and label smoothing failed to close the train-val divergence.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7477$ (Ep 128), $\delta_2 = 0.9487$ (Ep 128), $\delta_3 = 0.9889$ (Ep 128), $\text{RMSE} = 0.6312$m (Ep 107), $\text{AbsRel} = 0.1679$ (Ep 139).
  * **Seg:** $\text{mIoU} = 0.4081$ (Ep 136), $\text{Dice} = 0.5585$ (Ep 128) / 0.5581 (Ep 136), $\text{Pixel Acc} = 0.6906$ (Ep 142).
* **Key Takeaway & Critical Critique:**
  * **Negative Return on Depth:** Despite the $56\%$ compute jump, depth performance **degraded** relative to `vit-orientation-fixed` ($\delta_1$ dropped from $0.7596 \to 0.7477$; RMSE worsened from $0.6108\text{m} \to 0.6312\text{m}$).
  * **Marginal Segmentation Gain:** mIoU rose by only $+0.0076$ ($0.4005 \to 0.4081$).
  * **Capacity / Data Mismatch:** Fitting 46.0M unfrozen parameters on only 795 training images at full resolution allowed unconstrained self-attention maps to memorize spatial configurations.
  * **Compute Bottleneck:** Quadratic attention overhead on the enlarged $40 \times 53$ token grid made full resolution an inefficient use of compute.

---

### Run 14: `vit-increase-regularization` (Early Encoder Freezing & Heavy Regularization)
* **Branch:** `main` (Commits `4f55505`, `b32a174`, `296b305`, `83418a2`)
* **What Changed vs Run 13:**
  * **Reverted Resolution:** Restored input resolution back to **$384 \times 512$** (reducing compute from 60.20 GMac to 38.52 GMac).
  * **Dynamic Positional Embeddings:** Updated `MultiHeadDecoder` in [`models/multihead_decoder.py`](file:///home/totallynotminh/Documents/TwinForge/models/multihead_decoder.py#L112-L121) to dynamically adjust positional embedding grids based on input resolution `(H, W)`.
  * **Early Encoder Freezing:** Froze `stem`, `layer1`, and `layer2` in `ResNetEncoder` (`freeze_early=True` in [`models/encoder.py`](file:///home/totallynotminh/Documents/TwinForge/models/encoder.py#L29-L32)), training only `layer3`, `layer4`, and decoder heads (reducing trainable parameters from 46.04M to 44.24M).
  * **Aggressive Data Augmentation:**
    * Rotation angle expanded from $[-2^\circ, 2^\circ]$ ($p=0.2$) to $[-8^\circ, 8^\circ]$ ($p=0.7$).
    * Photometric jitter (brightness, contrast, saturation) probability raised to $0.7$, hue to $0.5$.
    * `RandomResizedCrop` scale expanded from $(0.8, 1.0)$ ($p=0.5$) to $(0.6, 1.0)$ ($p=0.7$) in [`data/augment.py`](file:///home/totallynotminh/Documents/TwinForge/data/augment.py#L80-L93).
    * Gaussian blur probability raised to $0.15$.
  * **Maintained Regularizations:** Positional embedding dropout (0.1), multi-scale spatial dropout (0.2), segmentation head dropout (0.3), label smoothing (0.05), and encoder weight decay ($5\times 10^{-3}$).
* **Parameters & MACs:** 45.68M Total | **44.24M Trainable (96.8%)** | 38.52 GMac ($384 \times 512$).
* **Training Dynamics:**
  * Ran all 151 epochs (0–150).
  * **Substantial Overfitting Suppression:** Validation loss bottomed at **4.6680** (Epoch 128) and remained flat through Epoch 150 (4.6882). The final generalization gap was compressed to **2.15** (Train: 2.5385, Val: 4.6882), a ~46% reduction compared to `vit-orientation-fixed` (gap: 3.99).
  * Segmentation continuously improved throughout training (peaking at Ep 131) instead of suffering early plateau/decay.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.7496$ (Ep 139), $\delta_2 = 0.9489$ (Ep 139), $\delta_3 = 0.9899$ (Ep 139), $\text{RMSE} = 0.6282$m (Ep 106) / 0.6354m (Ep 139), $\text{AbsRel} = 0.1687$ (Ep 139).
  * **Seg:** $\text{mIoU} = 0.4126$ (Ep 131), $\text{Dice} = 0.5628$ (Ep 131), $\text{Pixel Acc} = 0.6934$ (Ep 142) / 0.6923 (Ep 131).
* **Key Takeaway & Critical Critique:**
  * **Peak Semantic Performance with CNN Encoder:** Achieved $0.4126$ mIoU and $0.6934$ Pixel Acc, outperforming all prior ResNet-based models.
  * **Strictly Dominates Full-Res Run:** Outperformed `vit-full-res` across every single metric while requiring 35.7% fewer GMacs and training significantly faster.
  * **The Geometric Augmentation vs. Metric Depth Dilemma:** Depth accuracy ($\delta_1 = 0.7496$, $\text{RMSE} = 0.6282$m) remained below `vit-orientation-fixed` ($0.7596$ / $0.6108$m). Aggressive zooming (`scale=(0.6, 1.0)`) alters apparent object dimensions without scaling ground truth distance labels, corrupting metric depth cues while boosting semantic scale-invariance.

---

### Run 15: `dinov2-backbone` (Depth Anything V2 Pretrained ViT-S Backbone)
* **Branch:** [`dinov2-vit-s-backbone`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/dinov2-backbone) (Commits `cb10a5b`, `00607bd`)
* **What Changed vs Run 14:**
  * **Encoder Paradigm Shift:** Completely replaced the CNN ResNet50 encoder with a **DINOv2 ViT-S** backbone pre-trained via **Depth Anything V2** (`depth_anything_v2_vits.pth`).
  * **Encoder Freezing:** The entire 12-block ViT-S transformer backbone (embed dim=384, 6 heads, ~22.06M parameters) was fully frozen (`freeze=True`), running in `eval()` mode with zero backpropagation into backbone weights.
  * **Multi-Scale Feature Adapters:** Tapped intermediate representations from transformer blocks `[2, 5, 8, 11]`, projecting them through $1 \times 1$ convs + BatchNorm + ReLU (`proj1` through `proj5`) to hierarchical channel dimensions ($64, 256, 512, 1024, 2048$), emulating stages `f1` through `f5` for the cross-attention decoder.
  * **Optimized Attention:** Replaced naive dot-product attention with PyTorch 2.x `F.scaled_dot_product_attention` for accelerated memory and compute efficiency.
  * **Resolution Adaptation:** Adjusted input resolution to **$392 \times 518$** to satisfy the ViT patch size divisibility constraint (patch size 14: $392/14 = 28$, $518/14 = 37$).
  * **Maintained Multi-Task ViT Decoder:** Retained the 8-head `MultiHeadDecoder` with fixed 1:1 loss weighting ($L_{seg} + 1.0 \times L_{depth}$).
* **Parameters & MACs:** 45.75M Total | **23.69M Trainable (51.8%)** | **24.91 GMac** ($392 \times 518$).
  * Compute slashed by **-35.3%** compared to ResNet50 ViT ($38.52 \to 24.91$ GMac) and **-58.6%** compared to `vit-full-res` (60.20 GMac).
* **Training Dynamics:**
  * Ran all 151 epochs (0–150).
  * **Minimal Generalization Gap (1.05):** Minimum validation loss reached **3.3814** at Epoch 89 (Train loss: 2.5427), finishing at Train: 2.3430 / Val: 3.3922 at Epoch 150. Divergence gap ended at only **$\approx 1.05$** (compared to 3.99 in Run 11 and 2.15 in Run 14).
  * **Ultra-Fast Early Convergence:** Depth $\delta_1$ reached $0.7934$ by Epoch 2—already surpassing the 150-epoch peak of every previous model in the repository.
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.9096$ (Ep 88), $\delta_2 = \mathbf{0.9895}$ (Ep 88), $\delta_3 = \mathbf{0.9977}$ (Ep 88), $\text{RMSE} = 0.3755$m (Ep 88), $\text{AbsRel} = 0.1017$ (Ep 88), $\text{SqRel} = 0.0527$ (Ep 88), $\text{RMSElog} = 0.1287$ (Ep 88), $\text{log10} = 0.0434$ (Ep 88).
  * **Seg:** $\text{mIoU} = 0.5384$ (Ep 131), $\text{Dice} = 0.6781$ (Ep 131), $\text{Pixel Acc} = 0.7806$ (Ep 131).
  * **Min Val Loss:** 3.3814 (Ep 89).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = \mathbf{0.9096}$, $\text{RMSE} = 0.3755\text{m}$, $\text{AbsRel} = 0.1017$, $\text{mIoU} = 0.5310$.
* **Key Takeaway & Critical Critique:**
  * **Massive Leap Over ResNet Baselines Across Every Task:**
    * Depth $\delta_1$ leaped from $0.7596 \to 0.9096$ (+15.00 percentage points / +19.7% relative improvement).
    * Depth RMSE plummeted from $0.6108\text{m} \to 0.3755\text{m}$ (an astonishing **23.53 cm reduction in error**, -38.5% relative).
    * Semantic segmentation mIoU leaped from $0.4126 \to 0.5384$ (+12.58 percentage points / +30.5% relative improvement).
    * Segmentation Pixel Accuracy surged from $0.6934 \to 0.7806$.
  * **The Power of Foundation Depth Priors:** Freezing a vision foundation model pre-trained on massive synthetic and real datasets provided rich, scale-aware geometric priors and dense semantic tokens that the previous CNN backbones could not discover from only 795 NYUv2 training images.
  * **Efficiency Breakthrough:** Achieved these unprecedented benchmarks while running at only 24.91 GMac and training only 23.69M parameters, proving that encoder representation quality completely dominates raw architecture capacity or decoder parameter count.

---

### Run 16: `dino-v2-backbone-early-freeze` (Partial ViT-S Backbone Fine-Tuning)
* **Branch:** [`dinov2-vit-s-backbone`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/dino-v2-backbone-early-freeze) (Commit `84bfafb`)
* **What Changed vs Run 15:**
  * **Selective Encoder Fine-Tuning (Early Freeze):** Froze `patch_embed` and transformer blocks 0–5, while fine-tuning blocks 6–11 (`freeze_early=True`, `freeze=False`). This freed the upper 6 transformer layers (~10.65M parameters) to adapt to NYUv2 task-specific features while anchoring early geometric patch representations.
  * **Warm-Start Optimization:** Initialized model weights from the converged checkpoint of Run 15 (`dinov2-backbone-freeze`), re-initializing optimizer and scheduler from Epoch 0.
  * **Conservative Learning Rate:** Applied `encoder_lr = 1e-5` to trainable ViT blocks and `decoder_lr = 2e-4` to the decoder.
* **Parameters & MACs:** 45.74M Total | **34.34M Trainable (75.1%)** | **24.91 GMac** ($392 \times 518$).
* **Training Dynamics:**
  * Ran all 151 epochs (0–150).
  * Smooth convergence starting immediately from high baseline performance (Val loss: 3.3835 at Epoch 0).
  * Minimum validation loss reached **3.3394** at Epoch 94 (improving upon Run 15's 3.3814).
* **Best Scores:**
  * **Depth:** $\delta_1 = 0.9084$ (Ep 90), $\delta_2 = 0.9888$ (Ep 90), $\delta_3 = 0.9976$ (Ep 90), $\text{RMSE} = 0.3778$m (Ep 90), $\text{AbsRel} = \mathbf{0.1009}$ (Ep 90), $\text{SqRel} = 0.0529$ (Ep 90), $\text{RMSElog} = 0.1286$ (Ep 90), $\text{log10} = 0.0435$ (Ep 90).
  * **Seg:** $\text{mIoU} = 0.5652$ (Ep 138), $\text{Dice} = 0.7011$ (Ep 138), $\text{Pixel Acc} = 0.7982$ (Ep 138).
  * **Min Val Loss:** **3.3394** (Ep 94).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.9082$, $\text{RMSE} = 0.3766\text{m}$, $\text{AbsRel} = 0.1014$, $\text{mIoU} = 0.5603$.
* **Key Takeaway:**
  * **Substantial Semantic Boost (+2.68 pp mIoU):** Fine-tuning the top 6 transformer blocks (early freeze) allowed the network to learn rich semantic class distinctions, driving mIoU from $0.5384 \to 0.5652$ and pixel accuracy to $79.82\%$.
  * **Depth Metric Preservation:** Depth accuracy stayed virtually identical to the frozen foundation baseline ($\delta_1 = 0.9084$ vs $0.9096$, $\text{RMSE} = 0.3778$m vs $0.3755$m), demonstrating that warm-starting with a conservative learning rate avoids catastrophic forgetting.

---

### Run 17: `dinov2-backbone-unfreeze` (Blocks 6–11 Fine-Tuning from Scratch)
* **Branch:** [`dinov2-vit-s-backbone`](file:///home/totallynotminh/Documents/TwinForge/checkpoints/dinov2-backbone-unfreeze) (Commit `2d8946c`)
* **What Changed vs Run 16:**
  * **Trained from Scratch with Early Freezing:** Although designated `unfreeze`, commit `2d8946c` retained `freeze_early=True` in `scripts/train.py`. Weight verification against baseline confirmed that `patch_embed` and blocks 0–5 remained completely frozen ($0.0$ weight diff), while blocks 6–11 were actively fine-tuned.
  * **Scratch vs. Warm-Start Ablation:** Crucially, while Run 16 warm-started from Run 15's frozen checkpoint, Run 17 was trained **entirely from scratch** (`No checkpoint detected`) using pretrained Depth Anything V2 ViT-S weights, with batch size 16 on Kaggle GPU.
  * **Dual Learning Rate Scheme:** `encoder_lr = 1e-5` for the trainable ViT blocks and `decoder_lr = 2e-4` for the decoder.
* **Parameters & MACs:** 45.74M Total | **34.34M Trainable (75.1%)** | **24.91 GMac** ($392 \times 518$).
* **Training Dynamics:**
  * Ran all 151 epochs (0–150).
  * Reached best depth metrics at Epoch 102 and best segmentation metrics at Epoch 133.
  * Minimum validation loss reached **3.3068** at Epoch 87—the lowest validation loss across the entire TwinForge project.
* **Best Scores:**
  * **Depth:** $\delta_1 = \mathbf{0.9103}$ (Ep 102), $\delta_2 = 0.9878$ (Ep 102), $\delta_3 = \mathbf{0.9977}$ (Ep 102), $\text{RMSE} = \mathbf{0.3683}$m (Ep 102), $\text{AbsRel} = 0.1012$ (Ep 102), $\text{SqRel} = \mathbf{0.0526}$ (Ep 102), $\text{RMSElog} = \mathbf{0.1279}$ (Ep 102), $\text{log10} = \mathbf{0.0432}$ (Ep 102).
  * **Seg:** $\text{mIoU} = \mathbf{0.5761}$ (Ep 133), $\text{Dice} = \mathbf{0.7122}$ (Ep 133), $\text{Pixel Acc} = \mathbf{0.8003}$ (Ep 133).
  * **Min Val Loss:** $\mathbf{3.3068}$ (Ep 87).
  * **Eigen Protocol Benchmark (Raw 480x640 Depth + Eigen Crop):** $\delta_1 = 0.9095$, $\text{RMSE} = \mathbf{0.3681\text{m}}$, $\text{AbsRel} = 0.1017$, $\text{mIoU} = \mathbf{0.5704}$.
* **Key Takeaway & Critical Critique:**
  * **All-Time Project Records Shattered Across Both Tasks:**
    * Semantic segmentation reached an extraordinary all-time project benchmark of **0.5761 mIoU** (+3.77 pp over frozen Run 15, and +16.35 pp over best ResNet Run 14).
    * Pixel accuracy officially crossed the **80% threshold** (**0.8003**) for the first time.
    * Depth RMSE broke below 37 cm down to **0.3683m** (36.83 cm), achieving the highest $\delta_1$ accuracy in the repository at **0.9103** (and **0.3681m** under standard Eigen protocol).
  * **Co-Adaptation Dominates:** Fine-tuning the upper transformer blocks (6–11) from scratch with early freeze allowed the self-attention heads to simultaneously adjust receptive fields for indoor boundaries and depth surfaces without degrading early geometric patch representations.

---

## 4. Cross-Architecture Synthesis & Key Discoveries

### Discovery 1: The Boundary Prediction Regularizer Effect (ResNet)
Comparing Runs 1, 2, and 4 demonstrated that high-frequency edge supervision acts as an indispensable spatial prior for CNN decoders:
* **Single-Task Depth (`isolated_depth`):** $\delta_1 = 0.7248$
* **2-Task Depth + Seg (`drop_head`):** $\delta_1 = 0.7171$
* **3-Task Depth + Seg + Bound (`clamped_kendall`):** $\delta_1 = 0.7353$

### Discovery 2: Spatial Orientation Was the Hidden Bottleneck
Prior to Run 11, ViT models were believed to suffer an inherent inductive bias deficiency on continuous depth estimation ($\delta_1 \approx 0.670$, $\text{RMSE} \approx 0.746$m). 
Fixing the transposed image loading bug in Run 11 immediately revealed that the ViT Cross-Attention architecture is actually **superior to ResNet on both tasks simultaneously**:
* **Depth $\delta_1$:** $0.7353$ (Best ResNet) $\rightarrow$ $0.7596$ (`vit-orientation-fixed`)
* **Depth RMSE:** $0.6268$m (Best ResNet) $\rightarrow$ $0.6108$m (`vit-orientation-fixed`)
* **Seg mIoU:** $0.2929$ (Best ResNet) $\rightarrow$ $0.4005$ (`vit-orientation-fixed`) $\rightarrow$ $0.4081$ (`vit-full-res`) $\rightarrow$ $0.4126$ (`vit-increase-regularization`)

### Discovery 3: Multi-Task Gradient Dynamics (Kendall vs. Fixed)
* **Kendall Uncertainty Loss:** Tended to dynamically overweight segmentation ($2.94:1.00$), leading to severe cross-entropy overconfidence and high validation loss.
* **Fixed Weighting ($1.0 : 1.0$):** Provided predictable gradient parity, improved depth RMSE, and stabilized multi-task optimization.

### Discovery 4: Resolution Scaling & Quadratic Compute Penalty
Moving from $384 \times 512$ to full native resolution ($480 \times 640$) in Run 13 increased scale 1 tokens from 1,344 to 2,120:
$$\left(\frac{2120}{1344}\right)^2 \approx 2.49\times \text{ attention operations}$$
This increased total GMacs from 38.52 to 60.20 due to quadratic attention overhead on the enlarged token grid. Yet depth accuracy degraded ($\Delta -0.0119$ $\delta_1$) and segmentation gained less than $1\%$ mIoU. **$384 \times 512$ (or $392 \times 518$) remains the optimal resolution pareto-frontier.**

### Discovery 5: Overfitting Control via Architectural Freezing & Regularization
In unconstrained runs (`vit-orientation-fixed`), train-val divergence reached an alarming gap of **3.99** (Train 1.00 vs Val 4.99). In Run 14 (`vit-increase-regularization`), combining early encoder freezing (`stem`, `layer1`, `layer2`), spatial/positional dropouts, label smoothing, and heavy augmentations successfully constrained the final gap to **2.15**, allowing segmentation mIoU to steadily climb to an all-time record of **0.4126**.

### Discovery 6: The Geometric vs. Photometric Augmentation Trade-off
Photometric perturbations (jittering brightness, contrast, saturation, hue) and spatial dropout provide pure regularization benefit with zero geometric distortion. In contrast, heavy geometric transforms (rotation up to $\pm 8^\circ$ and unscaled random cropping down to $0.6$) improve scale/rotation invariance for semantic segmentation but corrupt the perspective geometry and metric scale necessary for monocular depth estimation.

### Discovery 7: The Foundation Model Paradigm Shift (DINOv2 / Depth Anything V2 Backbone)
Transitioning from standard supervised ImageNet pretraining (ResNet50) to a foundation vision transformer pre-trained on multi-dataset depth representations (Depth Anything V2 / DINOv2 ViT-S) produced the largest single-step performance leap across the entire project lifespan:
* **Depth $\delta_1$:** $0.7596 \rightarrow \mathbf{0.9096}$ (+15.00 pp)
* **Depth RMSE:** $0.6108\text{m} \rightarrow \mathbf{0.3755\text{m}}$ (-38.5% error reduction)
* **Seg mIoU:** $0.4126 \rightarrow \mathbf{0.5384}$ (+12.58 pp)
* **Min Val Loss:** $4.3281 \rightarrow \mathbf{3.3814}$
* **Generalization Gap:** Compressed from $3.99 \rightarrow \mathbf{1.05}$
* **Compute (GMac):** Slashed from $38.52 \rightarrow \mathbf{24.91}$ GMac (-35.3%)

Crucially, freezing the foundation backbone completely bypassed the severe data-scarcity bottleneck of NYUv2 (795 training images). While ResNet encoders rapidly memorized pixel configurations, the frozen ViT-S backbone provided linearly separable, viewpoint-invariant tokens that allowed the `MultiHeadDecoder` to learn multi-task relationships without degrading depth or segmentation.

### Discovery 8: Foundation Encoder Fine-Tuning Dynamics (Frozen vs. Early-Freeze Warm-Start vs. Scratch)
Tracking the three Depth Anything V2 ViT-S experiments reveals a decisive progression:
* **Fully Frozen (Run 15, `dinov2-backbone`):** mIoU = `0.5384` | $\delta_1 = 0.9096$ | RMSE = `0.3755m` | Min Val Loss = `3.3814`
* **Early Freeze (Blocks 6–11 Fine-Tuned), Warm-Started (Run 16, `dino-v2-backbone-early-freeze`):** mIoU = `0.5652` | $\delta_1 = 0.9084$ | RMSE = `0.3778m` | Min Val Loss = `3.3394`
* **Early Freeze (Blocks 6–11 Fine-Tuned), from Scratch (Run 17, `dinov2-backbone-unfreeze`):** mIoU = $\mathbf{0.5761}$ | $\delta_1 = \mathbf{0.9103}$ | RMSE = $\mathbf{0.3683m}$ | Min Val Loss = $\mathbf{3.3068}$

**Core Insight:** Training upper transformer blocks (6–11) from scratch with a conservative learning rate ($1\times 10^{-5}$) allowed simultaneous co-adaptation of high-level encoder tokens and decoder heads. This completely outperformed the two-stage warm-start strategy by **+1.09 percentage points mIoU** and set all-time records for depth RMSE (**0.3683m**) and validation loss (**3.3068**). A completely unfrozen encoder (all 12 blocks) has not yet been benchmarked.

---

## 5. Architectural Recommendations for Next Experiments

1. **Decouple Projection Layer Learning Rates (`proj1`–`proj5` at `2e-4`):**
   * In Runs 16 and 17, `proj1`–`proj5` were throttled at `1e-5` alongside the ViT backbone. Training these adaptation layers at `2e-4` while keeping the ViT at `1e-5` removes the feature bottleneck and is expected to drive mIoU past `0.58+`.
2. **Multi-Scale Feature Adapter Optimization:**
   * Currently, stages `f1` through `f5` are projected via simple $1 \times 1$ convs from blocks `[2, 5, 8, 11]`. Replacing these with lightweight residual adapters or multi-kernel depthwise convolutions could enhance high-frequency edge detail for both tasks.
3. **Re-evaluate Cross-Task Interaction on Foundation Tokens:**
   * In earlier ResNet runs, cross-task SE-gates caused task interference. With high-quality DINOv2 representations, re-testing bidirectional attention or cross-task gating between depth and segmentation tokens may foster complementary feature exchange.
4. **Benchmark Larger Foundation Variants (ViT-B / ViT-L):**
   * The current ViT-S backbone operates at only 24.91 GMac. Testing Depth Anything V2 ViT-Base (dim=768) could evaluate whether scaling foundation model capacity unlocks further gains toward $\delta_1 > 0.93$ and mIoU $> 0.60$.
