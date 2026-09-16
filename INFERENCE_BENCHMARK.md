# TwinForge Inference Speed Benchmark

## Environment & Hardware

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
- **Host OS:** Linux
- **PyTorch / CUDA:** 2.5.1 / CUDA 12.1
- **Conda Environment:** `twinforge`
- **Methodology:** Measured using hardware-synchronized CUDA Events (`torch.cuda.Event`) with 25 warmup iterations followed by 100 timed iterations. Peak VRAM tracked via `torch.cuda.max_memory_allocated()`.

---

## Model Architecture & Parameter Counts

| Model Variant | Component | Parameters |
| :--- | :--- | :---: |
| **ViT-S (`main`)** | Encoder Backbone (ViT-Small) | 23.56 M |
| **ViT-S (`main`)** | **Full TwinForge Model** | **45.74 M** |
| **ViT-B (`vit-b-encoder`)** | Encoder Backbone (ViT-Base) | 89.59 M |
| **ViT-B (`vit-b-encoder`)** | **Full TwinForge Model** | **111.76 M** |

---

## 1. Full TwinForge ViT-S Results (`main` branch)

### Target Resolution: `392 × 518` (Standard Training & Evaluation)

| Component | Precision | Batch Size | Latency (Mean ± Std) | Throughput (FPS) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ViT-S Encoder** | **FP16 (AMP)** | 1 | **9.96 ms** (±1.46) | **100.4 FPS** | 370.3 MB |
| **ViT-S Encoder** | FP32 | 1 | 29.05 ms (±3.67) | 34.4 FPS | 370.0 MB |
| **Full TwinForge ViT-S** | **FP16 (AMP)** | 1 | **25.02 ms** (±1.36) | **40.0 FPS** | 550.2 MB |
| **Full TwinForge ViT-S** | FP32 | 1 | 70.53 ms (±8.52) | 14.2 FPS | 518.5 MB |
| **ViT-S Encoder** | FP16 (AMP) | 8 | 57.84 ms (±5.83) | 138.3 FPS | 1012.9 MB |
| **Full TwinForge ViT-S** | FP16 (AMP) | 8 | 173.47 ms (±17.32) | 46.1 FPS | 2082.7 MB |

### Native NYUv2 Resolution: `480 × 640`

| Component | Precision | Batch Size | Latency (Mean ± Std) | Throughput (FPS) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ViT-S Encoder** | **FP16 (AMP)** | 1 | **14.19 ms** (±1.14) | **70.5 FPS** | 427.4 MB |
| **ViT-S Encoder** | FP32 | 1 | 45.19 ms (±2.57) | 22.1 FPS | 427.9 MB |
| **Full TwinForge ViT-S** | **FP16 (AMP)** | 1 | **39.57 ms** (±2.16) | **25.3 FPS** | 679.7 MB |
| **Full TwinForge ViT-S** | FP32 | 1 | 135.73 ms (±4.87) | 7.4 FPS | 660.6 MB |

---

## 2. Full TwinForge ViT-B Results (`vit-b-encoder` branch)

### Target Resolution: `392 × 518` (Standard Training & Evaluation)

| Component | Precision | Batch Size | Latency (Mean ± Std) | Throughput (FPS) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ViT-B Encoder** | **FP16 (AMP)** | 1 | **23.27 ms** (±2.64) | **43.0 FPS** | 888.0 MB |
| **ViT-B Encoder** | FP32 | 1 | 69.60 ms (±7.01) | 14.4 FPS | 883.3 MB |
| **Full TwinForge ViT-B** | **FP16 (AMP)** | 1 | **36.53 ms** (±0.57) | **27.4 FPS** | 1068.4 MB |
| **Full TwinForge ViT-B** | FP32 | 1 | 107.29 ms (±1.11) | 9.3 FPS | 1031.5 MB |
| **ViT-B Encoder** | FP16 (AMP) | 8 | 144.51 ms (±10.55) | 55.4 FPS | 1575.1 MB |
| **Full TwinForge ViT-B** | FP16 (AMP) | 8 | 272.70 ms (±27.42) | 29.3 FPS | 2596.4 MB |

### Native NYUv2 Resolution: `480 × 640`

| Component | Precision | Batch Size | Latency (Mean ± Std) | Throughput (FPS) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ViT-B Encoder** | **FP16 (AMP)** | 1 | **35.61 ms** (±2.66) | **28.1 FPS** | 951.8 MB |
| **ViT-B Encoder** | FP32 | 1 | 145.04 ms (±9.18) | 6.9 FPS | 949.1 MB |
| **Full TwinForge ViT-B** | **FP16 (AMP)** | 1 | **66.55 ms** (±8.13) | **15.0 FPS** | 1199.0 MB |
| **Full TwinForge ViT-B** | FP32 | 1 | 225.97 ms (±19.63) | 4.4 FPS | 1177.1 MB |

---

## 3. Head-to-Head Comparison: ViT-S vs. ViT-B

Input shape: `(1, 3, 392, 518)`:

| Component | Model | Params | FP32 Latency | FP16 Latency | FP16 FPS | Latency Ratio (ViT-B / ViT-S) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Encoder Alone** | ViT-S | 23.56 M | 29.05 ms | 9.96 ms | 100.4 FPS | 1.00x |
| **Encoder Alone** | ViT-B | 89.59 M | 69.60 ms | 23.27 ms | 43.0 FPS | **2.34x** |
| **Full TwinForge** | ViT-S | 45.74 M | 70.53 ms | 25.02 ms | 40.0 FPS | 1.00x |
| **Full TwinForge** | ViT-B | 111.76 M | 107.29 ms | 36.53 ms | 27.4 FPS | **1.46x** |

---

## Key Takeaways

1. **Full Model Overhead Ratio:** When stepping up from ViT-S to ViT-B, full model FP16 latency only increases by **1.46x** (25.02 ms vs. 36.53 ms), despite the backbone having **3.8x more parameters** (89.59 M vs. 23.56 M). This occurs because both variants share the same downstream multi-head transformer decoder resolution.
2. **Real-time Viability:** Both models achieve real-time performance (> 25 FPS) under FP16 on the mobile RTX 3060:
   - **Full TwinForge ViT-S:** **40.0 FPS** (25.0 ms)
   - **Full TwinForge ViT-B:** **27.4 FPS** (36.5 ms)
3. **VRAM Usage:** Peak VRAM for single-image inference is **550 MB** for ViT-S vs. **1068 MB** for ViT-B.
