import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import ScanNetVideoDataset
from losses import SegmentLoss, DepthLoss
from models import TwinForge
from metrics import MultiTaskMetrics
import argparse
import random
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per clip")
parser.add_argument("--data-path", type=str, default="data/scannet_frames_25k", help="Path to dataset")
parser.add_argument("--class-map-path", type=str, default="data/classMapping40.mat", help="Path to class mapping")
parser.add_argument("--train-split-path", type=str, default="data/scannetv2_train.txt", help="Train split path")
parser.add_argument("--val-split-path", type=str, default="data/scannetv2_val.txt", help="Val split path")
parser.add_argument("--num-epoch", type=int, default=100, help="Number of training epoch")

args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(checkpoint_dir, checkpoint_name, epoch, model, optimizer, scheduler, best_depth_delta1, best_seg_miou, epochs_without_improvement, scaler):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = {
        "epoch": epoch,

        # Model
        "model_state_dict": model.state_dict(),

        # Optimizer / scheduler
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        # Best metrics
        "best_depth_delta1": best_depth_delta1,
        "best_seg_miou": best_seg_miou,

        # Early stopping
        "epochs_without_improvement": epochs_without_improvement
    }
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_save_path)


def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler):
    if checkpoint_path is None:
        print("No checkpoint detected")
        return 0, 0.0, 0.0, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Adapt positional embeddings if resolution changed (e.g. 392x518 -> 378x504)
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if k in model_state and state_dict[k].shape != model_state[k].shape:
            if "pos_embed" in k and state_dict[k].dim() == 4:
                state_dict[k] = F.interpolate(
                    state_dict[k].float(),
                    size=model_state[k].shape[-2:],
                    mode="bicubic",
                    align_corners=False
                )
            else:
                del state_dict[k]

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    is_warmstart = not model_keys.issubset(ckpt_keys)

    if is_warmstart:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Warm-starting from pretrained checkpoint: {checkpoint_path}")
        print(f"  Loaded {len(model_keys) - len(missing)}/{len(model_keys)} layers.")
        print(f"  Newly initialized layers: {len(missing)} (temporal_head)")
        print(f"  Starting fresh training on ScanNet (Epoch 0)")
        return 0, 0.0, 0.0, 0
    else:
        model.load_state_dict(state_dict, strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", -1) + 1
        best_depth_delta1 = checkpoint.get("best_depth_delta1", 0.0)
        best_seg_miou = checkpoint.get("best_seg_miou", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        print(f"Resuming ScanNet training from Epoch {start_epoch}:")
        print(f"  Best depth δ1:               {best_depth_delta1:.4f}")
        print(f"  Best segmentation mIoU:      {best_seg_miou:.4f}")
        return start_epoch, best_depth_delta1, best_seg_miou, epochs_without_improvement
    
def save_summary(checkpoint_dir, best_records, current_epoch, total_epochs):
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_path = os.path.join(checkpoint_dir, "summary.txt")
    run_name = os.path.basename(os.path.normpath(checkpoint_dir))
    
    lines = []
    lines.append("=" * 65)
    lines.append(f"       TwinForge Training Summary: {run_name} (Epoch {current_epoch}/{total_epochs})")
    lines.append("=" * 65)
    
    if "depth_delta1" in best_records and best_records["depth_delta1"][1] is not None:
        lines.append(f"  • Best Depth δ1:       {best_records['depth_delta1'][0]:.4f} (Epoch {best_records['depth_delta1'][1]})")
        lines.append(f"  • Best Depth δ2:       {best_records['depth_delta2'][0]:.4f} (Epoch {best_records['depth_delta2'][1]})")
        lines.append(f"  • Best Depth δ3:       {best_records['depth_delta3'][0]:.4f} (Epoch {best_records['depth_delta3'][1]})")
        lines.append(f"  • Best Depth AbsRel:   {best_records['depth_absrel'][0]:.4f} (Epoch {best_records['depth_absrel'][1]})")
        if "depth_sqrel" in best_records and best_records["depth_sqrel"][1] is not None:
            lines.append(f"  • Best Depth SqRel:    {best_records['depth_sqrel'][0]:.4f} (Epoch {best_records['depth_sqrel'][1]})")
        lines.append(f"  • Best Depth RMSE:     {best_records['depth_rmse'][0]:.4f} (Epoch {best_records['depth_rmse'][1]})")
        if "depth_rmselog" in best_records and best_records["depth_rmselog"][1] is not None:
            lines.append(f"  • Best Depth RMSElog:  {best_records['depth_rmselog'][0]:.4f} (Epoch {best_records['depth_rmselog'][1]})")
        if "depth_log10" in best_records and best_records["depth_log10"][1] is not None:
            lines.append(f"  • Best Depth log10:    {best_records['depth_log10'][0]:.4f} (Epoch {best_records['depth_log10'][1]})")
        
    if "seg_miou" in best_records and best_records["seg_miou"][1] is not None:
        lines.append(f"  • Best Seg mIoU:       {best_records['seg_miou'][0]:.4f} (Epoch {best_records['seg_miou'][1]})")
        lines.append(f"  • Best Seg Dice:       {best_records['seg_dice'][0]:.4f} (Epoch {best_records['seg_dice'][1]})")
        lines.append(f"  • Best Seg Pixel Acc:  {best_records['seg_pixel_acc'][0]:.4f} (Epoch {best_records['seg_pixel_acc'][1]})")

    if "min_val_loss" in best_records and best_records["min_val_loss"][1] is not None:
        lines.append(f"  • Min Val Loss:        {best_records['min_val_loss'][0]:.4f} (Epoch {best_records['min_val_loss'][1]})")
    lines.append("=" * 65)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def train():
    set_seed(args.seed)

    # ============== Hyperparams ==============
    EPOCHS = args.num_epoch
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = 41
    NUM_HEADS = 8
    TOKEN_DIM = 256
    NUM_FRAMES = args.num_frames
    STRIDE = 1
    patience = 25
    epochs_without_improvement = 0
    resize = (392, 518) # Divisible by 14 as per ViT-S requirement 
    encoder_lr = 1e-5
    decoder_lr = 2e-4
    depth_weight = 1.4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint_path
    checkpoint_dir = args.checkpoint_dir
    start_epoch = 0

    best_seg_miou = 0
    best_depth_delta1 = 0
    best_records = {
        "depth_delta1": (0.0, None),
        "depth_delta2": (0.0, None),
        "depth_delta3": (0.0, None),
        "depth_absrel": (999.0, None),
        "depth_sqrel": (999.0, None),
        "depth_rmse": (999.0, None),
        "depth_rmselog": (999.0, None),
        "depth_log10": (999.0, None),
        "seg_miou": (0.0, None),
        "seg_dice": (0.0, None),
        "seg_pixel_acc": (0.0, None),
        "min_val_loss": (999.0, None)
    }

    # ============== Losses ==============
    crit_seg = SegmentLoss().to(device)
    crit_depth = DepthLoss().to(device)

    # ============== Load dataset ==============

    dataset_path = args.data_path

    train_dataset = ScanNetVideoDataset(
        root_dir=dataset_path,
        split="train",
        num_frames=NUM_FRAMES,
        stride=STRIDE,
        resize=resize,
        augment=True,
        split_file=args.train_split_path
    )    

    val_dataset = ScanNetVideoDataset(
        root_dir=dataset_path,
        split="val",
        num_frames=NUM_FRAMES,
        stride=STRIDE,
        resize=resize,
        augment=False,
        split_file=args.val_split_path
    ) 

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,      # B = 2 clips (total 2 * 4 = 8 frames per step)
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker
        
    )

    # ============== Model ==============
    
    model = TwinForge(NUM_CLASSES, NUM_HEADS, tok_dim=TOKEN_DIM, size=resize, freeze=False, freeze_early=True).to(device)

    # ============== Optimizer and Schedulers ==============
    
    vit_params = [p for p in model.encoder.vit.parameters() if p.requires_grad]
    proj_params = [p for name, p in model.encoder.named_parameters() if not name.startswith("vit") and p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": vit_params, "lr": encoder_lr, "weight_decay": 5e-3},
        {"params": proj_params + decoder_params, "lr": decoder_lr},
    ], weight_decay=1e-4)


    # Warm up with Linear scheduler then move to Consine Annealing
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 10, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[10])

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)

    # ============== Resume Training ==============
    start_epoch, best_depth_delta1, best_seg_miou, epochs_without_improvement = load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ============== Trainng and Validation loop ==============
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train(True)
        running_train_loss = 0.0

        # ============== Train loop ==============
        accum_steps = args.grad_accum_steps
        optimizer.zero_grad(set_to_none=True)  # Reset before loop

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Train)", leave=False)
        for batch_idx, (images, depths, labels) in enumerate(train_pbar):
            B, T = images.shape[0], images.shape[1]
            images = images.to(device, non_blocking=True)
        
            # Flatten (B, T, ...) -> (B * T, ...) to match model predictions
            depths = depths.view(B * T, 1, depths.shape[-2], depths.shape[-1]).float().to(device, non_blocking=True)
            labels = labels.view(B * T, labels.shape[-2], labels.shape[-1]).long().to(device, non_blocking=True)
        
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred_seg, pred_full, pred_half, pred_quarter = model(images)
        
                seg_loss = crit_seg(pred_seg, labels)

                # Full-resolution loss (SILog + gm_loss + smooth L1 + temporal)
                loss_full = crit_depth(pred_full, depths, labels, B=B, T=T)

                # Multi-scale intermediate supervision on downsampled ground truth
                depths_half = F.interpolate(depths, scale_factor=0.5, mode="nearest")
                depths_quarter = F.interpolate(depths, scale_factor=0.25, mode="nearest")
                loss_half = crit_depth.silog(pred_half, depths_half)
                loss_quarter = crit_depth.silog(pred_quarter, depths_quarter)

                depth_total_loss = loss_full + 0.5 * loss_half + 0.25 * loss_quarter
                tol_loss = seg_loss + depth_weight * depth_total_loss
                # Scale loss down by accumulation steps
                loss_to_backward = tol_loss / accum_steps

            # Backward accumulates gradients into .grad
            scaler.scale(loss_to_backward).backward()

            # Step optimizer only every accum_steps or on the last batch of the epoch
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Track full loss for logging (not the divided loss)
            running_train_loss += tol_loss.item()
            train_pbar.set_postfix({"loss": f"{tol_loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = running_train_loss / len(train_loader)

        # ============== Validation loop ==============
        if epoch % 2 == 0 or epoch == EPOCHS:
            running_val_loss = 0.0
            running_seg_loss = 0.0
            running_depth_loss = 0.0

            metrics.reset()
            model.eval()
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Val)  ", leave=False)
            with torch.no_grad():
                for images, depths, labels in val_pbar:
                    B, T = images.shape[0], images.shape[1]
                    images = images.to(device, non_blocking=True)
                    depths = depths.view(B * T, 1, depths.shape[-2], depths.shape[-1]).float().to(device, non_blocking=True)
                    labels = labels.view(B * T, labels.shape[-2], labels.shape[-1]).long().to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        pred_seg, pred_depth = model(images)

                        depths_loss = F.interpolate(depths, size=resize, mode="bilinear", align_corners=False) if depths.shape[-2:] != resize else depths
                        seg_loss = crit_seg(pred_seg, labels)
                        depth_loss = crit_depth(pred_depth, depths, labels, B=B, T=T)

                        tol_loss = seg_loss + depth_weight * depth_loss

                    metrics.update(
                        pred_seg,
                        pred_depth,
                        labels,
                        depths,
                    )

                    running_val_loss += tol_loss.item()
                    running_depth_loss += depth_loss.item()
                    running_seg_loss += seg_loss.item()
                    val_pbar.set_postfix({"val_loss": f"{tol_loss.item():.4f}"})

                avg_val_loss = running_val_loss / len(val_loader)
                avg_seg_loss = running_seg_loss / len(val_loader)
                avg_depth_loss = running_depth_loss / len(val_loader)

            # ============== Log ==============
            val_results = metrics.compute()
            total_depth = val_results["depth"]
            total_seg = val_results["segmentation"]

            print(f"\n === Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} === ")
            print(f"Validation Task Losses: Seg: {avg_seg_loss:.3f} | Depth: {avg_depth_loss:.3f} (depth_weight: {depth_weight})")

            print(
                f"Depth | "
                f"RMSE: {total_depth['rmse']:.4f} | "
                f"AbsRel: {total_depth['abs_rel']:.4f} | "
                f"δ1: {total_depth['delta1']:.4f} | "
                f"δ2: {total_depth['delta2']:.4f} | "
                f"δ3: {total_depth['delta3']:.4f} | "
                f"SqRel: {total_depth['sq_rel']:.4f} | "
                f"RMSElog: {total_depth['rmse_log']:.4f} | "
                f"log10: {total_depth['log10']:.4f}"
            )

            print(
                f"Seg   | "
                f"mIoU: {total_seg['miou']:.4f} | "
                f"Dice: {total_seg['dice']:.4f} | "
                f"Pixel Acc: {total_seg['pixel_acc']:.4f}"
            )

            # ============== Save current checkpoint ============== 
            save_checkpoint(checkpoint_dir, "checkpoint.pth", epoch, model, optimizer, scheduler, best_depth_delta1, best_seg_miou, epochs_without_improvement, scaler)

            # ============== Save best checkpoint for each task ==============
            improved = False

            if total_depth['delta1'] > best_depth_delta1:
                best_depth_delta1 = total_depth['delta1']
                best_records["depth_delta1"] = (total_depth['delta1'], epoch)
                best_records["depth_delta2"] = (total_depth['delta2'], epoch)
                best_records["depth_delta3"] = (total_depth['delta3'], epoch)
                best_records["depth_absrel"] = (total_depth['abs_rel'], epoch)
                best_records["depth_sqrel"] = (total_depth['sq_rel'], epoch)
                best_records["depth_rmse"] = (total_depth['rmse'], epoch)
                best_records["depth_rmselog"] = (total_depth['rmse_log'], epoch)
                best_records["depth_log10"] = (total_depth['log10'], epoch)
                save_checkpoint(checkpoint_dir, "best_depth.pth", epoch, model, optimizer, scheduler, best_depth_delta1, best_seg_miou, epochs_without_improvement, scaler)
                print(f"--> Saved new best DEPTH checkpoint.")
                improved = True

            if total_seg['miou'] > best_seg_miou:
                best_seg_miou = total_seg['miou'] 
                best_records["seg_miou"] = (total_seg['miou'], epoch)
                best_records["seg_dice"] = (total_seg['dice'], epoch)
                best_records["seg_pixel_acc"] = (total_seg['pixel_acc'], epoch)
                save_checkpoint(checkpoint_dir, "best_seg.pth", epoch, model, optimizer, scheduler, best_depth_delta1, best_seg_miou, epochs_without_improvement, scaler)
                print(f"--> Saved new best SEG checkpoint.")
                improved = True

            if avg_val_loss < best_records["min_val_loss"][0]:
                best_records["min_val_loss"] = (avg_val_loss, epoch)

            save_summary(checkpoint_dir, best_records, epoch, EPOCHS)

            if improved:
                epochs_without_improvement = 0
            else: 
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience: 
                    print("Early stop triggered.")
                    break
        
    print(best_records)

if __name__ == "__main__":
    train()