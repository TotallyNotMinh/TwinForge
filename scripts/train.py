import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import NYUv2Dataset
from losses import SegmentLoss, DepthLoss
from models import TwinForge
from metrics import MultiTaskMetrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint")

args = parser.parse_args()

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
        return 0, 0.0, 0.0, 0  # Default values
    
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1

        best_depth_delta1 = checkpoint["best_depth_delta1"]
        best_seg_miou = checkpoint["best_seg_miou"]

        epochs_without_improvement = checkpoint["epochs_without_improvement"]

        print(f"Resume training:")
        print(f"  Starting epoch:              {start_epoch}")
        print(f"  Best depth δ1:               {best_depth_delta1:.4f}")
        print(f"  Best segmentation mIoU:      {best_seg_miou:.4f}")
        print(f"  Epochs without improvement:   {epochs_without_improvement}")
        print(f"  Current LR:                   {optimizer.param_groups[0]['lr']:.8f}")
            
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
        lines.append(f"  • Best Depth RMSE:     {best_records['depth_rmse'][0]:.4f} (Epoch {best_records['depth_rmse'][1]})")
        lines.append(f"  • Best Depth AbsRel:   {best_records['depth_absrel'][0]:.4f} (Epoch {best_records['depth_absrel'][1]})")
        
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
    # ============== Hyperparams ==============
    EPOCHS = 150
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = 41
    NUM_HEADS = 8
    TOKEN_DIM = 256
    patience = 25
    epochs_without_improvement = 0
    resize = (384, 512)
    encoder_lr = 1e-4
    decoder_lr = 2e-4
    depth_weight = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint_path
    start_epoch = 0

    best_seg_miou = 0
    best_depth_delta1 = 0
    best_records = {
        "depth_delta1": (0.0, None),
        "depth_delta2": (0.0, None),
        "depth_delta3": (0.0, None),
        "depth_rmse": (999.0, None),
        "depth_absrel": (999.0, None),
        "seg_miou": (0.0, None),
        "seg_dice": (0.0, None),
        "seg_pixel_acc": (0.0, None),
        "min_val_loss": (999.0, None)
    }

    # ============== Losses ==============
    crit_seg = SegmentLoss().to(device)
    crit_depth = DepthLoss().to(device)

    # ============== Load dataset ==============

    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    checkpoint_dir = "checkpoints/"

    train_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="train", resize=resize)
    val_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="val", resize=resize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ============== Model ==============
    
    model = TwinForge(NUM_CLASSES, NUM_HEADS, tok_dim=TOKEN_DIM, freeze=False).to(device)

    # ============== Optimizer and Schedulers ==============
    
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
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
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Train)", leave=False)
        for images, depths, labels in train_pbar:
            images = images.to(device, non_blocking=True)
            depths = depths.unsqueeze(1).float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred_seg, pred_depth = model(images)

                seg_loss = crit_seg(pred_seg, labels)
                depth_loss = crit_depth(pred_depth, depths, labels)

                tol_loss = seg_loss + depth_weight * depth_loss

            scaler.scale(tol_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += tol_loss.item()

            train_pbar.set_postfix({"loss": f"{tol_loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = running_train_loss / len(train_loader)

        # ============== Validation loop ==============
        running_val_loss = 0.0

        running_seg_loss = 0.0
        running_depth_loss = 0.0

        metrics.reset()
        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Val)  ", leave=False)
        with torch.no_grad():
            for images, depths, labels in val_pbar:
                images = images.to(device, non_blocking=True)
                depths = depths.unsqueeze(1).float().to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred_seg, pred_depth = model(images)

                    seg_loss = crit_seg(pred_seg, labels)
                    depth_loss = crit_depth(pred_depth, depths, labels)

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
            f"δ3: {total_depth['delta3']:.4f}"
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
            best_records["depth_rmse"] = (total_depth['rmse'], epoch)
            best_records["depth_absrel"] = (total_depth['abs_rel'], epoch)
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