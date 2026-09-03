import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import NYUv2Dataset
from losses import SegmentLoss, DepthLoss, KendallMultiTaskLoss
from models import TwinForge
from metrics import MultiTaskMetrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint")

args = parser.parse_args()

def save_checkpoint(checkpoint_dir, checkpoint_name, epoch, model, kendall_loss, optimizer, scheduler, best_depth_delta1, best_seg_miou, epochs_without_improvement, scaler):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = {
        "epoch": epoch,

        # Model
        "model_state_dict": model.state_dict(),

        # Kendall uncertainty parameters
        "kendall_state_dict": kendall_loss.state_dict(),
        "kendall_log_vars": kendall_loss.log_vars.detach().cpu(),

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


def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler, kendall_loss):
    if checkpoint_path == None:
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

        kendall_loss.load_state_dict(checkpoint["kendall_state_dict"])

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
        print(f"  Kendall log_vars:             {kendall_loss.log_vars.detach().cpu().tolist()}")
        print(f"  Kendall weights:              {torch.exp(-kendall_loss.log_vars.detach()).cpu().tolist()}")
            
    return start_epoch, best_depth_delta1, best_seg_miou,  epochs_without_improvement

def log_cross_task_weights(model):
    decoder = model.decoder 
    heads = {
        "Seg": decoder.segment_head,
        "Depth": decoder.depth_head,
    }
    parts = []
    for name, head in heads.items():
        w = head.cross_task_weights.detach().cpu()
        parts.append(f"{name}: [{w[0]:.4f}, {w[1]:.4f}]")
    print("Cross-Task Weights -> " + " | ".join(parts))

def train():
    # ============== Hyperparams ==============
    EPOCHS = 150
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = 41
    patience = 25
    epochs_without_improvement = 0
    resize = (288, 384)
    encoder_lr = 1e-4
    decoder_lr = 1e-3
    kendall_lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint_path
    start_epoch = 0
    seg_boost_factor = 2.0

    best_seg_miou = 0
    best_depth_delta1 = 0

    # ============== Losses ==============
    crit_seg = SegmentLoss().to(device)
    crit_depth = DepthLoss().to(device)
    kendall_loss = KendallMultiTaskLoss(num_tasks=2).to(device)

    # ============== Load dataset ==============

    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    checkpoint_dir = "checkpoints/"

    train_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="train", resize=resize)
    val_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="val", resize=resize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ============== Model ==============

    model = TwinForge(NUM_CLASSES, freeze=False).to(device)

    # ============== Optimizer and Schedulers ==============
    
    cross_task_params = [
        p for name, p in model.named_parameters()
        if "cross_task_weights" in name
    ]
    other_decoder_params = [
        p for name, p in model.decoder.named_parameters()
        if "cross_task_weights" not in name
    ]

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": other_decoder_params, "lr": decoder_lr},
        {"params": kendall_loss.parameters(), "lr": kendall_lr, "weight_decay": 0.0},
        {"params": cross_task_params, "lr": 1e-4 , "weight_decay": 0.0}
    ], weight_decay=1e-3)    


    # Warm up with Linear scheduler then move to Consine Annealing
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 10, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[10])

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)

    # ============== Resume Training ==============
    start_epoch, best_depth_delta1, best_seg_miou, epochs_without_improvement = load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler, kendall_loss)

    # ============== Trainng and Validation loop ==============
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train(True)
        running_train_loss = 0.0

        # ============== Train loop ==============
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Train)", leave=False)
        for images, depths, labels, boundaries in train_pbar:
            images = images.to(device)
            depths = depths.unsqueeze(1).float().to(device)
            labels = labels.long().to(device)
            boundaries = boundaries.float().to(device)
            if boundaries.dim() == 3:
                boundaries = boundaries.unsqueeze(1)
            elif boundaries.dim() == 5:
                boundaries = boundaries.squeeze(1)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred_seg, pred_depth = model(images)

                seg_loss = crit_seg(pred_seg, labels) * seg_boost_factor
                depth_loss = crit_depth(pred_depth, depths, boundaries)

                tol_loss = kendall_loss([seg_loss, depth_loss])

            scaler.scale(tol_loss).backward()
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
        running_boundary_loss = 0.0

        total_depth = {
            "rmse": 0.0,
            "abs_rel": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
        }

        total_seg = {
            "miou": 0.0,
            "dice": 0.0,
            "pixel_acc": 0.0,
        }

        num_batches = 0

        model.eval()

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Val)  ", leave=False)
        with torch.no_grad():
            for images, depths, labels, boundaries in val_pbar:
                images = images.to(device)
                depths = depths.unsqueeze(1).float().to(device)
                labels = labels.long().to(device)
                boundaries = boundaries.float().to(device)
                if boundaries.dim() == 3:
                    boundaries = boundaries.unsqueeze(1)
                elif boundaries.dim() == 5:
                    boundaries = boundaries.squeeze(1)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred_seg, pred_depth, pred_bound = model(images)

                    seg_loss = crit_seg(pred_seg, labels) * seg_boost_factor 
                    depth_loss = crit_depth(pred_depth, depths, boundaries)

                    tol_loss = kendall_loss([seg_loss, depth_loss])

                result = metrics.compute(
                    pred_seg,
                    pred_depth,
                    labels,
                    depths,
                    boundaries
                )

                for key in total_depth:
                    total_depth[key] += result["depth"][key]

                for key in total_seg:
                    total_seg[key] += result["segmentation"][key]

                num_batches += 1

                running_val_loss += tol_loss.item()

                running_depth_loss += depth_loss
                running_seg_loss += seg_loss
                val_pbar.set_postfix({"val_loss": f"{tol_loss.item():.4f}"})


            avg_val_loss = running_val_loss / len(val_loader)

            avg_seg_loss = running_seg_loss / len(val_loader)
            avg_depth_loss = running_depth_loss / len(val_loader)
            avg_boundary_loss = running_boundary_loss / len(val_loader)

        # ============== Log ==============

        for key in total_depth:
            total_depth[key] /= num_batches

        for key in total_seg:
            total_seg[key] /= num_batches


        weights = (0.5 * torch.exp(-kendall_loss.log_vars.detach())).cpu().numpy()

        print(f"\n === Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} === ")
        print(f"Average Task Losses: Seg: {avg_seg_loss:.3f} | Depth: {avg_depth_loss:.3f} ")
        print(f"Kendall Weights -> Seg: {weights[0]:.3f} | Depth: {weights[1]:.3f}")
        print(f"Kendall log_vars -> Seg: {kendall_loss.log_vars[0]:.3f} | Depth: {kendall_loss.log_vars[1]:.3f}")
        log_cross_task_weights(model)

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
        save_checkpoint(checkpoint_dir, "checkpoint.pth", epoch, model, kendall_loss, optimizer, scheduler, best_depth_delta1, best_seg_miou, best_bound_f1, epochs_without_improvement, scaler)

        # ============== Save best checkpoint for each task ==============
        improved = False

        if total_depth['delta1'] > best_depth_delta1:
            best_depth_delta1 = total_depth['delta1']
            save_checkpoint(checkpoint_dir, "best_depth.pth", epoch, model, kendall_loss, optimizer, scheduler, best_depth_delta1, best_seg_miou, best_bound_f1, epochs_without_improvement, scaler)
            print(f"--> Saved new best DEPTH checkpoint.")
            improved = True

        if total_seg['miou'] > best_seg_miou:
            best_seg_miou = total_seg['miou'] 
            save_checkpoint(checkpoint_dir, "best_seg.pth", epoch, model, kendall_loss, optimizer, scheduler, best_depth_delta1, best_seg_miou, best_bound_f1, epochs_without_improvement, scaler)
            print(f"--> Saved new best SEG checkpoint.")
            improved = True

        if improved:
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience: 
                print("Early stop triggered.")
                break
        

if __name__ == "__main__":
    train()