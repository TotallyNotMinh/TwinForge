import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data import NYUv2Dataset
from losses import SegmentLoss, DepthLoss, BoundaryLoss, KendallMultiTaskLoss
from models import TwinForge
from metrics import MultiTaskMetrics

def train():
    # ============== Hyperparams ==============
    EPOCHS = 150
    BATCH_SIZE = 12
    NUM_CLASSES = 41
    patience = 25
    epochs_without_improvement = 0
    resize = (640, 480)
    encoder_lr = 1e-4
    decoder_lr = 1e-3
    kendall_lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============== Losses ==============
    crit_seg = SegmentLoss().to(device)
    crit_depth = DepthLoss().to(device)
    crit_bound = BoundaryLoss().to(device)
    kendall_loss = KendallMultiTaskLoss().to(device)

    # ============== Load dataset ==============

    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    checkpoint_dir = "checkpoints/"

    train_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="train", resize=resize)
    val_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="val", resize=resize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ============== Model ==============

    model = TwinForge(NUM_CLASSES, freeze=False).to(device)

    # Load checkpoint:
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)        
        model.load_state_dict(checkpoint) 
        print("Successfully loaded model weights.")
        
    except FileNotFoundError:
        print("No checkpoint available.")

    # ============== Optimizer and Schedulers ==============
    
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
        {"params": kendall_loss.parameters(), "lr": kendall_lr, "weight_decay": 0.0}
    ], weight_decay=1e-4)    


    # Warm up with Linear scheduler then move to Consine Annealing
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 10, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[10])

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # ============== Trainng and Validation loop ==============
    val_losses = []
    train_losses = []
    best_val_loss = float("inf")

    metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)

    for epoch in range(EPOCHS + 1):
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
                pred_seg, pred_depth, pred_bound = model(images)

                seg_loss = crit_seg(pred_seg, labels)
                depth_loss = crit_depth(pred_depth, depths, pred_bound)
                bound_loss = crit_bound(pred_bound, boundaries)

                tol_loss = kendall_loss([seg_loss, depth_loss, bound_loss])

            scaler.scale(tol_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += tol_loss.item()
            train_pbar.set_postfix({"loss": f"{tol_loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = running_train_loss / len(train_loader)

        # ============== Validation loop ==============
        running_val_loss = 0.0

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

        total_bound = {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
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

                    seg_loss = crit_seg(pred_seg, labels)
                    depth_loss = crit_depth(pred_depth, depths, pred_bound)
                    bound_loss = crit_bound(pred_bound, boundaries)

                    tol_loss = kendall_loss([seg_loss, depth_loss, bound_loss])

                result = metrics.compute(
                    pred_seg,
                    pred_depth,
                    pred_bound,
                    labels,
                    depths,
                    boundaries
                )

                for key in total_depth:
                    total_depth[key] += result["depth"][key]

                for key in total_seg:
                    total_seg[key] += result["segmentation"][key]

                for key in total_bound:
                    total_bound[key] += result["boundary"][key]

                num_batches += 1

                running_val_loss += tol_loss.item()
                val_pbar.set_postfix({"val_loss": f"{tol_loss.item():.4f}"})


            avg_val_loss = running_val_loss / len(val_loader)

            # print(
            # f"(Val) Seg: {seg_loss.item():.4f} | "
            # f"(Val) Depth: {depth_loss.item():.4f} | "
            # f"(Val) Bound: {bound_loss.item():.4f}"
            # )

        # ============== Log ==============

        for key in total_depth:
            total_depth[key] /= num_batches

        for key in total_seg:
            total_seg[key] /= num_batches

        for key in total_bound:
            total_bound[key] /= num_batches


        weights = (0.5 * torch.exp(-kendall_loss.log_vars.detach())).cpu().numpy()
        print(f"\n === Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} === ")
        print(f"Effective Loss Weights -> Seg: {weights[0]:.3f} | Depth: {weights[1]:.3f} | Bound: {weights[2]:.3f}")

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

        print(
            f"Bound | "
            f"F1: {total_bound['f1']:.4f} | "
            f"Precision: {total_bound['precision']:.4f} | "
            f"Recall: {total_bound['recall']:.4f}"
)


        # --- Save Best Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best checkpoint to {save_path}")
            epochs_without_improvement = 0

        else: 
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience: 
                print("Early stop triggered.")
                break
        

if __name__ == "__main__":
    train()