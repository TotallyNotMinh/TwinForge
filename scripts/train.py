import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
import os
from data import NYUv2Dataset
from losses import SegmentLoss, DepthLoss, BoundaryLoss
from models import TwinForge
from metrics import MultiTaskMetrics

def train():
    W_SEG = 0.4
    W_DEPTH = 0.5
    W_BOUND = 0.2
    EPOCHS = 100
    BATCH_SIZE = 12
    NUM_CLASSES = 41
    patience = 10
    epochs_without_improvement = 0
    resize = (640, 480)
    encoder_lr = 1e-4
    decoder_lr = 1e-3

    crit_seg = SegmentLoss()
    crit_depth = DepthLoss()
    crit_bound = BoundaryLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    checkpoint_dir = "checkpoints/"

    train_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="train", resize=resize)
    val_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="val", resize=resize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = TwinForge(NUM_CLASSES, freeze=False).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
    ], weight_decay=1e-4)    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    val_losses = []
    train_losses = []
    best_val_loss = float("inf")

    metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)

    for epoch in range(EPOCHS + 1):
        print(f"Epoch {epoch}")
        model.train(True)
        running_train_loss = 0.0

        # Train loop
        for images, depths, labels, boundaries in train_loader:
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

                tol_loss = (W_SEG * seg_loss) + (W_DEPTH * depth_loss) + (W_BOUND * bound_loss) 

            scaler.scale(tol_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += tol_loss.item()

            # print(
            # f"(Train) Seg: {seg_loss.item():.4f} | "
            # f"(Train) Depth: {depth_loss.item():.4f} | "
            # f"(Train) Bound: {bound_loss.item():.4f}"
            # )

        scheduler.step()
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation loop
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

        with torch.no_grad():
            for images, depths, labels, boundaries in val_loader:
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

                    tol_loss = (W_SEG * seg_loss) + (W_DEPTH * depth_loss) + (W_BOUND * bound_loss) 

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

            avg_val_loss = running_val_loss / len(val_loader)

            # print(
            # f"(Val) Seg: {seg_loss.item():.4f} | "
            # f"(Val) Depth: {depth_loss.item():.4f} | "
            # f"(Val) Bound: {bound_loss.item():.4f}"
            # )

        for key in total_depth:
            total_depth[key] /= num_batches

        for key in total_seg:
            total_seg[key] /= num_batches

        for key in total_bound:
            total_bound[key] /= num_batches


        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

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