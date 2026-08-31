import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
import os
from data import NYUv2Dataset
from losses import SegmentLoss, DepthLoss, BoundaryLoss
from models import TwinForge

def train():
    W_SEG = 0.1
    W_DEPTH = 0.6
    W_BOUND = 0.3
    EPOCHS = 100
    BATCH_SIZE = 32
    NUM_CLASSES = 41
    patience = 10
    epochs_without_improvement = 0


    crit_seg = SegmentLoss()
    crit_depth = DepthLoss()
    crit_bound = BoundaryLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = "data/nyu_depth_v2_labeled.mat"
    class_map_path = "data/classMapping40.mat"
    checkpoint_dir = "checkpoints/"

    train_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="train")
    val_dataset = NYUv2Dataset(data_path=dataset_path, class_map_path=class_map_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = TwinForge(NUM_CLASSES).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    val_losses = []
    train_losses = []
    best_val_loss = float("inf")

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
                depth_loss = crit_depth(pred_depth, depths)
                bound_loss = crit_bound(pred_bound, boundaries)

                tol_loss = (W_SEG * seg_loss) + (W_DEPTH * depth_loss) + (W_BOUND * bound_loss) 

            scaler.scale(tol_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += tol_loss.item()

            print(
            f"(Train) Seg: {seg_loss.item():.4f} | "
            f"(Train) Depth: {depth_loss.item():.4f} | "
            f"(Train) Bound: {bound_loss.item():.4f}"
            )

        scheduler.step()
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation loop
        running_val_loss = 0.0

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
                    depth_loss = crit_depth(pred_depth, depths)
                    bound_loss = crit_bound(pred_bound, boundaries)

                    tol_loss = (W_SEG * seg_loss) + (W_DEPTH * depth_loss) + (W_BOUND * bound_loss) 

                running_val_loss += tol_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)

            print(
            f"(Val) Seg: {seg_loss.item():.4f} | "
            f"(Val) Depth: {depth_loss.item():.4f} | "
            f"(Val) Bound: {bound_loss.item():.4f}"
            )
            
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")


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