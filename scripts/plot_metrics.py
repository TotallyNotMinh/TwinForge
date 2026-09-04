import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path: str):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    data = {
        "epoch": [],
        "timestamp": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "seg_loss": [],
        "depth_loss": [],
        "bound_loss": [],
        "kendall_weight_seg": [],
        "kendall_weight_depth": [],
        "kendall_weight_bound": [],
        "kendall_logvar_seg": [],
        "kendall_logvar_depth": [],
        "kendall_logvar_bound": [],
        "depth_rmse": [],
        "depth_absrel": [],
        "depth_delta1": [],
        "depth_delta2": [],
        "depth_delta3": [],
        "seg_miou": [],
        "seg_dice": [],
        "seg_pixel_acc": [],
        "bound_f1": [],
        "bound_prec": [],
        "bound_recall": []
    }

    epoch_header_re = re.compile(
        r"(?:([0-9.]+)s\s+\d+\s+)?===\s*Epoch\s*\[(\d+)/\d+\]\s*\|\s*Train\s*Loss:\s*([-\d.]+)\s*\|\s*Val\s*Loss:\s*([-\d.]+)\s*\|\s*LR:\s*([-\d.eE]+)\s*==="
    )
    task_losses_re = re.compile(r"Average\s+Task\s+Losses:\s*(.*)")
    kendall_weights_re = re.compile(r"Kendall\s+Weights\s*->\s*(.*)")
    kendall_logvars_re = re.compile(r"Kendall\s+log_vars\s*->\s*(.*)")
    depth_re = re.compile(
        r"Depth\s*\|\s*RMSE:\s*([-\d.]+)\s*\|\s*AbsRel:\s*([-\d.]+)\s*\|\s*[δ\?d]1:\s*([-\d.]+)\s*\|\s*[δ\?d]2:\s*([-\d.]+)\s*\|\s*[δ\?d]3:\s*([-\d.]+)"
    )
    seg_re = re.compile(
        r"Seg\s*\|\s*mIoU:\s*([-\d.]+)\s*\|\s*Dice:\s*([-\d.]+)\s*\|\s*Pixel\s*Acc:\s*([-\d.]+)"
    )
    bound_re = re.compile(
        r"Bound\s*\|\s*F1:\s*([-\d.]+)\s*\|\s*Precision:\s*([-\d.]+)\s*\|\s*Recall:\s*([-\d.]+)"
    )

    def parse_kv(text: str):
        # Parses key-value pairs formatted like "Key: val | Key2: val2"
        return dict(re.findall(r"(\w+):\s*([-\d.]+)", text))

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_epoch = epoch_header_re.search(line)
        if m_epoch:
            ts_str, ep_str, tr_str, val_str, lr_str = m_epoch.groups()
            current_epoch = int(ep_str)
            ts = float(ts_str) if ts_str else None

            # Collect subsequent lines for this epoch block
            block = []
            j = i + 1
            while j < len(lines) and not epoch_header_re.search(lines[j]) and "Early stop" not in lines[j]:
                block.append(lines[j])
                j += 1
            block_text = "\n".join(block)

            data["epoch"].append(current_epoch)
            data["timestamp"].append(ts)
            data["train_loss"].append(float(tr_str))
            data["val_loss"].append(float(val_str))
            data["lr"].append(float(lr_str))

            m_tloss = task_losses_re.search(block_text)
            if m_tloss:
                kvs = parse_kv(m_tloss.group(1))
                if "Seg" in kvs:
                    data["seg_loss"].append(float(kvs["Seg"]))
                if "Depth" in kvs:
                    data["depth_loss"].append(float(kvs["Depth"]))
                if "Bound" in kvs:
                    data["bound_loss"].append(float(kvs["Bound"]))

            m_kw = kendall_weights_re.search(block_text)
            if m_kw:
                kvs = parse_kv(m_kw.group(1))
                if "Seg" in kvs:
                    data["kendall_weight_seg"].append(float(kvs["Seg"]))
                if "Depth" in kvs:
                    data["kendall_weight_depth"].append(float(kvs["Depth"]))
                if "Bound" in kvs:
                    data["kendall_weight_bound"].append(float(kvs["Bound"]))

            m_kl = kendall_logvars_re.search(block_text)
            if m_kl:
                kvs = parse_kv(m_kl.group(1))
                if "Seg" in kvs:
                    data["kendall_logvar_seg"].append(float(kvs["Seg"]))
                if "Depth" in kvs:
                    data["kendall_logvar_depth"].append(float(kvs["Depth"]))
                if "Bound" in kvs:
                    data["kendall_logvar_bound"].append(float(kvs["Bound"]))

            m_dep = depth_re.search(block_text)
            if m_dep:
                data["depth_rmse"].append(float(m_dep.group(1)))
                data["depth_absrel"].append(float(m_dep.group(2)))
                data["depth_delta1"].append(float(m_dep.group(3)))
                data["depth_delta2"].append(float(m_dep.group(4)))
                data["depth_delta3"].append(float(m_dep.group(5)))

            m_seg = seg_re.search(block_text)
            if m_seg:
                data["seg_miou"].append(float(m_seg.group(1)))
                data["seg_dice"].append(float(m_seg.group(2)))
                data["seg_pixel_acc"].append(float(m_seg.group(3)))

            m_bnd = bound_re.search(block_text)
            if m_bnd:
                data["bound_f1"].append(float(m_bnd.group(1)))
                data["bound_prec"].append(float(m_bnd.group(2)))
                data["bound_recall"].append(float(m_bnd.group(3)))

            i = j
        else:
            i += 1

    return data


def plot_metrics(data: dict, run_name: str, output_path: str, show: bool = False):
    if not data["epoch"]:
        print(f"Error: No epoch metrics found in log file for run '{run_name}'.")
        return

    epochs = np.array(data["epoch"])
    
    # Calculate per-epoch duration if timestamps exist
    epoch_durations = []
    if all(ts is not None for ts in data["timestamp"]) and len(data["timestamp"]) > 1:
        ts_arr = np.array(data["timestamp"])
        epoch_durations = np.diff(ts_arr, prepend=ts_arr[0])
        if len(epoch_durations) > 1:
            epoch_durations[0] = epoch_durations[1]

    # Collect available plot functions
    plot_items = []

    # 1. Total Losses
    if data["train_loss"] and data["val_loss"]:
        def plot_total_loss(ax):
            ep = epochs[:len(data["train_loss"])]
            ax.plot(ep, data["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
            ax.plot(ep, data["val_loss"], label="Val Loss", color="#d62728", linewidth=2)
            best_val_idx = int(np.argmin(data["val_loss"]))
            ax.scatter(ep[best_val_idx], data["val_loss"][best_val_idx], color="#d62728", s=80, zorder=5,
                       label=f"Best Val: {data['val_loss'][best_val_idx]:.4f} (Ep {ep[best_val_idx]})")
            ax.set_title("Total Loss", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="best", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_total_loss)

    # 2. Individual Task Losses
    has_task_losses = any(len(data[k]) > 0 for k in ["seg_loss", "depth_loss", "bound_loss"])
    if has_task_losses:
        def plot_task_losses(ax):
            if data["seg_loss"]:
                ax.plot(epochs[:len(data["seg_loss"])], data["seg_loss"], label="Seg Loss", color="#2ca02c", linewidth=2)
            if data["depth_loss"]:
                ax.plot(epochs[:len(data["depth_loss"])], data["depth_loss"], label="Depth Loss", color="#ff7f0e", linewidth=2)
            if data["bound_loss"]:
                ax.plot(epochs[:len(data["bound_loss"])], data["bound_loss"], label="Bound Loss", color="#9467bd", linewidth=2)
            ax.set_title("Average Task Losses (Validation)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="best", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_task_losses)

    # 3. Kendall Uncertainty Weights
    has_kendall_weights = any(len(data[k]) > 0 for k in ["kendall_weight_seg", "kendall_weight_depth", "kendall_weight_bound"])
    if has_kendall_weights:
        def plot_kendall_weights(ax):
            if data["kendall_weight_seg"]:
                ax.plot(epochs[:len(data["kendall_weight_seg"])], data["kendall_weight_seg"], label="Seg Weight", color="#2ca02c", linewidth=2)
            if data["kendall_weight_depth"]:
                ax.plot(epochs[:len(data["kendall_weight_depth"])], data["kendall_weight_depth"], label="Depth Weight", color="#ff7f0e", linewidth=2)
            if data["kendall_weight_bound"]:
                ax.plot(epochs[:len(data["kendall_weight_bound"])], data["kendall_weight_bound"], label="Bound Weight", color="#9467bd", linewidth=2)
            ax.set_title("Kendall Multi-Task Weights (0.5 * exp(-s))", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.legend(loc="best", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_kendall_weights)

    # 4. Kendall Log-Vars
    has_kendall_logvars = any(len(data[k]) > 0 for k in ["kendall_logvar_seg", "kendall_logvar_depth", "kendall_logvar_bound"])
    if has_kendall_logvars:
        def plot_kendall_logvars(ax):
            if data["kendall_logvar_seg"]:
                ax.plot(epochs[:len(data["kendall_logvar_seg"])], data["kendall_logvar_seg"], label="Seg log_var (s1)", color="#2ca02c", linestyle="--", linewidth=2)
            if data["kendall_logvar_depth"]:
                ax.plot(epochs[:len(data["kendall_logvar_depth"])], data["kendall_logvar_depth"], label="Depth log_var (s2)", color="#ff7f0e", linestyle="--", linewidth=2)
            if data["kendall_logvar_bound"]:
                ax.plot(epochs[:len(data["kendall_logvar_bound"])], data["kendall_logvar_bound"], label="Bound log_var (s3)", color="#9467bd", linestyle="--", linewidth=2)
            ax.set_title("Kendall Uncertainty log_vars (s)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("log(sigma^2)")
            ax.legend(loc="best", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_kendall_logvars)

    # 5. Depth Accuracy (delta1, delta2, delta3)
    if data["depth_delta1"]:
        def plot_depth_acc(ax):
            ep = epochs[:len(data["depth_delta1"])]
            ax.plot(ep, data["depth_delta1"], label="δ1 (< 1.25)", color="#1f77b4", linewidth=2)
            if data["depth_delta2"]:
                ax.plot(ep, data["depth_delta2"], label="δ2 (< 1.25²)", color="#17becf", linewidth=1.8, linestyle="-.")
            if data["depth_delta3"]:
                ax.plot(ep, data["depth_delta3"], label="δ3 (< 1.25³)", color="#bcbd22", linewidth=1.8, linestyle=":")
            best_d1_idx = int(np.argmax(data["depth_delta1"]))
            ax.scatter(ep[best_d1_idx], data["depth_delta1"][best_d1_idx], color="#1f77b4", s=80, zorder=5,
                       label=f"Best δ1: {data['depth_delta1'][best_d1_idx]:.4f} (Ep {ep[best_d1_idx]})")
            ax.set_title("Depth Accuracy (Threshold Metrics)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_depth_acc)

    # 6. Depth Error Metrics (RMSE & AbsRel)
    if data["depth_rmse"] or data["depth_absrel"]:
        def plot_depth_err(ax):
            if data["depth_rmse"]:
                ep_rmse = epochs[:len(data["depth_rmse"])]
                ax.plot(ep_rmse, data["depth_rmse"], label="RMSE (m)", color="#e377c2", linewidth=2)
                best_rmse_idx = int(np.argmin(data["depth_rmse"]))
                ax.scatter(ep_rmse[best_rmse_idx], data["depth_rmse"][best_rmse_idx], color="#e377c2", s=70, zorder=5,
                           label=f"Best RMSE: {data['depth_rmse'][best_rmse_idx]:.4f} (Ep {ep_rmse[best_rmse_idx]})")
            if data["depth_absrel"]:
                ep_abs = epochs[:len(data["depth_absrel"])]
                ax.plot(ep_abs, data["depth_absrel"], label="AbsRel", color="#8c564b", linewidth=2)
                best_absrel_idx = int(np.argmin(data["depth_absrel"]))
                ax.scatter(ep_abs[best_absrel_idx], data["depth_absrel"][best_absrel_idx], color="#8c564b", s=70, zorder=5,
                           label=f"Best AbsRel: {data['depth_absrel'][best_absrel_idx]:.4f} (Ep {ep_abs[best_absrel_idx]})")
            ax.set_title("Depth Errors (RMSE & AbsRel)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Error")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_depth_err)

    # 7. Semantic Segmentation Metrics
    if data["seg_miou"]:
        def plot_seg_metrics(ax):
            ep = epochs[:len(data["seg_miou"])]
            ax.plot(ep, data["seg_miou"], label="mIoU", color="#2ca02c", linewidth=2.2)
            if data["seg_dice"]:
                ax.plot(ep, data["seg_dice"], label="Dice Score", color="#34bf49", linestyle="--", linewidth=1.8)
            if data["seg_pixel_acc"]:
                ax.plot(ep, data["seg_pixel_acc"], label="Pixel Acc", color="#1e7e34", linestyle=":", linewidth=1.8)
            best_miou_idx = int(np.argmax(data["seg_miou"]))
            ax.scatter(ep[best_miou_idx], data["seg_miou"][best_miou_idx], color="#2ca02c", s=80, zorder=5,
                       label=f"Best mIoU: {data['seg_miou'][best_miou_idx]:.4f} (Ep {ep[best_miou_idx]})")
            ax.set_title("Semantic Segmentation (mIoU / Dice / Pixel Acc)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_seg_metrics)

    # 8. Boundary Detection Metrics
    if data["bound_f1"]:
        def plot_bound_metrics(ax):
            ep = epochs[:len(data["bound_f1"])]
            ax.plot(ep, data["bound_f1"], label="Boundary F1", color="#9467bd", linewidth=2.2)
            if data["bound_prec"]:
                ax.plot(ep, data["bound_prec"], label="Precision", color="#c5b0d5", linestyle="--", linewidth=1.8)
            if data["bound_recall"]:
                ax.plot(ep, data["bound_recall"], label="Recall", color="#7b4173", linestyle=":", linewidth=1.8)
            best_f1_idx = int(np.argmax(data["bound_f1"]))
            ax.scatter(ep[best_f1_idx], data["bound_f1"][best_f1_idx], color="#9467bd", s=80, zorder=5,
                       label=f"Best F1: {data['bound_f1'][best_f1_idx]:.4f} (Ep {ep[best_f1_idx]})")
            ax.set_title("Boundary Detection (F1 / Precision / Recall)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        plot_items.append(plot_bound_metrics)

    num_plots = len(plot_items)
    if num_plots == 0:
        print(f"Error: No metrics available to plot for run '{run_name}'.")
        return

    # Layout calculation: 2 columns if >= 2 plots, else 1
    ncols = 2 if num_plots > 1 else 1
    nrows = (num_plots + ncols - 1) // ncols

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"TwinForge Training Metrics: {run_name}", fontsize=18, fontweight="bold", y=0.995)

    flat_axes = axes.flatten()
    for idx, plot_fn in enumerate(plot_items):
        plot_fn(flat_axes[idx])

    # Hide unused axes
    for idx in range(num_plots, len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved metrics visualization to: {output_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"       TwinForge Training Summary: {run_name} ({len(epochs)} Epochs)")
    print("=" * 65)
    if data["depth_delta1"]:
        best_d1_idx = int(np.argmax(data["depth_delta1"]))
        print(f"  • Best Depth δ1:       {data['depth_delta1'][best_d1_idx]:.4f} (Epoch {epochs[best_d1_idx]})")
    if data["depth_rmse"]:
        best_rmse_idx = int(np.argmin(data["depth_rmse"]))
        print(f"  • Best Depth RMSE:     {data['depth_rmse'][best_rmse_idx]:.4f} (Epoch {epochs[best_rmse_idx]})")
    if data["depth_absrel"]:
        best_absrel_idx = int(np.argmin(data["depth_absrel"]))
        print(f"  • Best Depth AbsRel:   {data['depth_absrel'][best_absrel_idx]:.4f} (Epoch {epochs[best_absrel_idx]})")
    if data["seg_miou"]:
        best_miou_idx = int(np.argmax(data["seg_miou"]))
        print(f"  • Best Seg mIoU:       {data['seg_miou'][best_miou_idx]:.4f} (Epoch {epochs[best_miou_idx]})")
    if data["seg_dice"]:
        print(f"  • Best Seg Dice:       {data['seg_dice'][int(np.argmax(data['seg_dice']))]:.4f}")
    if data["bound_f1"]:
        best_f1_idx = int(np.argmax(data["bound_f1"]))
        print(f"  • Best Bound F1:       {data['bound_f1'][best_f1_idx]:.4f} (Epoch {epochs[best_f1_idx]})")
    if data["bound_recall"]:
        print(f"  • Best Bound Recall:   {data['bound_recall'][int(np.argmax(data['bound_recall']))]:.4f}")
    if data["val_loss"]:
        best_val_idx = int(np.argmin(data["val_loss"]))
        print(f"  • Min Val Loss:        {data['val_loss'][best_val_idx]:.4f} (Epoch {epochs[best_val_idx]})")
    if len(epoch_durations) > 0:
        print(f"  • Avg Epoch Time:      {np.mean(epoch_durations):.1f}s (~{np.mean(epoch_durations)/60:.2f} min/epoch)")
        if data["timestamp"] and data["timestamp"][-1] is not None:
            print(f"  • Total Training Time: {data['timestamp'][-1]/3600:.2f} hours")
    print("=" * 65 + "\n")

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot TwinForge per-epoch training metrics.")
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to training log text file "
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output plot (default: <folder_name>.png)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively"
    )

    args = parser.parse_args()
    log_file = Path(args.log_path)

    if not log_file.exists():
        print(f"Error: Log file '{args.log_path}' not found.")
        return

    # Derive run name from parent folder
    folder_name = log_file.parent.name if log_file.parent.name not in ("", ".", "checkpoints") else log_file.stem
    output_path = args.output if args.output is not None else f"{folder_name}.png"

    data = parse_log(str(log_file))
    plot_metrics(data, run_name=folder_name, output_path=output_path, show=args.show)

if __name__ == "__main__":
    main()
