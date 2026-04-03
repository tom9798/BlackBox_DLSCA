"""Generate compact, poster-quality convergence plots for all 5 successful models."""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = "/home/projects/sipl-prj10622/ASCAD/ASCAD_r/metrics/"

models = [
    {
        "name": "XOR Hard Sharing",
        "history": "history_training_basic_MTL_hard_sharing_all",
        "filename": "MTL_shared_mask_hard_shaing_convergence_compact",
    },
    {
        "name": "XOR Hard Sharing + ResNet",
        "history": "history_training_basic_MTL_hard_sharing_resnet_all",
        "filename": "MTL_shared_mask_hard_shaing_resnet_convergence_compact",
    },
    {
        "name": "XOR Low Sharing",
        "history": "history_training_basic_MTL_low_sharing_all",
        "filename": "MTL_shared_mask_low_shaing_convergence_compact",
    },
    {
        "name": "XOR Low Sharing + ResNet",
        "history": "history_training_basic_MTL_low_sharing_resnet_all",
        "filename": "MTL_shared_mask_low_shaing_resnet_convergence_compact",
    },
    {
        "name": "Bilinear Combiner (Exp F)",
        "history": "bv2_allF_history.pkl",
        "filename": "bv2_ascad_r_exp6_bilinear_skip_reg_convergence_compact",
    },
]


def compute_mean_acc(history, prefix=""):
    """Compute mean accuracy across all byte outputs per epoch."""
    acc_keys = [k for k in history.keys()
                if k.endswith("_accuracy") and k.startswith(prefix)]
    if not acc_keys:
        return None
    return np.mean([history[k] for k in acc_keys], axis=0)


for m in models:
    path = METRICS_DIR + m["history"]
    with open(path, "rb") as f:
        h = pickle.load(f)

    epochs = np.arange(1, len(h["loss"]) + 1)
    train_loss = h["loss"]
    val_loss = h["val_loss"]
    train_acc = compute_mean_acc(h, prefix="output_")
    val_acc = compute_mean_acc(h, prefix="val_output_")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Loss plot
    ax1.plot(epochs, train_loss, linewidth=2, color='#E74C3C', label='Train')
    ax1.plot(epochs, val_loss, linewidth=2, color='#4A90D9', label='Validation')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.set_title('Loss', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=13)

    # Accuracy plot
    if train_acc is not None and val_acc is not None:
        ax2.plot(epochs, train_acc * 100, linewidth=2, color='#E74C3C', label='Train')
        ax2.plot(epochs, val_acc * 100, linewidth=2, color='#4A90D9', label='Validation')
        ax2.set_ylabel('Mean Accuracy (%)', fontsize=15)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_title('Accuracy', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)

    fig.suptitle(m["name"], fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{m['filename']}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{m['filename']}.pdf", bbox_inches='tight')
    print(f"Saved: {m['filename']}.png / .pdf")
    plt.close()

print("Done!")
