"""Generate poster-quality plots for DLSCA GeneralModel results.

Plot 1: GE convergence — mean rank vs. number of traces (3 models)
Plot 2: Per-byte single-trace top-1 accuracy comparison (3 models)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Data from detailed attack logs ──────────────────────────────────────────

# Bilinear Exp F (GeneralArch) — 5 experiments
expF_experiments = {
    0: {
        1: [2,13,3,1,4,4,1,2,1,4,9,1,2,3],
        2: [1,1,1,2,1,2,1,1,1,1,2,1,1,1],
        3: [1,1,1,1,1,2,1,1,1,1,1,1,1,1],
        4: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    1: {
        1: [12,1,2,4,4,10,12,4,4,1,2,1,2,9],
        2: [1,1,1,1,1,1,3,1,1,1,1,1,1,3],
        3: [1,1,1,1,1,1,2,1,1,1,1,1,1,1],
        4: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    2: {
        1: [1,2,1,1,1,2,1,2,4,1,2,2,1,2],
        2: [1,1,1,1,1,1,1,1,1,1,1,2,1,1],
        3: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    3: {
        1: [1,5,1,3,1,1,1,4,4,1,1,1,3,3],
        2: [1,1,1,2,1,1,1,1,1,1,1,1,1,2],
        3: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    4: {
        1: [2,6,2,1,2,1,1,1,1,2,2,2,1,1],
        2: [1,2,1,1,1,1,1,1,1,2,1,1,1,1],
        3: [1,2,1,1,1,1,1,1,1,2,1,1,1,1],
        4: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
}

# XOR Low Sharing + ResNet (base model, best XOR variant) — 5 experiments
xor_low_resnet_experiments = {
    0: {
        1: [1,1,1,1,4,1,1,3,1,1,1,1,11,1],
        2: [1,1,3,1,1,1,1,1,1,1,1,1,3,1],
        3: [1,1,1,1,1,1,1,1,1,1,1,1,2,1],
        4: [1,1,1,1,2,1,1,1,1,2,1,1,1,1],
        5: [1,1,1,1,2,1,1,1,1,1,1,1,1,1],
        6: [1,1,1,1,2,1,1,1,1,2,1,1,1,1],
        7: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    1: {
        1: [14,4,19,2,2,1,2,9,43,1,2,3,16,5],
        2: [2,1,1,1,1,1,1,1,2,1,2,1,3,2],
        3: [1,1,1,1,1,1,1,1,2,1,1,1,2,2],
        4: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        5: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        6: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        7: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        8: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        9: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        10: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        11: [1,1,1,1,1,1,1,1,2,1,1,1,1,2],
        12: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        13: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        14: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    2: {
        1: [1,2,1,7,1,2,2,1,1,5,1,5,7,1],
        2: [1,1,2,1,1,2,1,1,1,7,1,1,1,1],
        3: [1,1,2,1,1,2,1,1,1,1,1,1,1,1],
        4: [1,1,1,1,1,2,1,1,1,1,1,1,1,1],
        5: [1,1,1,1,1,2,1,1,1,1,1,1,1,1],
        6: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    3: {
        1: [11,4,5,3,1,22,1,21,5,3,3,3,1,1],
        2: [5,2,2,1,1,3,1,1,2,2,1,2,1,1],
        3: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    4: {
        1: [17,34,15,11,6,22,10,6,8,4,28,5,4,5],
        2: [1,3,2,3,2,4,5,1,2,4,4,1,3,1],
        3: [1,4,2,3,6,9,6,1,6,3,5,4,4,1],
        4: [1,1,1,2,1,7,1,1,1,1,1,1,1,1],
        5: [1,1,1,1,1,5,1,1,1,1,1,1,1,1],
        6: [1,1,1,1,1,2,1,1,1,2,1,1,1,1],
        7: [1,1,1,1,1,2,1,1,1,1,1,1,1,1],
        8: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
}

# XOR Hard Sharing + ResNet — 5 experiments
xor_hard_resnet_experiments = {
    0: {
        1: [16,2,3,13,15,11,5,9,6,1,2,1,2,1],
        2: [12,2,3,3,14,3,1,6,2,1,1,1,2,1],
        3: [4,2,3,1,7,1,2,1,2,1,1,1,2,1],
        4: [2,1,2,2,7,1,3,1,1,1,1,1,1,1],
        5: [1,1,2,1,1,2,3,1,1,1,1,1,1,1],
        6: [1,1,2,2,1,2,3,1,1,1,1,1,1,1],
        7: [1,1,2,1,1,2,3,1,1,1,1,1,1,1],
        8: [1,1,1,1,1,2,3,1,1,1,1,1,1,1],
        9: [1,1,2,1,1,2,2,1,1,1,1,1,1,1],
        10: [1,1,2,1,1,2,2,1,1,1,1,1,1,1],
        11: [1,1,1,1,1,2,3,1,1,1,1,1,1,1],
        12: [1,1,1,1,1,1,3,1,1,1,1,1,1,1],
        13: [1,1,1,1,1,1,2,1,1,1,1,1,1,1],
        14: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    1: {
        1: [2,1,1,1,1,20,1,8,6,1,1,1,5,3],
        2: [1,1,1,1,1,2,1,10,5,1,1,1,3,2],
        3: [1,1,1,1,1,1,1,1,4,1,1,1,1,2],
        4: [1,1,1,1,1,1,1,1,5,1,1,2,1,1],
        5: [1,1,1,1,1,1,1,1,6,1,1,1,1,1],
        6: [1,1,1,1,1,1,1,1,6,1,1,1,1,1],
        7: [1,1,1,1,1,1,1,1,5,1,1,1,1,1],
        8: [1,1,1,1,1,1,1,1,6,1,1,1,1,1],
        9: [1,1,1,1,1,1,1,1,6,1,1,1,1,1],
        10: [1,1,1,1,1,1,1,1,3,1,1,1,1,1],
        11: [1,1,1,1,1,1,1,1,3,1,1,1,1,1],
        12: [1,1,1,1,1,1,1,1,3,1,1,1,1,1],
        13: [1,1,1,1,1,1,1,1,3,1,1,1,1,1],
        14: [1,1,1,1,1,1,1,1,2,1,1,1,1,1],
        15: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    2: {
        1: [3,11,10,32,3,33,14,15,7,4,4,20,21,8],
        2: [1,10,4,15,1,15,14,1,1,4,1,3,8,1],
        3: [1,7,3,5,1,3,4,1,1,1,1,2,2,1],
        4: [1,4,3,5,2,2,4,1,1,1,1,2,3,1],
        5: [1,4,1,1,1,1,1,1,2,1,3,1,2,1],
        6: [1,4,1,1,2,1,1,1,1,1,2,1,2,1],
        7: [1,4,1,1,2,1,1,1,2,1,2,1,2,1],
        8: [1,4,1,1,3,1,1,1,1,1,1,1,2,1],
        9: [1,4,1,1,2,1,1,1,1,1,1,1,2,1],
        10: [1,2,1,1,2,1,1,1,1,2,1,1,2,1],
        11: [1,2,2,1,2,1,1,1,1,1,1,1,2,1],
        12: [1,2,1,1,2,1,1,1,1,1,1,1,1,1],
        13: [1,2,1,1,2,1,1,1,1,1,1,1,1,1],
        14: [1,1,1,1,2,1,1,1,1,1,1,1,1,1],
        15: [1,1,1,1,2,1,1,1,1,1,1,1,1,1],
        16: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    3: {
        1: [10,5,4,5,12,3,1,1,1,1,3,3,5,2],
        2: [2,2,3,1,2,4,6,1,1,3,2,1,1,1],
        3: [1,2,2,1,1,1,3,1,1,2,2,1,1,1],
        4: [1,2,1,1,1,1,2,1,1,2,2,1,1,1],
        5: [1,3,1,1,1,1,1,1,1,1,2,1,1,1],
        6: [1,3,1,1,1,1,1,1,1,1,2,1,1,1],
        7: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
    4: {
        1: [7,7,11,10,1,27,19,10,3,1,41,4,3,2],
        2: [2,12,2,5,2,11,2,5,2,2,3,1,1,1],
        3: [1,5,4,2,3,9,1,7,2,4,3,1,1,1],
        4: [1,5,1,2,1,8,2,2,2,4,1,1,1,1],
        5: [1,5,1,1,1,5,1,2,1,4,1,1,1,1],
        6: [1,3,1,1,1,1,1,2,1,3,1,1,1,1],
        7: [1,2,1,1,1,2,1,2,1,3,1,1,1,1],
        8: [1,2,1,1,1,1,2,2,1,3,1,1,1,1],
        9: [1,2,1,1,1,1,1,2,1,3,1,1,1,1],
        10: [1,2,1,1,1,1,1,2,1,3,1,1,1,1],
        11: [1,1,1,1,1,1,1,1,1,3,1,1,1,1],
        12: [1,1,1,1,1,1,1,1,1,3,1,1,1,1],
        13: [1,1,1,1,1,1,1,1,1,2,1,1,1,1],
        14: [1,1,1,1,1,1,1,1,1,2,1,1,1,1],
        15: [1,1,1,1,1,1,1,1,1,2,1,1,1,1],
        16: [1,1,1,1,1,1,1,1,1,2,1,1,1,1],
        17: [1,2,1,1,1,1,1,1,1,2,1,1,1,1],
        18: [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    },
}


def compute_mean_ge_curve(experiments, max_traces=20):
    """Compute mean GE (log2 of mean rank) per trace across experiments."""
    all_means = []
    for exp_data in experiments.values():
        means = []
        max_t = max(exp_data.keys())
        for t in range(1, max_traces + 1):
            if t in exp_data:
                means.append(np.mean(exp_data[t]))
            elif t > max_t:
                means.append(1.0)  # already recovered
            else:
                means.append(1.0)
        all_means.append(means)

    all_means = np.array(all_means)
    mean_rank = np.mean(all_means, axis=0)
    # Convert to log2 GE (rank 1 -> GE=0, rank 128 -> GE=7)
    ge = np.log2(np.clip(mean_rank, 1, None))
    return ge


# ─── Plot 1: GE Convergence Curve (full) ─────────────────────────────────────

max_traces = 20
traces_x = np.arange(1, max_traces + 1)

ge_expF = compute_mean_ge_curve(expF_experiments, max_traces)
ge_xor_low = compute_mean_ge_curve(xor_low_resnet_experiments, max_traces)
ge_xor_hard = compute_mean_ge_curve(xor_hard_resnet_experiments, max_traces)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(traces_x, ge_xor_hard, 'o-', color='#888888', linewidth=2, markersize=6,
        label='XOR Hard Sharing + ResNet', zorder=2)
ax.plot(traces_x, ge_xor_low, 's-', color='#4A90D9', linewidth=2, markersize=6,
        label='XOR Low Sharing + ResNet', zorder=3)
ax.plot(traces_x, ge_expF, 'D-', color='#E74C3C', linewidth=2.5, markersize=7,
        label='Bilinear Combiner (Exp F, ours)', zorder=4)

# Success threshold
ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7)
ax.text(max_traces - 0.3, 0.12, 'Full recovery (GE = 0)', ha='right',
        fontsize=9, color='green', alpha=0.8)

# Random guess baseline
ax.axhline(y=7, color='grey', linestyle=':', linewidth=1, alpha=0.5)
ax.text(max_traces - 0.3, 7.15, 'Random guess (GE = 7)', ha='right',
        fontsize=9, color='grey', alpha=0.7)

ax.set_xlabel('Number of attack traces', fontsize=13)
ax.set_ylabel('Guessing Entropy (log$_2$)', fontsize=13)
ax.set_title('Key Recovery: Guessing Entropy vs. Number of Traces\n(ASCAD_r, boolean-masked AES, mean over 5 experiments)',
             fontsize=14, fontweight='bold')
ax.set_xlim(0.5, max_traces + 0.5)
ax.set_ylim(-0.3, 4.5)
ax.set_xticks(traces_x)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("poster_ge_convergence.png", dpi=300, bbox_inches='tight')
plt.savefig("poster_ge_convergence.pdf", bbox_inches='tight')
print("Saved: poster_ge_convergence.png / .pdf")
plt.close()


# ─── Plot 1b: GE Convergence Curve (compact, poster version) ─────────────────

max_traces_compact = 9
traces_x_c = np.arange(1, max_traces_compact + 1)

ge_expF_c = ge_expF[:max_traces_compact]
ge_xor_low_c = ge_xor_low[:max_traces_compact]
ge_xor_hard_c = ge_xor_hard[:max_traces_compact]

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.plot(traces_x_c, ge_xor_hard_c, 'o-', color='#888888', linewidth=2.5, markersize=9,
        label='XOR Hard Sharing + ResNet', zorder=2)
ax.plot(traces_x_c, ge_xor_low_c, 's-', color='#4A90D9', linewidth=2.5, markersize=9,
        label='XOR Low Sharing + ResNet', zorder=3)
ax.plot(traces_x_c, ge_expF_c, 'D-', color='#E74C3C', linewidth=3, markersize=10,
        label='Bilinear Combiner (Exp F, ours)', zorder=4)

# Success threshold
ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(max_traces_compact - 0.2, 0.75, 'Full recovery (GE = 0)', ha='right',
        fontsize=15, color='green', fontweight='bold', alpha=0.9)

ax.set_xlabel('Number of Attack Traces', fontsize=16)
ax.set_ylabel('Guessing Entropy (log$_2$)', fontsize=16)
ax.set_title('Key Recovery: Guessing Entropy vs. Number of Traces',
             fontsize=16, fontweight='bold')
ax.set_xlim(0.5, max_traces_compact + 0.5)
ax.set_ylim(-0.3, 3.5)
ax.set_xticks(traces_x_c)
ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("poster_ge_convergence_compact.png", dpi=300, bbox_inches='tight')
plt.savefig("poster_ge_convergence_compact.pdf", bbox_inches='tight')
print("Saved: poster_ge_convergence_compact.png / .pdf")
plt.close()


# ─── Plot 2: Per-Byte Top-1 Accuracy Comparison ──────────────────────────────

bytes_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
byte_labels = [str(b) for b in bytes_list]

# Top-1% from detailed attack logs
top1_expF =      [39.1, 23.2, 42.9, 52.8, 40.2, 32.6, 30.7, 43.8, 29.5, 48.9, 22.8, 34.0, 52.7, 40.6]
top1_xor_low =   [27.5, 25.1, 29.5, 30.1, 24.9, 21.8, 29.0, 25.4, 30.8, 27.5, 25.7, 29.6, 27.0, 35.3]
top1_xor_hard =  [18.1, 16.8, 21.0, 22.6, 17.0, 14.9, 20.3, 17.8, 22.6, 17.9, 21.0, 20.3, 22.7, 36.5]

x = np.arange(len(bytes_list))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5.5))

bars1 = ax.bar(x - width, top1_xor_hard, width, label='XOR Hard Sharing + ResNet',
               color='#BBBBBB', edgecolor='#888888', linewidth=0.5, zorder=2)
bars2 = ax.bar(x, top1_xor_low, width, label='XOR Low Sharing + ResNet',
               color='#7CB9E8', edgecolor='#4A90D9', linewidth=0.5, zorder=3)
bars3 = ax.bar(x + width, top1_expF, width, label='Bilinear Combiner (Exp F, ours)',
               color='#F1948A', edgecolor='#E74C3C', linewidth=0.5, zorder=4)

ax.set_xlabel('AES Key Byte Index', fontsize=13)
ax.set_ylabel('Single-Trace Top-1 Accuracy (%)', fontsize=13)
ax.set_title('Per-Byte Single-Trace Attack Accuracy\n(ASCAD_r, boolean-masked AES, 10,000 test traces)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(byte_labels)
ax.set_ylim(0, 62)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)
ax.tick_params(labelsize=11)

# Add mean lines
mean_expF = np.mean(top1_expF)
mean_xor_low = np.mean(top1_xor_low)
mean_xor_hard = np.mean(top1_xor_hard)
ax.axhline(y=mean_expF, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.6)
ax.axhline(y=mean_xor_low, color='#4A90D9', linestyle='--', linewidth=1, alpha=0.6)
ax.axhline(y=mean_xor_hard, color='#888888', linestyle='--', linewidth=1, alpha=0.6)

ax.text(len(bytes_list) - 0.6, mean_expF + 0.8, f'mean={mean_expF:.1f}%',
        fontsize=9, color='#E74C3C', ha='right')
ax.text(len(bytes_list) - 0.6, mean_xor_low + 0.8, f'mean={mean_xor_low:.1f}%',
        fontsize=9, color='#4A90D9', ha='right')
ax.text(len(bytes_list) - 0.6, mean_xor_hard + 0.8, f'mean={mean_xor_hard:.1f}%',
        fontsize=9, color='#888888', ha='right')

plt.tight_layout()
plt.savefig("poster_perbyte_accuracy.png", dpi=300, bbox_inches='tight')
plt.savefig("poster_perbyte_accuracy.pdf", bbox_inches='tight')
print("Saved: poster_perbyte_accuracy.png / .pdf")
plt.close()

print("Done!")
