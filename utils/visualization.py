#!/usr/bin/env python
"""
Cross-modality visualization for Phase 3 error analysis.

Loads JSON evaluation results from multiple modalities and generates
comparison charts: normalized confusion matrices, per-class F1/error
comparisons, and confusion flow diagrams.

Usage:
    python utils/visualization.py \
        --multimodal results/multimodal.json \
        --audio results/audio_only.json \
        --text results/text_only.json \
        --output results/comparison/

    # Only compare two modalities:
    python utils/visualization.py \
        --multimodal results/multimodal.json \
        --audio results/audio_only.json \
        --output results/comparison/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Styling ───────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 120,
})

LABEL_NAMES = {
    0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
    4: 'neutral', 5: 'other', 6: 'sad', 7: 'surprised',
}

MODALITY_COLORS = {
    'multimodal': '#2ecc71',
    'audio_only': '#3498db',
    'text_only': '#e74c3c',
}

MODALITY_LABELS = {
    'multimodal': 'Multimodal (Audio + Text)',
    'audio_only': 'Audio-only',
    'text_only': 'Text-only',
}


def load_result(path, name):
    """Load a single evaluation JSON and attach a modality name."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['_name'] = name
    data['_label'] = MODALITY_LABELS.get(name, name)
    return data


def plot_confusion_comparison(results_dict, output_dir):
    """
    3x2 grid: one row per modality, left=recall CM, right=precision CM.
    Skips modalities not present in results_dict.
    """
    modalities = [k for k in ['multimodal', 'audio_only', 'text_only'] if k in results_dict]
    if len(modalities) == 0:
        print("[visualization] No modality results to plot.")
        return

    n_rows = len(modalities)
    fig, axes = plt.subplots(n_rows, 2, figsize=(22, 9 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row_idx, mod in enumerate(modalities):
        data = results_dict[mod]
        detailed = data.get('detailed', {})
        class_names = [LABEL_NAMES[i] for i in range(8)]

        # Row-normalized (recall)
        cm_row = np.array(detailed.get('confusion_matrix_row_normalized',
                         detailed.get('confusion_matrix_raw', [[0]])))
        sns.heatmap(cm_row, annot=True, fmt='.3f' if cm_row.max() <= 1 else 'd',
                    cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=axes[row_idx][0],
                    cbar_kws={'label': 'Recall Rate', 'shrink': 0.8})
        axes[row_idx][0].set_title(f'{MODALITY_LABELS.get(mod, mod)} — Recall View')
        axes[row_idx][0].set_xlabel('Predicted')
        axes[row_idx][0].set_ylabel('True')

        # Column-normalized (precision)
        cm_col = np.array(detailed.get('confusion_matrix_col_normalized',
                         detailed.get('confusion_matrix_raw', [[0]])))
        sns.heatmap(cm_col, annot=True, fmt='.3f' if cm_col.max() <= 1 else 'd',
                    cmap='Oranges', xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=axes[row_idx][1],
                    cbar_kws={'label': 'Precision Rate', 'shrink': 0.8})
        axes[row_idx][1].set_title(f'{MODALITY_LABELS.get(mod, mod)} — Precision View')
        axes[row_idx][1].set_xlabel('Predicted')
        axes[row_idx][1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_all_modalities.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] Saved: {path}")


def plot_per_class_f1_comparison(results_dict, output_dir):
    """Grouped bar chart: per-class F1 across modalities."""
    modalities = [k for k in ['multimodal', 'audio_only', 'text_only'] if k in results_dict]
    if len(modalities) == 0:
        return

    class_names = [LABEL_NAMES[i] for i in range(8)]
    x = np.arange(len(class_names))
    width = 0.8 / len(modalities)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, mod in enumerate(modalities):
        data = results_dict[mod]
        per_class = data.get('per_class', {})
        f1_vals = [per_class.get(c, {}).get('f1', 0) for c in class_names]
        bars = ax.bar(x + idx * width, f1_vals, width,
                      label=MODALITY_LABELS.get(mod, mod),
                      color=MODALITY_COLORS.get(mod, '#95a5a6'),
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5,
                    rotation=90)

    ax.set_title('Per-Class F1-Score Comparison Across Modalities')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x + width * (len(modalities) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_f1_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] Saved: {path}")


def plot_error_rate_comparison(results_dict, output_dir):
    """Grouped bar chart: per-class error rate across modalities."""
    modalities = [k for k in ['multimodal', 'audio_only', 'text_only'] if k in results_dict]
    if len(modalities) == 0:
        return

    class_names = [LABEL_NAMES[i] for i in range(8)]
    x = np.arange(len(class_names))
    width = 0.8 / len(modalities)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, mod in enumerate(modalities):
        data = results_dict[mod]
        detailed = data.get('detailed', {})
        error_info = detailed.get('per_class_error_rate', {})
        err_vals = [error_info.get(c, {}).get('error_rate_pct', 0) for c in class_names]
        bars = ax.bar(x + idx * width, err_vals, width,
                      label=MODALITY_LABELS.get(mod, mod),
                      color=MODALITY_COLORS.get(mod, '#95a5a6'),
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, err_vals):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title('Per-Class Error Rate Comparison Across Modalities')
    ax.set_ylabel('Error Rate (%)')
    ax.set_xticks(x + width * (len(modalities) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'error_rate_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] Saved: {path}")


def plot_overall_metrics_comparison(results_dict, output_dir):
    """Bar chart comparing accuracy, macro F1, weighted F1 across modalities."""
    modalities = [k for k in ['multimodal', 'audio_only', 'text_only'] if k in results_dict]
    if len(modalities) == 0:
        return

    metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    metric_labels = ['Accuracy', 'Macro F1', 'Weighted F1']
    x = np.arange(len(metrics))
    width = 0.8 / len(modalities)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, mod in enumerate(modalities):
        data = results_dict[mod]
        vals = []
        acc = data.get('accuracy', 0)
        vals.append(acc * 100 if acc < 1 else acc)  # accuracy as percentage
        vals.append(data.get('macro_f1', 0))
        vals.append(data.get('weighted_f1', 0))
        ax.bar(x + idx * width, vals, width,
               label=MODALITY_LABELS.get(mod, mod),
               color=MODALITY_COLORS.get(mod, '#95a5a6'),
               edgecolor='white', linewidth=0.5)

    ax.set_title('Overall Metrics Comparison Across Modalities')
    ax.set_xticks(x + width * (len(modalities) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Value (% for Accuracy, Score for F1)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'overall_metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] Saved: {path}")


def plot_top_confusion_pairs_comparison(results_dict, output_dir):
    """
    Side-by-side horizontal bar charts showing top-5 confusion pairs
    for each modality, highlighting shared vs unique confusion patterns.
    """
    modalities = [k for k in ['multimodal', 'audio_only', 'text_only'] if k in results_dict]
    if len(modalities) == 0:
        return

    n_cols = len(modalities)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    for col_idx, mod in enumerate(modalities):
        data = results_dict[mod]
        detailed = data.get('detailed', {})
        pairs = detailed.get('top_confusion_pairs', [])[:5]
        labels = [f"{p['true']}\n→ {p['pred']}" for p in pairs]
        counts = [p['count'] for p in pairs]

        ax = axes[col_idx]
        bars = ax.barh(range(len(labels)), counts,
                       color=MODALITY_COLORS.get(mod, '#3498db'),
                       edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{MODALITY_LABELS.get(mod, mod)}\nTop-5 Confusions',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Misclassified Samples')
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.02 if counts else 0,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'top_confusion_pairs_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[visualization] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-modality visualization for SER error analysis"
    )
    parser.add_argument("--multimodal", default=None,
                        help="Path to multimodal evaluation JSON")
    parser.add_argument("--audio", default=None,
                        help="Path to audio-only evaluation JSON")
    parser.add_argument("--text", default=None,
                        help="Path to text-only evaluation JSON")
    parser.add_argument("--output", default="results/comparison",
                        help="Output directory for comparison charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results_dict = {}
    if args.multimodal and os.path.exists(args.multimodal):
        results_dict['multimodal'] = load_result(args.multimodal, 'multimodal')
        print(f"[visualization] Loaded multimodal: {args.multimodal}")
    if args.audio and os.path.exists(args.audio):
        results_dict['audio_only'] = load_result(args.audio, 'audio_only')
        print(f"[visualization] Loaded audio-only: {args.audio}")
    if args.text and os.path.exists(args.text):
        results_dict['text_only'] = load_result(args.text, 'text_only')
        print(f"[visualization] Loaded text-only: {args.text}")

    if len(results_dict) == 0:
        print("[visualization] ERROR: No valid result files provided.")
        return

    print(f"\n[visualization] Generating comparison charts for "
          f"{', '.join(results_dict.keys())}...\n")

    plot_confusion_comparison(results_dict, args.output)
    plot_per_class_f1_comparison(results_dict, args.output)
    plot_error_rate_comparison(results_dict, args.output)
    plot_overall_metrics_comparison(results_dict, args.output)
    plot_top_confusion_pairs_comparison(results_dict, args.output)

    print(f"\n[visualization] All charts saved to: {args.output}/")


if __name__ == "__main__":
    main()
