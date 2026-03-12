import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_main_comparison_matrix(viz, data_dict, optimal_dist=None):
    """Main 3-panel comparison matrix."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])

    for label, data in data_dict.items():
        color = viz.method_colors.get(label, '#777777')
        
        valid_mask = data['depth'] >= 0
        totals = np.asarray(viz.get_total_moves(data)).astype(np.int32)
        opt_depths = np.asarray(data['depth'][valid_mask]).astype(np.int32)
        gap = np.maximum(totals - opt_depths, 0)

        # 1. Efficiency Gap PMF (%)
        c1 = np.bincount(gap)
        ax1.plot(np.arange(len(c1)), c1/len(gap)*100, '-o', label=label, color=color, markersize=4, linewidth=2)

        # 2. Total Move Distribution PMF (%)
        c2 = np.bincount(totals)
        ax2.plot(np.arange(len(c2)), c2/len(totals)*100, '-o', label=label, color=color, markersize=4, linewidth=2)

        # 3. Cumulative Distribution (%)
        ax3.step(np.sort(totals), np.linspace(0, 100, len(totals)), where='post', label=label, color=color, linewidth=2)

    # Overlays for Optimal Distribution
    if optimal_dist is not None:
        opt = optimal_dist[optimal_dist >= 0].astype(np.int32)
        c_opt = np.bincount(opt)
        
        # Overlay on PMF
        ax2.plot(np.arange(len(c_opt)), c_opt/len(opt)*100, marker='o', linestyle='--', linewidth=2.0, 
                 markersize=7, color='black', label="Optimal", zorder=20, 
                 markerfacecolor='white', markeredgewidth=1.5)
        
        # Overlay on Cumulative
        ax3.step(np.sort(opt), np.linspace(0, 100, len(opt)), where='post', 
                 color='black', linestyle='--', alpha=0.6, label="Optimal CDF", zorder=10)

    ax1.set_title("Efficiency Gap (Moves Above Optimal)", fontweight='bold')
    ax2.set_title("Total Move Distribution (PMF)", fontweight='bold')
    ax3.set_title("Cumulative Solve Capacity", fontweight='bold')
    
    ax1.legend(fontsize=9, loc='upper right') 
    ax2.legend(fontsize=9, loc='upper left')
    ax3.legend(fontsize=9, loc='lower right')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel("% of States")
        ax.grid(True, alpha=0.3, linestyle='--')
    
    ax3.set_xlabel("Total Moves (HTM)")
    viz.save_plot(fig, "method_comparison_matrix.png")

def plot_efficiency_gap_faceted(viz, data_dict):
    """Small-multiple histograms showing move waste per method."""
    n = len(data_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]
        valid_mask = data['depth'] >= 0
        totals = viz.get_total_moves(data)
        gap = np.maximum(totals - data['depth'][valid_mask], 0).astype(np.int32)
        
        color = viz.method_colors.get(label, '#777777')
        weights = np.ones_like(gap) / len(gap) * 100
        ax.hist(gap, bins=np.arange(gap.max()+2)-0.5, color=color, rwidth=0.8, edgecolor='black', weights=weights, alpha=0.8)
        
        ax.axvline(np.mean(gap), color='red', linestyle='--', label=f'Mean: {np.mean(gap):.2f}')
        ax.set_title(f"{label}\nEfficiency Gap", fontweight='bold')
        ax.set_xlabel("Extra Moves")
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    axes[0].set_ylabel("% of States")
    for k in range(n, len(axes)): axes[k].axis('off')
    viz.save_plot(fig, "efficiency_gap_faceted.png")

def plot_per_method_totals(viz, data_dict):
    """Faceted histograms of total moves with stats box."""
    n = len(data_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]
        totals = viz.get_total_moves(data).astype(np.int32)
        color = viz.method_colors.get(label, '#777777')
        
        weights = np.ones_like(totals) / len(totals) * 100
        ax.hist(totals, bins=np.arange(totals.max()+2)-0.5, color=color, rwidth=0.8, edgecolor='black', weights=weights)
        
        stats_text = f"Avg: {np.mean(totals):.2f}\nMed: {int(np.median(totals))}\nN: {len(totals):,}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_title(label)
        ax.set_xlabel("Total Moves")
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    axes[0].set_ylabel("% of States")
    for k in range(n, len(axes)): axes[k].axis('off')
    viz.save_plot(fig, "method_totals_faceted.png")

def plot_auf_grouped_bars(viz, data_dict):
    """AUF Distribution bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    auf_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'] 
    labels = list(data_dict.keys())
    x = np.arange(len(labels))
    width = 0.18

    for i, label in enumerate(labels):
        data = data_dict[label]
        valid = data['depth'] >= 0
        total_auf = (data['pre_auf'][valid] + data['mid_auf'][valid] + data['post_auf'][valid]).astype(np.int32)
        counts = np.bincount(np.clip(total_auf, 0, 3), minlength=4)
        pcts = counts / len(total_auf) * 100
        
        for val in range(4):
            if counts[val] <= 0: continue
            
            pos = i + (val-1.5)*width
            rect = ax.bar(pos, pcts[val], width, color=auf_colors[val], edgecolor='black', alpha=0.85)
            ax.text(pos, pcts[val], f"{int(counts[val]):,}\n({pcts[val]:.1f}%)", 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        mean_val = np.mean(total_auf)
        ax.plot(i, 0.5, marker='D', color='white', markeredgecolor='black', markersize=12, zorder=5)
        ax.text(i, 1.3, f"mean: {mean_val:.2f}", ha='center', fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% of States")
    ax.set_title("Total AUF Moves Distribution", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    legend_elements = [Line2D([0], [0], color=auf_colors[v], lw=6, label=f'{v} AUF moves') for v in range(4)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    viz.save_plot(fig, "auf_comparison.png")

def plot_first_vs_remainder_lines(viz, data_dict, optimal_dist=None):
    """Compares setup steps vs algorithmic remainder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for label, data in data_dict.items():
        color = viz.method_colors.get(label, '#777777')
        valid = data['depth'] >= 0
        
        step_names = [n for n in data.dtype.names if n not in ['depth', 'pre_auf', 'mid_auf', 'post_auf']]
        first_col = next((s for s in step_names if 'layer' in s.lower() or 'face' in s.lower()), step_names[0])
        
        first_vals = data[first_col][valid].astype(np.int32)
        totals = viz.get_total_moves(data).astype(np.int32)
        remainder = totals - first_vals - data['pre_auf'][valid].astype(np.int32)

        # Plot First Step
        c1 = np.bincount(first_vals)
        ax1.plot(np.arange(len(c1)), c1/len(first_vals)*100, '-o', label=f"{label} ({first_col.upper()})", color=color, markersize=4)
        
        # Plot Remainder
        c2 = np.bincount(remainder)
        ax2.plot(np.arange(len(c2)), c2/len(remainder)*100, '--o', label=label, color=color, markerfacecolor='white', markersize=4)

    ax1.set_title("First Step Distribution"); ax2.set_title("Remainder Distribution")
    for ax in [ax1, ax2]: 
        ax.set_ylabel("% of States")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='upper right')
    viz.save_plot(fig, "first_vs_remainder.png")

def plot_per_step_volume(viz, data_dict):
    """Grid of PMFs for individual solve steps."""
    all_steps = set()
    for d in data_dict.values():
        all_steps.update([n for n in d.dtype.names if n not in ['depth', 'pre_auf', 'mid_auf', 'post_auf']])
    all_steps = sorted(list(all_steps))

    cols = 2
    rows = (len(all_steps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, step in enumerate(all_steps):
        ax = axes[i]
        for label, data in data_dict.items():
            if step in data.dtype.names:
                vals = data[step][data[step] >= 0].astype(np.int32)
                if len(vals) == 0: continue
                counts = np.bincount(vals)
                color = viz.method_colors.get(label, '#777777')
                ax.plot(np.arange(len(counts)), counts/len(vals)*100, '-o', label=label, color=color, markersize=3, alpha=0.8)
        
        ax.set_title(f"Step: {step.upper()}", fontweight='bold')
        ax.set_ylabel("% of States")
        ax.grid(True, alpha=0.2, linestyle='--')
        if i == 0: ax.legend(fontsize=8, loc='upper right')

    for k in range(len(all_steps), len(axes)): axes[k].axis('off')
    plt.tight_layout()
    viz.save_plot(fig, "per_step_distribution.png")