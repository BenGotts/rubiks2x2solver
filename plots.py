import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_main_comparison_matrix(viz, data_dict, optimal_dist=None):
    """Main 3-panel comparison matrix."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])

    gap_rows, total_rows, cdf_rows = [], [], []

    for label, data in data_dict.items():
        color = viz.method_colors.get(label, '#777777')
        
        valid_mask = data['depth'] >= 0
        totals = np.asarray(viz.get_total_moves(data)).astype(np.int32)
        opt_depths = np.asarray(data['depth'][valid_mask]).astype(np.int32)
        gap = np.maximum(totals - opt_depths, 0)

        # 1. Efficiency Gap PMF (%)
        c1 = np.bincount(gap)
        c1_pct = c1/len(gap)*100
        ax1.plot(np.arange(len(c1)), c1_pct, '-o', label=label, color=color, markersize=4, linewidth=2)
        for moves, (count, pct) in enumerate(zip(c1, c1_pct)): 
            gap_rows.append([label, moves, count, pct])

        # 2. Total Move Distribution PMF (%)
        c2 = np.bincount(totals)
        c2_pct = c2/len(totals)*100
        ax2.plot(np.arange(len(c2)), c2_pct, '-o', label=label, color=color, markersize=4, linewidth=2)
        for moves, (count, pct) in enumerate(zip(c2, c2_pct)): 
            total_rows.append([label, moves, count, pct])

        # 3. Cumulative Distribution (%)
        ax3.step(np.sort(totals), np.linspace(0, 100, len(totals)), where='post', label=label, color=color, linewidth=2)
        
        c2_cum_count = np.cumsum(c2)
        c2_cdf = c2_cum_count / len(totals) * 100
        for moves, (cum_count, pct) in enumerate(zip(c2_cum_count, c2_cdf)): 
            cdf_rows.append([label, moves, cum_count, pct])

    if optimal_dist is not None:
        opt = optimal_dist[optimal_dist >= 0].astype(np.int32)
        c_opt = np.bincount(opt)
        c_opt_pct = c_opt/len(opt)*100
        
        ax2.plot(np.arange(len(c_opt)), c_opt_pct, marker='o', linestyle='--', linewidth=2.0, 
                 markersize=7, color='black', label="Optimal", zorder=20, 
                 markerfacecolor='white', markeredgewidth=1.5)
        for moves, (count, pct) in enumerate(zip(c_opt, c_opt_pct)): 
            total_rows.append(["Optimal", moves, count, pct])
        
        ax3.step(np.sort(opt), np.linspace(0, 100, len(opt)), where='post', 
                 color='black', linestyle='--', alpha=0.6, label="Optimal CDF", zorder=10)
                 
        c_opt_cum_count = np.cumsum(c_opt)
        c_opt_cdf = c_opt_cum_count / len(opt) * 100
        for moves, (cum_count, pct) in enumerate(zip(c_opt_cum_count, c_opt_cdf)): 
            cdf_rows.append(["Optimal", moves, cum_count, pct])

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

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_matrix_efficiency_gap.csv", ["Method", "Moves_Above_Optimal", "Count", "Percentage"], gap_rows)
        viz.save_computed_csv("computed_matrix_total_moves.csv", ["Method", "Total_Moves", "Count", "Percentage"], total_rows)
        viz.save_computed_csv("computed_matrix_cumulative.csv", ["Method", "Total_Moves", "Cumulative_Count", "Cumulative_Percentage"], cdf_rows)

def plot_efficiency_gap_faceted(viz, data_dict):
    """Small-multiple histograms showing move waste per method."""
    n = len(data_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    stats_rows = []

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]
        valid_mask = data['depth'] >= 0
        totals = viz.get_total_moves(data)
        gap = np.maximum(totals - data['depth'][valid_mask], 0).astype(np.int32)
        
        color = viz.method_colors.get(label, '#777777')
        weights = np.ones_like(gap) / len(gap) * 100
        ax.hist(gap, bins=np.arange(gap.max()+2)-0.5, color=color, rwidth=0.8, edgecolor='black', weights=weights, alpha=0.8)
        
        mean_gap = np.mean(gap)
        stats_rows.append([label, mean_gap, int(np.median(gap)), np.max(gap), len(gap)])

        ax.axvline(mean_gap, color='red', linestyle='--', label=f'Mean: {mean_gap:.2f}')
        ax.set_title(f"{label}\nEfficiency Gap", fontweight='bold')
        ax.set_xlabel("Extra Moves")
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    axes[0].set_ylabel("% of States")
    for k in range(n, len(axes)): axes[k].axis('off')
    viz.save_plot(fig, "efficiency_gap_faceted.png")

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_gap_stats.csv", ["Method", "Mean_Gap", "Median_Gap", "Max_Gap", "N_States"], stats_rows)

def plot_per_method_totals(viz, data_dict):
    """Faceted histograms of total moves with stats box."""
    n = len(data_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    stats_rows = []

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]
        totals = viz.get_total_moves(data).astype(np.int32)
        color = viz.method_colors.get(label, '#777777')
        
        weights = np.ones_like(totals) / len(totals) * 100
        ax.hist(totals, bins=np.arange(totals.max()+2)-0.5, color=color, rwidth=0.8, edgecolor='black', weights=weights)
        
        mean_val = np.mean(totals)
        median_val = int(np.median(totals))
        stats_rows.append([label, mean_val, median_val, np.max(totals), len(totals)])

        stats_text = f"Avg: {mean_val:.2f}\nMed: {median_val}\nN: {len(totals):,}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_title(label)
        ax.set_xlabel("Total Moves")
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    axes[0].set_ylabel("% of States")
    for k in range(n, len(axes)): axes[k].axis('off')
    viz.save_plot(fig, "method_totals_faceted.png")

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_totals_stats.csv", ["Method", "Mean_Moves", "Median_Moves", "Max_Moves", "N_States"], stats_rows)

def plot_auf_grouped_bars(viz, data_dict):
    """AUF Distribution bars."""
    fig, ax = plt.subplots(figsize=(14, 7))
    auf_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'] 
    labels = list(data_dict.keys())
    x = np.arange(len(labels))
    width = 0.18

    auf_rows = []

    for i, label in enumerate(labels):
        data = data_dict[label]
        valid = data['depth'] >= 0
        total_auf = (data['pre_auf'][valid] + data['mid_auf'][valid] + data['post_auf'][valid]).astype(np.int32)
        counts = np.bincount(np.clip(total_auf, 0, 3), minlength=4)
        pcts = counts / len(total_auf) * 100
        
        row_data = [label]
        for val in range(4):
            if counts[val] > 0:
                pos = i + (val-1.5)*width
                ax.bar(pos, pcts[val], width, color=auf_colors[val], edgecolor='black', alpha=0.85)
                ax.text(pos, pcts[val], f"{int(counts[val]):,}\n({pcts[val]:.1f}%)", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            row_data.extend([int(counts[val]), pcts[val]])

        mean_val = np.mean(total_auf)
        row_data.append(mean_val)
        auf_rows.append(row_data)

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

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_auf_distribution.csv", 
            ["Method", "AUF_0_Count", "AUF_0_Pct", "AUF_1_Count", "AUF_1_Pct", "AUF_2_Count", "AUF_2_Pct", "AUF_3_Count", "AUF_3_Pct", "Mean_Total_AUF"], 
            auf_rows)

def plot_first_vs_remainder_lines(viz, data_dict, optimal_dist=None):
    """Compares setup steps vs algorithmic remainder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    split_rows = []

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
        c1_pct = c1/len(first_vals)*100
        ax1.plot(np.arange(len(c1)), c1_pct, '-o', label=f"{label} ({first_col.upper()})", color=color, markersize=4)
        for moves, (count, pct) in enumerate(zip(c1, c1_pct)): 
            split_rows.append([label, "First_Step", moves, count, pct])
        
        # Plot Remainder
        c2 = np.bincount(remainder)
        c2_pct = c2/len(remainder)*100
        ax2.plot(np.arange(len(c2)), c2_pct, '--o', label=label, color=color, markerfacecolor='white', markersize=4)
        for moves, (count, pct) in enumerate(zip(c2, c2_pct)): 
            split_rows.append([label, "Remainder", moves, count, pct])

    ax1.set_title("First Step Distribution"); ax2.set_title("Remainder Distribution")
    for ax in [ax1, ax2]: 
        ax.set_ylabel("% of States")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='upper right')
    viz.save_plot(fig, "first_vs_remainder.png")

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_first_vs_remainder.csv", ["Method", "Segment", "Moves", "Count", "Percentage"], split_rows)

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

    step_rows = []

    for i, step in enumerate(all_steps):
        ax = axes[i]
        for label, data in data_dict.items():
            if step in data.dtype.names:
                vals = data[step][data[step] >= 0].astype(np.int32)
                if len(vals) == 0: continue
                
                counts = np.bincount(vals)
                pcts = counts/len(vals)*100
                color = viz.method_colors.get(label, '#777777')
                
                ax.plot(np.arange(len(counts)), pcts, '-o', label=label, color=color, markersize=3, alpha=0.8)
                
                for moves, (count, pct) in enumerate(zip(counts, pcts)):
                    step_rows.append([label, step, moves, count, pct])
        
        ax.set_title(f"Step: {step.upper()}", fontweight='bold')
        ax.set_ylabel("% of States")
        ax.grid(True, alpha=0.2, linestyle='--')
        if i == 0: ax.legend(fontsize=8, loc='upper right')

    for k in range(len(all_steps), len(axes)): axes[k].axis('off')
    plt.tight_layout()
    viz.save_plot(fig, "per_step_distribution.png")

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_per_step_pmf.csv", ["Method", "Step_Name", "Moves", "Count", "Percentage"], step_rows)
    
def plot_random_efficiency_gap(viz, data_dict):
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_trials = len(list(data_dict.values())[0]) if data_dict else 0
    n_str = f"{num_trials // 1000}k" if num_trials >= 1000 else str(num_trials)
        
    ax.set_title(f"Performance vs Optimal ({n_str} Random WCA Scrambles)", fontsize=14, fontweight='bold')
    
    max_gap = 0
    export_rows = []
    
    for label, data in data_dict.items():
        color = viz.method_colors.get(label, '#777777')
        
        valid_mask = data['depth'] >= 0
        if np.sum(valid_mask) == 0: continue
        
        totals = viz.get_total_moves(data)
        opt_depths = np.asarray(data['depth'][valid_mask]).astype(np.int32)
        
        gap = np.maximum(totals - opt_depths, 0)
        if len(gap) == 0: continue
        
        max_gap = max(max_gap, np.max(gap))
        
        counts = np.bincount(gap)
        percentages = (counts / len(gap)) * 100
        
        for moves_over, (count, pct) in enumerate(zip(counts, percentages)):
            export_rows.append([label, moves_over, count, pct])
        
        ax.plot(np.arange(len(counts)), percentages, '-o', label=label, color=color, markersize=5, linewidth=2)
        
    ax.set_xlabel("Moves Over Optimal (Efficiency Gap)", fontsize=12)
    ax.set_ylabel("% of Solves", fontsize=12)
    ax.set_xlim(0, max(5, max_gap + 1))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    viz.save_plot(fig, "random_performance_vs_optimal.png")
    
    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv(
            "computed_random_efficiency_gap.csv",
            ["Method", "Moves_Over_Optimal", "Count", "Percentage_of_Solves"],
            export_rows
        )

def plot_random_comparison_summary(viz, data_dict):
    """
    Creates a 3-panel summary plot (Violin, CDF, Boxplot) of random trial data,
    mimicking the original random solver's output format, and prints stats to console.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not data_dict:
        return

    method_labels = []
    extra_moves_data = []
    total_moves_data = []
    optimal_depths_data = []
    first_step_data = []
    colors_list = []

    # Determine number of trials from the first array for the title
    first_key = list(data_dict.keys())[0]
    num_trials = len(data_dict[first_key])
    n_str = f"{num_trials // 1000}k" if num_trials >= 1000 else str(num_trials)

    for label, data in data_dict.items():
        valid_mask = data['depth'] >= 0
        if np.sum(valid_mask) == 0:
            continue

        method_labels.append(label)
        colors_list.append(viz.method_colors.get(label, '#777777'))

        total_moves = viz.get_total_moves(data)
        opt_depths = np.asarray(data['depth'][valid_mask]).astype(np.int32)
        extra_moves = np.maximum(total_moves - opt_depths, 0)

        total_moves_data.append(total_moves)
        optimal_depths_data.append(opt_depths)
        extra_moves_data.append(extra_moves)

        step_cols = [n for n in data.dtype.names if n not in ['depth', 'pre_auf', 'mid_auf', 'post_auf']]
        if step_cols:
            first_step_data.append(data[step_cols[0]][valid_mask])
        else:
            first_step_data.append(np.zeros(len(total_moves)))

    if not method_labels:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Random Trials Summary N={n_str}\nDistribution of Extra Moves Above Optimal (Violin)', fontsize=14, y=0.99)

    # 1. Violin plot of extra moves
    ax1 = axes[0]
    parts = ax1.violinplot(extra_moves_data, positions=range(len(method_labels)),
                           widths=0.7, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
    
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('#555555')
        vp.set_linewidth(1)

    ax1.set_xticks(range(len(method_labels)))
    ax1.set_xticklabels(method_labels)
    ax1.set_ylabel('Extra Moves')
    ax1.grid(axis='y', alpha=0.3)

    # 2. CDF plot
    ax2 = axes[1]
    ax2.set_title('CDF of Extra Moves (percent)')

    stats_rows = []
    cdf_rows = []
    violin_rows = []
    boxplot_rows = []

    for i, (label, extra_moves) in enumerate(zip(method_labels, extra_moves_data)):
        # Generate visual points for CDF
        sorted_moves = np.sort(extra_moves)
        plot_cum_counts = np.arange(1, len(sorted_moves) + 1)
        plot_cdf = plot_cum_counts / len(sorted_moves) * 100
        
        ax2.plot(sorted_moves, plot_cdf, label=label, color=colors_list[i], linewidth=2)

        # Export 1: CDF Data
        binned_extra = np.bincount(extra_moves)
        cum_counts = np.cumsum(binned_extra)
        cum_pcts = cum_counts / len(extra_moves) * 100
        
        for moves, (count, pct) in enumerate(zip(cum_counts, cum_pcts)):
            cdf_rows.append([label, moves, count, pct])

        # Export 2: Violin Plot Data (Extra Moves Frequency)
        extra_pcts = binned_extra / len(extra_moves) * 100
        for moves, (count, pct) in enumerate(zip(binned_extra, extra_pcts)):
            violin_rows.append([label, moves, count, pct])

        # Export 3: Box Plot Data (Total Moves Frequency)
        total = total_moves_data[i]
        binned_total = np.bincount(total)
        total_pcts = binned_total / len(total) * 100
        for moves, (count, pct) in enumerate(zip(binned_total, total_pcts)):
            boxplot_rows.append([label, moves, count, pct])

        # Export 4: Summary Statistics
        optimal = optimal_depths_data[i]
        first_step = first_step_data[i]

        stats_rows.append([
            label, 
            np.mean(extra_moves), np.median(extra_moves), np.std(extra_moves),
            np.mean(total), np.median(total), np.std(total),
            np.mean(first_step), np.median(first_step),
            np.mean(optimal), np.median(optimal),
            len(extra_moves)
        ])

    ax2.set_xlabel('Extra Moves')
    ax2.set_ylabel('% Cumulative')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    # 3. Box plot of total moves
    ax3 = axes[2]
    ax3.set_title('Total Moves Distribution per Method')

    bp = ax3.boxplot(total_moves_data, positions=range(len(method_labels)), patch_artist=True,
                     showmeans=True, meanprops=dict(marker='^', markerfacecolor='black',
                                                    markeredgecolor='black', markersize=6))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_alpha(0.6)

    ax3.set_xticks(range(len(method_labels)))
    ax3.set_xticklabels(method_labels)
    ax3.set_ylabel('Total Moves')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    viz.save_plot(fig, "random_comparison_summary.png")

    if hasattr(viz, 'save_computed_csv'):
        viz.save_computed_csv("computed_random_summary_stats.csv", 
            ["Method", "Extra_Mean", "Extra_Median", "Extra_Std", "Total_Mean", "Total_Median", "Total_Std", "FirstStep_Mean", "FirstStep_Median", "Optimal_Mean", "Optimal_Median", "N_States"], 
            stats_rows)
        viz.save_computed_csv("computed_random_summary_cdf.csv", 
            ["Method", "Extra_Moves", "Cumulative_Count", "Cumulative_Percentage"], 
            cdf_rows)
            
        viz.save_computed_csv("computed_random_summary_extra_moves_pmf.csv", 
            ["Method", "Extra_Moves", "Count", "Percentage"], 
            violin_rows)
        viz.save_computed_csv("computed_random_summary_total_moves_pmf.csv", 
            ["Method", "Total_Moves", "Count", "Percentage"], 
            boxplot_rows)

    # Print summary statistics to the terminal
    print("\n" + "="*60)
    print(f"RANDOM TRIALS SUMMARY STATISTICS (N={num_trials})")
    print("="*60)
    for i, label in enumerate(method_labels):
        extra = extra_moves_data[i]
        total = total_moves_data[i]
        optimal = optimal_depths_data[i]
        first_step = first_step_data[i]

        print(f"\n{label.upper()}:")
        print(f"  Extra moves:   mean={np.mean(extra):.2f}, median={np.median(extra):.1f}, std={np.std(extra):.2f}")
        print(f"  Total moves:   mean={np.mean(total):.2f}, median={np.median(total):.1f}")
        print(f"  First step:    mean={np.mean(first_step):.2f}, median={np.median(first_step):.1f}")
        print(f"  Optimal depth: mean={np.mean(optimal):.2f}, median={np.median(optimal):.1f}")
    print("-" * 60 + "\n")