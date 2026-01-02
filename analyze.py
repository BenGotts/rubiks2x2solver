"""
Main analysis script for 2x2x2 Rubik's Cube solving methods.

Stores separate data files for each color in method subdirectories.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time

import pocket_cube
from solver import load_solver


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze 2x2x2 Rubik\'s Cube solving methods'
    )
    
    # Input/Output files
    parser.add_argument(
        '--dist-npy',
        required=True,
        help='NumPy file containing optimal distances for all states'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save analysis results'
    )
    
    # Method selection
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['ortega', 'cll', 'lbl', 'eg', 'all'],
        default=['ortega', 'cll', 'lbl'],
        help='Which methods to analyze (use "all" for all methods)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of states to analyze (default: all)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100000,
        help='Print progress every N states'
    )
    
    return parser.parse_args()


def calculate_statistics(data: np.ndarray, step_names: List[str], color: str) -> Dict:
    """Calculate statistics from analysis data for a specific color."""
    valid_mask = data[step_names[0]] >= 0
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        return {
            'error': 'No valid data',
            'processed_states': 0,
            'color': color
        }
    
    # Calculate total moves
    total_moves = sum(valid_data[step] for step in step_names)
    
    stats = {
        'color': color,
        'processed_states': int(len(valid_data)),
        'avg_total_moves': float(np.mean(total_moves)),
        'std_total_moves': float(np.std(total_moves)),
        'min_total_moves': int(np.min(total_moves)),
        'max_total_moves': int(np.max(total_moves)),
        'avg_optimal_moves': float(np.mean(valid_data['depth'])),
        'avg_efficiency_gap': float(np.mean(total_moves - valid_data['depth'])),
        'step_breakdown': {}
    }
    
    # Per-step statistics
    for step in step_names:
        stats['step_breakdown'][step] = {
            'avg': float(np.mean(valid_data[step])),
            'std': float(np.std(valid_data[step])),
            'min': int(np.min(valid_data[step])),
            'max': int(np.max(valid_data[step]))
        }
    
    return stats


def print_color_statistics(color: str, stats: Dict):
    """Print statistics for a specific color."""
    if 'error' in stats:
        print(f"    {color}: ⚠️  {stats['error']}")
        return
    
    print(f"    {color}: {stats['avg_total_moves']:.2f} avg moves "
          f"(gap: {stats['avg_efficiency_gap']:.2f}, states: {stats['processed_states']:,})")


def save_method_results(output_dir: Path, method_name: str, color_data: Dict[str, np.ndarray],
                        step_names: List[str], failed_states: List[Dict]):
    """Save analysis results organized by method."""
    # Create method subdirectory
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving results to {method_dir}/")
    
    # Save data file for each color
    for color, data in color_data.items():
        np.save(method_dir / f"{color}_data.npy", data)
        
        # Calculate and save statistics for this color
        stats = calculate_statistics(data, step_names, color)
        with open(method_dir / f"{color}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print_color_statistics(color, stats)
    
    # Save failures if any
    if failed_states:
        with open(method_dir / "failures.json", 'w') as f:
            json.dump(failed_states, f, indent=2)
        print(f"    Failures: {len(failed_states)} (saved to failures.json)")
    
    # Create summary with all colors
    summary = {
        'method': method_name,
        'colors': list(color_data.keys()),
        'failures': len(failed_states),
        'per_color_stats': {}
    }
    
    for color in sorted(color_data.keys()):
        stats = calculate_statistics(color_data[color], step_names, color)
        summary['per_color_stats'][color] = {
            'processed_states': stats.get('processed_states', 0),
            'avg_total_moves': stats.get('avg_total_moves', None),
            'avg_efficiency_gap': stats.get('avg_efficiency_gap', None)
        }
    
    with open(method_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_arguments()
    
    total_start_time = time.time()
    
    # Load distance data
    try:
        dist = np.load(args.dist_npy)
        print(f"✓ Loaded distance data: {args.dist_npy}")
        print(f"  States: {len(dist):,}")
    except FileNotFoundError:
        print(f"❌ Error: Distance file '{args.dist_npy}' not found")
        return
    
    # Determine methods to analyze
    if 'all' in args.methods:
        methods = ['ortega', 'cll', 'lbl', 'eg']
    else:
        methods = args.methods
    
    print(f"✓ Methods to analyze: {', '.join(methods)}")
    print(f"✓ Running analysis for ALL colors (W, Y, G, B, O, R)")
    
    # Determine max states
    max_states = args.limit if args.limit else pocket_cube.N_STATES
    max_states = min(max_states, len(dist))
    print(f"✓ Analyzing up to {max_states:,} states")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {method.upper()}")
        print(f"{'='*80}")
        
        method_start_time = time.time()
        
        try:
            solver = load_solver(method, list(pocket_cube.COLOR_NEUTRAL))
            color_data, failed_states = solver.run_analysis(
                dist,
                max_states=max_states,
                log_interval=args.log_interval
            )
            
            step_names = [step['name'] for step in solver.steps]
            save_method_results(output_dir, method, color_data, step_names, failed_states)
            
            method_elapsed = time.time() - method_start_time
            print(f"\n✓ Results saved to {output_dir}/{method}/")
            print(f"⏱️  Time elapsed: {method_elapsed:.1f} seconds ({method_elapsed/60:.1f} minutes)")
            
        except Exception as e:
            print(f"❌ Error analyzing {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! All results saved to '{output_dir}/'")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"\n💡 Next steps:")
    print(f"   1. Create comparison config: config.json")
    print(f"   2. Run: python visualize.py --config config.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()