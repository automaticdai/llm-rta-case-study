"""Schedulability vs Utilisation Experiment.

Generates random task sets at various utilisation levels using UUniFast,
runs RTA on each, and plots the schedulability ratio as a function of utilisation.
"""

import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from rta.generators import generate_taskset
from rta.analysis import analyze_taskset


def run_schedulability_experiment(
    utilisation_points: list,
    num_task_sets_per_point: int = 100,
    num_tasks: int = 5,
    min_period: float = 10.0,
    max_period: float = 1000.0,
    seed: int = 42,
) -> dict:
    """Run schedulability experiment across utilisation levels.
    
    Args:
        utilisation_points: List of utilisation values to test (e.g. [0.1, 0.2, ..., 0.9]).
        num_task_sets_per_point: Number of random task sets to generate per utilisation.
        num_tasks: Number of tasks per task set.
        min_period: Minimum task period.
        max_period: Maximum task period.
        seed: Base random seed (will be varied per task set).
    
    Returns:
        Dictionary mapping utilisation -> schedulability ratio.
    """
    results = {}
    
    for u_total in utilisation_points:
        schedulable_count = 0
        
        for i in range(num_task_sets_per_point):
            # Use different seed for each task set
            task_set_seed = seed + int(u_total * 1000) + i
            
            # Generate random task set
            taskset = generate_taskset(
                n=num_tasks,
                target_utilization=u_total,
                period_min=min_period,
                period_max=max_period,
                seed=task_set_seed,
            )
            
            # Run RTA
            schedulable, _ = analyze_taskset(taskset)
            
            # Check if schedulable
            if schedulable:
                schedulable_count += 1
        
        # Compute schedulability ratio
        schedulability_ratio = schedulable_count / num_task_sets_per_point
        results[u_total] = schedulability_ratio
    
    return results


def plot_schedulability_vs_utilisation(
    results: dict,
    output_path: str = "results/schedulability_vs_utilisation.png",
) -> None:
    """Plot schedulability ratio vs utilisation.
    
    Args:
        results: Dictionary mapping utilisation -> schedulability ratio.
        output_path: Path to save the plot.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by utilisation for plotting
    utilisations = sorted(results.keys())
    schedulability_ratios = [results[u] for u in utilisations]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(utilisations, schedulability_ratios, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Total Utilisation', fontsize=12)
    plt.ylabel('Schedulability Ratio', fontsize=12)
    plt.title('Schedulability vs Utilisation (RTA)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def main():
    """Run the full schedulability vs utilisation experiment."""
    print("Running schedulability vs utilisation experiment...")
    
    # Define utilisation points
    utilisation_points = [u / 10.0 for u in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    
    # Run experiment
    results = run_schedulability_experiment(
        utilisation_points=utilisation_points,
        num_task_sets_per_point=150,
        num_tasks=5,
        min_period=10.0,
        max_period=1000.0,
        seed=42,
    )
    
    # Print results
    print("\nResults:")
    for u, ratio in sorted(results.items()):
        print(f"  U = {u:.1f}: {ratio:.3f} schedulable")
    
    # Plot results
    plot_schedulability_vs_utilisation(results)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
