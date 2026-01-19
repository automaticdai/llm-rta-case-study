"""Task set generators for testing and experiments."""

import random
from typing import List, Optional
import math

from rta.models import Task, TaskSet


def uunifast(n: int, u_total: float, seed: Optional[int] = None) -> List[float]:
    """Generate task utilizations using the UUniFast algorithm.
    
    UUniFast generates uniformly distributed task utilizations that sum to
    the target total utilization.
    
    Reference:
    Bini, E., & Buttazzo, G. C. (2005). Measuring the performance of schedulability tests.
    Real-Time Systems, 30(1-2), 129-154.
    
    Args:
        n: Number of tasks.
        u_total: Target total utilization (should be <= n for feasibility).
        seed: Optional random seed for reproducibility.
    
    Returns:
        List of n utilization values that sum to approximately u_total.
    
    Raises:
        ValueError: If n <= 0 or u_total < 0.
    """
    if n <= 0:
        raise ValueError("Number of tasks must be positive")
    if u_total < 0:
        raise ValueError("Target utilization must be non-negative")
    
    # Create Random instance from seed
    rng = random.Random(seed) if seed is not None else random.Random()
    
    utilizations = []
    sum_u = u_total
    
    for i in range(1, n):
        # Generate next utilization
        next_sum_u = sum_u * (rng.random() ** (1.0 / (n - i)))
        utilizations.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    
    # Last utilization is whatever remains
    utilizations.append(sum_u)
    
    return utilizations


def generate_random_task_set_uunifast(
    n: int,
    u_total: float,
    min_period: int = 10,
    max_period: int = 1000,
    seed: Optional[int] = None
) -> List[Task]:
    """Generate a random task set using UUniFast for utilization distribution.
    
    This function generates n tasks with periods uniformly distributed (log-scale)
    between min_period and max_period, and execution times determined by the
    UUniFast algorithm to achieve the target total utilization.
    
    Args:
        n: Number of tasks to generate.
        u_total: Target total utilization.
        min_period: Minimum task period.
        max_period: Maximum task period.
        seed: Optional random seed for reproducibility.
    
    Returns:
        List of Task objects with Rate Monotonic priorities assigned.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if n <= 0:
        raise ValueError("Number of tasks must be positive")
    if min_period <= 0 or max_period <= 0 or min_period > max_period:
        raise ValueError("Invalid period range")
    if u_total < 0:
        raise ValueError("Target utilization must be non-negative")
    
    # Create Random instance from seed
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Generate utilizations using UUniFast with same seed
    utilizations = uunifast(n, u_total, seed=seed)
    
    tasks = []
    for i, u in enumerate(utilizations):
        # Generate random period (log-uniform distribution)
        log_min = math.log(min_period)
        log_max = math.log(max_period)
        T = math.exp(rng.uniform(log_min, log_max))
        
        # Compute WCET from utilization
        C = u * T
        
        # Ensure C is at least a small positive value
        if C < 0.001:
            C = 0.001
        
        # Deadline equals period (implicit deadline)
        D = T
        
        # Ensure C <= D (should hold by construction, but check for rounding)
        if C > D:
            C = D
        
        task = Task(
            C=C,
            T=T,
            D=D,
            name=f"τ{i+1}"
        )
        tasks.append(task)
    
    return tasks


def generate_taskset(
    n: int,
    target_utilization: float,
    period_min: float = 10.0,
    period_max: float = 1000.0,
    deadline_factor_min: float = 1.0,
    deadline_factor_max: float = 1.0,
    seed: Optional[int] = None
) -> TaskSet:
    """Generate a random task set using UUniFast for utilization distribution.
    
    This is a convenience wrapper that returns a TaskSet object instead of a list.
    
    Args:
        n: Number of tasks to generate.
        target_utilization: Target total utilization.
        period_min: Minimum task period.
        period_max: Maximum task period.
        deadline_factor_min: Minimum ratio D/T (1.0 means D=T).
        deadline_factor_max: Maximum ratio D/T (1.0 means D=T).
        seed: Random seed for reproducibility.
    
    Returns:
        A TaskSet with n tasks.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if period_min <= 0 or period_max <= 0 or period_min > period_max:
        raise ValueError("Invalid period range")
    if deadline_factor_min <= 0 or deadline_factor_max < deadline_factor_min:
        raise ValueError("Invalid deadline factor range")
    if deadline_factor_max > 1.0:
        raise ValueError("Deadline factor cannot exceed 1.0 (D must be <= T)")
    
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Generate utilizations using UUniFast with same seed
    utilizations = uunifast(n, target_utilization, seed=seed)
    
    tasks = []
    for i, u in enumerate(utilizations):
        # Generate random period (log-uniform distribution)
        log_min = math.log(period_min)
        log_max = math.log(period_max)
        T = math.exp(rng.uniform(log_min, log_max))
        
        # Compute WCET from utilization
        C = u * T
        
        # Generate deadline
        if deadline_factor_min == deadline_factor_max:
            deadline_factor = deadline_factor_min
        else:
            deadline_factor = rng.uniform(deadline_factor_min, deadline_factor_max)
        D = T * deadline_factor
        
        # Ensure C <= D (might not hold due to rounding)
        if C > D:
            D = C
        
        task = Task(
            C=C,
            T=T,
            D=D,
            name=f"τ{i+1}"
        )
        tasks.append(task)
    
    return TaskSet(tasks=tasks)
