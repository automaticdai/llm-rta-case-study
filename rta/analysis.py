"""Response-time analysis algorithms for fixed-priority scheduling.

This module implements the standard iterative response-time analysis (RTA)
for fixed-priority preemptive scheduling on a single processor.

RTA Formula (base case, no jitter, no blocking):
    R_i^(k+1) = C_i + sum_{j in hp(i)} ceil(R_i^(k) / T_j) * C_j

where:
    - R_i^(k) is the response time estimate at iteration k
    - C_i is the worst-case execution time of task i
    - hp(i) is the set of tasks with higher priority than task i
    - T_j is the period of task j
    - C_j is the worst-case execution time of task j

The iteration starts with R_i^(0) = C_i and continues until either:
    1. Convergence: R_i^(k+1) = R_i^(k)
    2. Deadline miss: R_i^(k+1) > D_i

Assumptions:
    - Single processor
    - Fixed-priority preemptive scheduling
    - Periodic or sporadic tasks
    - No release jitter
    - No blocking (no shared resources)
    - Priority assignment: Rate Monotonic (shorter period = higher priority)
"""

from typing import Optional, Dict, Tuple
import math

from rta.models import Task, TaskSet


def compute_response_time(
    task: Task,
    higher_priority_tasks: list[Task],
    max_iterations: int = 1000
) -> Optional[float]:
    """Compute the worst-case response time for a task using iterative RTA.
    
    Uses the standard fixed-point iteration:
        R_i^(k+1) = C_i + sum_{j in hp(i)} ceil(R_i^(k) / T_j) * C_j
    
    The iteration starts with R_i^(0) = C_i and continues until either:
        1. Convergence: |R_i^(k+1) - R_i^(k)| < epsilon
        2. Deadline miss: R_i^(k+1) > D_i
        3. Max iterations reached (returns None)
    
    Args:
        task: The task to analyze.
        higher_priority_tasks: List of tasks with higher priority than task.
        max_iterations: Maximum number of iterations before giving up.
    
    Returns:
        The worst-case response time if it converges and is <= D,
        None if the task is unschedulable (R > D or doesn't converge).
    """
    # Initial response time is just the task's own execution time
    R_prev = task.C
    
    for iteration in range(max_iterations):
        # Compute interference from higher-priority tasks
        interference = 0.0
        for hp_task in higher_priority_tasks:
            # Number of times hp_task can preempt during R_prev
            num_preemptions = math.ceil(R_prev / hp_task.T)
            interference += num_preemptions * hp_task.C
        
        # New response time estimate
        R_new = task.C + interference
        
        # Check if response time exceeds deadline
        if R_new > task.D:
            return None  # Unschedulable
        
        # Check for convergence
        if abs(R_new - R_prev) < 1e-9:  # Use small epsilon for floating-point comparison
            return R_new
        
        R_prev = R_new
    
    # Did not converge within max_iterations
    return None


def is_schedulable(task: Task, higher_priority_tasks: list[Task]) -> bool:
    """Check if a task is schedulable given higher-priority tasks.
    
    A task is schedulable if its worst-case response time is less than
    or equal to its deadline.
    
    Args:
        task: The task to check.
        higher_priority_tasks: List of tasks with higher priority.
    
    Returns:
        True if the task meets its deadline, False otherwise.
    """
    response_time = compute_response_time(task, higher_priority_tasks)
    return response_time is not None


def analyze_taskset(taskset: TaskSet) -> Tuple[bool, Dict[str, Optional[float]]]:
    """Analyze the schedulability of an entire task set.
    
    Performs response-time analysis on each task in priority order
    (highest priority first). A task set is schedulable if all tasks
    meet their deadlines.
    
    Args:
        taskset: The task set to analyze.
    
    Returns:
        A tuple of (schedulable, response_times) where:
        - schedulable: True if all tasks meet their deadlines.
        - response_times: Dict mapping task names to their response times
                         (None if unschedulable).
    """
    sorted_tasks = taskset.get_sorted_tasks()
    response_times: Dict[str, Optional[float]] = {}
    all_schedulable = True
    
    for task in sorted_tasks:
        hp_tasks = taskset.get_higher_priority_tasks(task)
        rt = compute_response_time(task, hp_tasks)
        
        task_name = task.name if task.name else f"Task_prio_{task.priority}"
        response_times[task_name] = rt
        
        if rt is None:
            all_schedulable = False
    
    return all_schedulable, response_times
