"""RTA: Response-Time Analysis library for fixed-priority scheduling.

This package provides tools for analyzing the schedulability of periodic/sporadic
task sets under fixed-priority preemptive scheduling on a single processor.
"""

from rta.models import Task, TaskSet
from rta.analysis import compute_response_time, is_schedulable, analyze_taskset

__version__ = "0.1.0"
__all__ = [
    "Task",
    "TaskSet",
    "compute_response_time",
    "is_schedulable",
    "analyze_taskset",
]
