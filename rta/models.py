"""Data models for tasks and task sets."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Task:
    """Represents a periodic or sporadic task.
    
    Attributes:
        C: Worst-case execution time (WCET).
        T: Period (or minimum inter-arrival time).
        D: Relative deadline (defaults to T if not specified).
        name: Optional task identifier.
        priority: Task priority (lower value = higher priority).
                  If not set, will be assigned by TaskSet based on Rate Monotonic.
    """
    C: float
    T: float
    D: Optional[float] = None
    name: str = ""
    priority: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate task parameters."""
        if self.C <= 0:
            raise ValueError(f"Task {self.name}: C must be positive, got {self.C}")
        if self.T <= 0:
            raise ValueError(f"Task {self.name}: T must be positive, got {self.T}")
        if self.C > self.T:
            raise ValueError(f"Task {self.name}: C ({self.C}) cannot exceed T ({self.T})")
        
        # Set default deadline to period if not specified
        if self.D is None:
            object.__setattr__(self, 'D', self.T)
        elif self.D <= 0:
            raise ValueError(f"Task {self.name}: D must be positive, got {self.D}")
        elif self.D > self.T:
            raise ValueError(f"Task {self.name}: D ({self.D}) cannot exceed T ({self.T})")
        
        if self.C > self.D:
            raise ValueError(f"Task {self.name}: C ({self.C}) cannot exceed D ({self.D})")
    
    @property
    def utilization(self) -> float:
        """Return the utilization of this task (C/T)."""
        return self.C / self.T
    
    def __str__(self) -> str:
        name_str = f"{self.name}: " if self.name else ""
        return f"Task({name_str}C={self.C}, T={self.T}, D={self.D}, prio={self.priority})"


@dataclass
class TaskSet:
    """Represents a set of tasks with priority assignment.
    
    Attributes:
        tasks: List of tasks in the set.
        _sorted: Whether tasks are sorted by priority.
    """
    tasks: List[Task] = field(default_factory=list)
    _sorted: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Assign priorities using Rate Monotonic if not already set."""
        if not self.tasks:
            return
        
        # Check if all tasks have priorities assigned
        if all(t.priority is not None for t in self.tasks):
            self._sorted = False
            return
        
        # Check if none have priorities (we'll assign them)
        if all(t.priority is None for t in self.tasks):
            self._assign_rate_monotonic_priorities()
            return
        
        # Mixed case: some have priorities, some don't
        raise ValueError("Either all tasks must have priorities set, or none should.")
    
    def _assign_rate_monotonic_priorities(self) -> None:
        """Assign priorities using Rate Monotonic (shorter period = higher priority)."""
        # Sort by period (ascending)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.T)
        
        # Assign priorities (0 = highest)
        new_tasks = []
        for i, task in enumerate(sorted_tasks):
            new_task = Task(
                C=task.C,
                T=task.T,
                D=task.D,
                name=task.name if task.name else f"τ{i+1}",
                priority=i
            )
            new_tasks.append(new_task)
        
        self.tasks = new_tasks
        self._sorted = True
    
    def get_sorted_tasks(self) -> List[Task]:
        """Return tasks sorted by priority (highest priority first)."""
        if not self._sorted:
            self.tasks.sort(key=lambda t: t.priority if t.priority is not None else float('inf'))
            self._sorted = True
        return self.tasks
    
    def get_higher_priority_tasks(self, task: Task) -> List[Task]:
        """Return all tasks with higher priority than the given task."""
        if task.priority is None:
            raise ValueError(f"Task {task.name} has no priority assigned")
        
        sorted_tasks = self.get_sorted_tasks()
        return [t for t in sorted_tasks if t.priority is not None and t.priority < task.priority]
    
    @property
    def total_utilization(self) -> float:
        """Return the total utilization of all tasks."""
        return sum(t.utilization for t in self.tasks)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)
    
    def __getitem__(self, index: int) -> Task:
        return self.tasks[index]
