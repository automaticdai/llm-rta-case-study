"""Hand-crafted unit tests for base-case RTA."""

import unittest
from rta.models import Task, TaskSet
from rta.analysis import compute_response_time, is_schedulable, analyze_taskset


class TestTask(unittest.TestCase):
    """Test Task model validation and properties."""
    
    def test_valid_task(self):
        """Test creating a valid task."""
        task = Task(C=2.0, T=10.0, D=10.0, name="τ1")
        self.assertEqual(task.C, 2.0)
        self.assertEqual(task.T, 10.0)
        self.assertEqual(task.D, 10.0)
        self.assertEqual(task.utilization, 0.2)
    
    def test_default_deadline(self):
        """Test that deadline defaults to period."""
        task = Task(C=2.0, T=10.0)
        self.assertEqual(task.D, 10.0)
    
    def test_invalid_negative_c(self):
        """Test that negative C raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=-1.0, T=10.0)
    
    def test_invalid_zero_c(self):
        """Test that zero C raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=0.0, T=10.0)
    
    def test_invalid_negative_t(self):
        """Test that negative T raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=1.0, T=-10.0)
    
    def test_invalid_zero_t(self):
        """Test that zero T raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=1.0, T=0.0)
    
    def test_invalid_c_exceeds_t(self):
        """Test that C > T raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=11.0, T=10.0)
    
    def test_invalid_c_exceeds_d(self):
        """Test that C > D raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=8.0, T=10.0, D=5.0)
    
    def test_invalid_d_exceeds_t(self):
        """Test that D > T raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=5.0, T=10.0, D=15.0)
    
    def test_invalid_negative_d(self):
        """Test that negative D raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=1.0, T=10.0, D=-5.0)
    
    def test_invalid_zero_d(self):
        """Test that zero D raises ValueError."""
        with self.assertRaises(ValueError):
            Task(C=1.0, T=10.0, D=0.0)
    
    def test_c_equals_t(self):
        """Test edge case where C = T (utilization = 1.0)."""
        task = Task(C=10.0, T=10.0)
        self.assertEqual(task.C, 10.0)
        self.assertEqual(task.T, 10.0)
        self.assertEqual(task.D, 10.0)
        self.assertEqual(task.utilization, 1.0)
    
    def test_c_equals_d_less_than_t(self):
        """Test edge case where C = D < T."""
        task = Task(C=5.0, T=10.0, D=5.0)
        self.assertEqual(task.C, 5.0)
        self.assertEqual(task.D, 5.0)


class TestTaskSet(unittest.TestCase):
    """Test TaskSet model and priority assignment."""
    
    def test_rate_monotonic_assignment(self):
        """Test that Rate Monotonic priorities are assigned correctly."""
        tasks = [
            Task(C=1.0, T=10.0, name="τ1"),
            Task(C=2.0, T=20.0, name="τ2"),
            Task(C=3.0, T=5.0, name="τ3"),
        ]
        taskset = TaskSet(tasks=tasks)
        sorted_tasks = taskset.get_sorted_tasks()
        
        # τ3 has shortest period, should have priority 0
        self.assertEqual(sorted_tasks[0].name, "τ3")
        self.assertEqual(sorted_tasks[0].priority, 0)
        
        # τ1 has middle period, should have priority 1
        self.assertEqual(sorted_tasks[1].name, "τ1")
        self.assertEqual(sorted_tasks[1].priority, 1)
        
        # τ2 has longest period, should have priority 2
        self.assertEqual(sorted_tasks[2].name, "τ2")
        self.assertEqual(sorted_tasks[2].priority, 2)
    
    def test_identical_periods(self):
        """Test tasks with identical periods (tie-breaking by original order)."""
        tasks = [
            Task(C=1.0, T=10.0, name="τ1"),
            Task(C=2.0, T=10.0, name="τ2"),
            Task(C=3.0, T=10.0, name="τ3"),
        ]
        taskset = TaskSet(tasks=tasks)
        sorted_tasks = taskset.get_sorted_tasks()
        
        # All have same period, so priorities should be 0, 1, 2 in original order
        self.assertEqual(sorted_tasks[0].priority, 0)
        self.assertEqual(sorted_tasks[1].priority, 1)
        self.assertEqual(sorted_tasks[2].priority, 2)
    
    def test_total_utilization(self):
        """Test total utilization calculation."""
        tasks = [
            Task(C=2.0, T=10.0),  # U = 0.2
            Task(C=3.0, T=15.0),  # U = 0.2
            Task(C=4.0, T=20.0),  # U = 0.2
        ]
        taskset = TaskSet(tasks=tasks)
        self.assertAlmostEqual(taskset.total_utilization, 0.6, places=6)
    
    def test_get_higher_priority_tasks(self):
        """Test getting higher priority tasks."""
        tasks = [
            Task(C=1.0, T=5.0, name="τ1"),
            Task(C=2.0, T=10.0, name="τ2"),
            Task(C=3.0, T=20.0, name="τ3"),
        ]
        taskset = TaskSet(tasks=tasks)
        sorted_tasks = taskset.get_sorted_tasks()
        
        # For τ3 (priority 2), should get τ1 and τ2
        hp_tasks = taskset.get_higher_priority_tasks(sorted_tasks[2])
        self.assertEqual(len(hp_tasks), 2)
        self.assertEqual(hp_tasks[0].name, "τ1")
        self.assertEqual(hp_tasks[1].name, "τ2")
        
        # For τ1 (priority 0), should get no tasks
        hp_tasks = taskset.get_higher_priority_tasks(sorted_tasks[0])
        self.assertEqual(len(hp_tasks), 0)
    
    def test_empty_taskset(self):
        """Test creating an empty task set."""
        taskset = TaskSet()
        self.assertEqual(len(taskset), 0)
        self.assertEqual(taskset.total_utilization, 0.0)


class TestRTA(unittest.TestCase):
    """Test response-time analysis algorithms."""
    
    def test_single_task_schedulable(self):
        """Test RTA for a single task (always schedulable if C <= D)."""
        task = Task(C=2.0, T=10.0, D=10.0)
        rt = compute_response_time(task, [])
        self.assertIsNotNone(rt)
        self.assertAlmostEqual(rt, 2.0, places=6)
    
    def test_single_task_c_equals_d(self):
        """Test single task where C = D (exactly schedulable)."""
        task = Task(C=10.0, T=10.0, D=10.0)
        rt = compute_response_time(task, [])
        self.assertIsNotNone(rt)
        self.assertAlmostEqual(rt, 10.0, places=6)
        self.assertTrue(is_schedulable(task, []))
    
    def test_single_task_unschedulable(self):
        """Test single task where C > D (should be caught by Task validation)."""
        # This should raise ValueError during Task construction
        with self.assertRaises(ValueError):
            Task(C=11.0, T=10.0, D=10.0)
    
    def test_two_tasks_schedulable(self):
        """Test RTA for two schedulable tasks."""
        # τ1: C=1, T=4 (higher priority)
        # τ2: C=2, T=6 (lower priority)
        task1 = Task(C=1.0, T=4.0, priority=0)
        task2 = Task(C=2.0, T=6.0, priority=1)
        
        # Task 1 has no higher priority tasks
        rt1 = compute_response_time(task1, [])
        self.assertAlmostEqual(rt1, 1.0, places=6)
        
        # Task 2 has task1 as higher priority
        # R2 = 2 + ceil(R2/4)*1
        # Iteration 0: R = 2
        # Iteration 1: R = 2 + ceil(2/4)*1 = 2 + 1 = 3
        # Iteration 2: R = 2 + ceil(3/4)*1 = 2 + 1 = 3 (converged)
        rt2 = compute_response_time(task2, [task1])
        self.assertAlmostEqual(rt2, 3.0, places=6)
    
    def test_task_unschedulable(self):
        """Test RTA for an unschedulable task."""
        # τ1: C=3, T=5 (higher priority, U=0.6)
        # τ2: C=3, T=5, D=5 (lower priority, U=0.6)
        # Total U = 1.2 > 1, so τ2 should be unschedulable
        task1 = Task(C=3.0, T=5.0, priority=0)
        task2 = Task(C=3.0, T=5.0, D=5.0, priority=1)
        
        rt2 = compute_response_time(task2, [task1])
        self.assertIsNone(rt2)
    
    def test_exact_deadline_met(self):
        """Test task that exactly meets its deadline (R_i = D_i)."""
        # τ1: C=2, T=5 (U=0.4)
        # τ2: C=3, T=10, D=5 (U=0.3)
        # R2 = 3 + ceil(R2/5)*2
        # Iteration 0: R = 3
        # Iteration 1: R = 3 + ceil(3/5)*2 = 3 + 2 = 5
        # Iteration 2: R = 3 + ceil(5/5)*2 = 3 + 2 = 5 (converged)
        # R2 = 5 = D2, so schedulable
        task1 = Task(C=2.0, T=5.0, priority=0)
        task2 = Task(C=3.0, T=10.0, D=5.0, priority=1)
        
        rt2 = compute_response_time(task2, [task1])
        self.assertIsNotNone(rt2)
        self.assertAlmostEqual(rt2, 5.0, places=6)
        self.assertTrue(is_schedulable(task2, [task1]))
    
    def test_just_unschedulable(self):
        """Test task that just misses its deadline (R_i = D_i + epsilon)."""
        # Design a case where response time slightly exceeds deadline
        # τ1: C=2, T=5
        # τ2: C=3, T=10, D=4.9
        # R2 = 3 + ceil(R2/5)*2 = 5 > 4.9
        task1 = Task(C=2.0, T=5.0, priority=0)
        task2 = Task(C=3.0, T=10.0, D=4.9, priority=1)
        
        rt2 = compute_response_time(task2, [task1])
        self.assertIsNone(rt2)  # Should be unschedulable
    
    def test_taskset_all_schedulable(self):
        """Test analyzing a fully schedulable task set."""
        tasks = [
            Task(C=1.0, T=4.0, name="τ1"),
            Task(C=2.0, T=6.0, name="τ2"),
            Task(C=1.0, T=12.0, name="τ3"),
        ]
        taskset = TaskSet(tasks=tasks)
        schedulable, response_times = analyze_taskset(taskset)
        
        self.assertTrue(schedulable)
        self.assertIsNotNone(response_times["τ1"])
        self.assertIsNotNone(response_times["τ2"])
        self.assertIsNotNone(response_times["τ3"])
    
    def test_taskset_one_unschedulable(self):
        """Test analyzing a task set with one unschedulable task."""
        tasks = [
            Task(C=3.0, T=5.0, name="τ1"),
            Task(C=3.0, T=5.0, name="τ2"),
        ]
        taskset = TaskSet(tasks=tasks)
        schedulable, response_times = analyze_taskset(taskset)
        
        self.assertFalse(schedulable)
        self.assertIsNotNone(response_times["τ1"])  # First task should be schedulable
        self.assertIsNone(response_times["τ2"])  # Second task should fail
    
    def test_taskset_single_task(self):
        """Test analyzing a task set with a single task."""
        tasks = [Task(C=5.0, T=10.0, name="τ1")]
        taskset = TaskSet(tasks=tasks)
        schedulable, response_times = analyze_taskset(taskset)
        
        self.assertTrue(schedulable)
        self.assertAlmostEqual(response_times["τ1"], 5.0, places=6)
    
    def test_response_time_at_least_wcet(self):
        """Test that response time is always at least WCET."""
        tasks = [
            Task(C=1.0, T=5.0, name="τ1"),
            Task(C=2.0, T=10.0, name="τ2"),
        ]
        taskset = TaskSet(tasks=tasks)
        schedulable, response_times = analyze_taskset(taskset)
        
        for task in taskset:
            rt = response_times[task.name]
            if rt is not None:
                self.assertGreaterEqual(rt, task.C)
    
    def test_response_time_bounded_by_deadline(self):
        """Test that response time is bounded by deadline for schedulable tasks."""
        tasks = [
            Task(C=1.0, T=5.0, name="τ1"),
            Task(C=2.0, T=10.0, name="τ2"),
            Task(C=1.5, T=15.0, name="τ3"),
        ]
        taskset = TaskSet(tasks=tasks)
        schedulable, response_times = analyze_taskset(taskset)
        
        if schedulable:
            for task in taskset:
                rt = response_times[task.name]
                self.assertIsNotNone(rt)
                self.assertLessEqual(rt, task.D)


if __name__ == "__main__":
    unittest.main()
