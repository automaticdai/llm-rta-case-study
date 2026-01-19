"""UUniFast-based random tests for RTA robustness."""

import unittest
from rta.generators import uunifast, generate_taskset
from rta.analysis import analyze_taskset


class TestUUniFast(unittest.TestCase):
    """Test UUniFast utilization generation."""
    
    def test_uunifast_sum(self):
        """Test that UUniFast generates utilizations summing to target."""
        target_u = 0.7
        n = 5
        utilizations = uunifast(n, target_u, seed=42)
        
        self.assertEqual(len(utilizations), n)
        self.assertAlmostEqual(sum(utilizations), target_u, places=6)
    
    def test_uunifast_all_positive(self):
        """Test that all generated utilizations are non-negative."""
        utilizations = uunifast(10, 0.8, seed=123)
        for u in utilizations:
            self.assertGreaterEqual(u, 0.0)
    
    def test_uunifast_reproducibility(self):
        """Test that same seed produces same results."""
        u1 = uunifast(5, 0.6, seed=999)
        u2 = uunifast(5, 0.6, seed=999)
        
        for a, b in zip(u1, u2):
            self.assertAlmostEqual(a, b, places=10)
    
    def test_uunifast_invalid_n(self):
        """Test that invalid n raises ValueError."""
        with self.assertRaises(ValueError):
            uunifast(0, 0.5)
        with self.assertRaises(ValueError):
            uunifast(-1, 0.5)
    
    def test_uunifast_invalid_utilization(self):
        """Test that negative utilization raises ValueError."""
        with self.assertRaises(ValueError):
            uunifast(5, -0.1)


class TestTaskSetGenerator(unittest.TestCase):
    """Test random task set generation."""
    
    def test_generate_taskset_count(self):
        """Test that correct number of tasks are generated."""
        n = 7
        taskset = generate_taskset(n, 0.6, seed=42)
        self.assertEqual(len(taskset), n)
    
    def test_generate_taskset_utilization(self):
        """Test that total utilization is approximately correct."""
        target_u = 0.75
        taskset = generate_taskset(10, target_u, seed=123)
        self.assertAlmostEqual(taskset.total_utilization, target_u, places=2)
    
    def test_generate_taskset_periods_in_range(self):
        """Test that generated periods are within specified range."""
        period_min = 50.0
        period_max = 500.0
        taskset = generate_taskset(5, 0.5, period_min=period_min, period_max=period_max, seed=456)
        
        for task in taskset:
            self.assertGreaterEqual(task.T, period_min)
            self.assertLessEqual(task.T, period_max)
    
    def test_generate_taskset_valid_tasks(self):
        """Test that all generated tasks are valid (C <= D <= T)."""
        taskset = generate_taskset(8, 0.8, seed=789)
        
        for task in taskset:
            self.assertLessEqual(task.C, task.D)
            self.assertLessEqual(task.D, task.T)
            self.assertGreater(task.C, 0)
            self.assertGreater(task.T, 0)
    
    def test_generate_taskset_reproducibility(self):
        """Test that same seed produces same task set."""
        ts1 = generate_taskset(5, 0.6, seed=111)
        ts2 = generate_taskset(5, 0.6, seed=111)
        
        for t1, t2 in zip(ts1, ts2):
            self.assertAlmostEqual(t1.C, t2.C, places=10)
            self.assertAlmostEqual(t1.T, t2.T, places=10)
            self.assertAlmostEqual(t1.D, t2.D, places=10)


class TestRandomSchedulability(unittest.TestCase):
    """Test schedulability analysis on randomly generated task sets."""
    
    def test_low_utilization_schedulable(self):
        """Test that low-utilization task sets are typically schedulable."""
        # With U < 0.5, most task sets should be schedulable
        schedulable_count = 0
        num_tests = 20
        
        for i in range(num_tests):
            taskset = generate_taskset(5, 0.4, seed=1000+i)
            schedulable, _ = analyze_taskset(taskset)
            if schedulable:
                schedulable_count += 1
        
        # Expect at least 80% to be schedulable
        self.assertGreater(schedulable_count, num_tests * 0.8)
    
    def test_high_utilization_some_unschedulable(self):
        """Test that high-utilization task sets may be unschedulable."""
        # With U close to 1.0, some task sets should fail
        unschedulable_count = 0
        num_tests = 20
        
        for i in range(num_tests):
            taskset = generate_taskset(10, 0.95, seed=2000+i)
            schedulable, _ = analyze_taskset(taskset)
            if not schedulable:
                unschedulable_count += 1
        
        # Expect at least some to be unschedulable
        self.assertGreater(unschedulable_count, 0)
    
    def test_over_utilization_unschedulable(self):
        """Test that task sets with U > 1 are unschedulable."""
        # Generate task sets with total utilization > 1
        for i in range(10):
            taskset = generate_taskset(5, 1.2, seed=3000+i)
            schedulable, _ = analyze_taskset(taskset)
            # All should be unschedulable (at least the lowest priority task)
            self.assertFalse(schedulable)
    
    def test_single_task_always_schedulable(self):
        """Test that single-task sets are always schedulable if U <= 1."""
        for i in range(10):
            taskset = generate_taskset(1, 0.9, seed=4000+i)
            schedulable, response_times = analyze_taskset(taskset)
            self.assertTrue(schedulable)
            # Response time should equal WCET for single task
            task = taskset.tasks[0]
            self.assertAlmostEqual(response_times[task.name], task.C, places=6)
    
    def test_response_times_bounded(self):
        """Test that response times are bounded by deadlines for schedulable tasks."""
        for i in range(10):
            taskset = generate_taskset(6, 0.7, seed=5000+i)
            schedulable, response_times = analyze_taskset(taskset)
            
            for task in taskset:
                rt = response_times[task.name]
                if rt is not None:
                    # Response time should be at least C and at most D
                    self.assertGreaterEqual(rt, task.C)
                    self.assertLessEqual(rt, task.D)


class TestSchedulabilityExperiment(unittest.TestCase):
    """Test the schedulability vs utilisation experiment."""
    
    def test_experiment_smoke(self):
        """Smoke test: verify schedulability experiment runs without error.
        
        Uses a reduced configuration (fewer utilisation points and task sets)
        to ensure the experiment code is functional.
        """
        try:
            from experiments.sched_util_plot import run_schedulability_experiment
        except ImportError:
            self.skipTest("experiments.sched_util_plot not available")
        
        # Run with minimal configuration
        utilisation_points = [0.3, 0.5, 0.7]
        results = run_schedulability_experiment(
            utilisation_points=utilisation_points,
            num_task_sets_per_point=10,  # Reduced for speed
            num_tasks=3,
            min_period=10.0,
            max_period=100.0,
            seed=12345,
        )
        
        # Verify results structure
        self.assertEqual(len(results), 3)
        self.assertTrue(all(u in results for u in utilisation_points))
        
        # Verify schedulability ratios are in valid range
        for u, ratio in results.items():
            self.assertGreaterEqual(ratio, 0.0, f"Invalid ratio {ratio} for U={u}")
            self.assertLessEqual(ratio, 1.0, f"Invalid ratio {ratio} for U={u}")
        
        # Verify general trend: higher U should have lower or equal schedulability
        # Note: with only 10 samples there can be noise, so we use a weak check
        self.assertGreaterEqual(results[0.3], results[0.7] - 0.3)


if __name__ == "__main__":
    unittest.main()
