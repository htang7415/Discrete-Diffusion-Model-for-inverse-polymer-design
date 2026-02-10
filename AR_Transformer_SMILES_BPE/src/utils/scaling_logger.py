"""Scaling experiment logger for tracking timing, parameters, and metrics.

This module provides logging functionality for scaling law experiments,
recording model configuration, timing for each step, and metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ScalingLogger:
    """Log scaling experiment results to JSON file.

    Records:
    - Model size and configuration
    - Number of parameters
    - Timing for each pipeline step
    - Metrics from each step
    """

    def __init__(self, results_dir: Path, model_size: str):
        """Initialize the scaling logger.

        Args:
            results_dir: Directory to save results.
            model_size: Model size identifier (small, medium, large, xl).
        """
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / 'scaling_results.json'
        self.results = {
            'model_size': model_size,
            'experiment_start': datetime.now().isoformat(),
            'model_config': {},
            'training_config': {},
            'num_parameters': 0,
            'steps': {}
        }
        self._save()

    def _save(self):
        """Save results to JSON file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _load(self):
        """Load existing results if available."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)

    def log_model_config(self, model_config: Dict[str, Any],
                         training_config: Dict[str, Any],
                         num_params: int):
        """Log model and training configuration.

        Args:
            model_config: Model architecture configuration.
            training_config: Training hyperparameters.
            num_params: Total number of model parameters.
        """
        self.results['model_config'] = model_config
        self.results['training_config'] = training_config
        self.results['num_parameters'] = num_params
        self._save()

    def start_step(self, step_name: str):
        """Mark the start of a pipeline step.

        Args:
            step_name: Name of the step (e.g., 'step1_backbone').
        """
        self.results['steps'][step_name] = {
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'end_time': None,
            'duration_seconds': None,
            'metrics': {}
        }
        self._save()
        print(f"\n{'='*60}")
        print(f"Starting: {step_name}")
        print(f"Time: {self.results['steps'][step_name]['start_time']}")
        print(f"{'='*60}")

    def end_step(self, step_name: str, metrics: Optional[Dict[str, Any]] = None,
                 status: str = 'completed'):
        """Mark the end of a pipeline step.

        Args:
            step_name: Name of the step.
            metrics: Dictionary of metrics from this step.
            status: Status of the step ('completed', 'failed', 'skipped').
        """
        if step_name not in self.results['steps']:
            self.results['steps'][step_name] = {
                'start_time': None,
                'status': status,
            }

        step = self.results['steps'][step_name]
        step['end_time'] = datetime.now().isoformat()
        step['status'] = status

        # Calculate duration if start time exists
        if step.get('start_time'):
            start = datetime.fromisoformat(step['start_time'])
            end = datetime.fromisoformat(step['end_time'])
            step['duration_seconds'] = (end - start).total_seconds()
            duration_str = self._format_duration(step['duration_seconds'])
        else:
            duration_str = 'N/A'

        if metrics:
            step['metrics'] = metrics

        self._save()

        print(f"\n{'='*60}")
        print(f"Completed: {step_name}")
        print(f"Status: {status}")
        print(f"Duration: {duration_str}")
        if metrics:
            print(f"Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        print(f"{'='*60}")

    def update_step_metrics(self, step_name: str, metrics: Dict[str, Any]):
        """Update metrics for a step without changing status.

        Args:
            step_name: Name of the step.
            metrics: Dictionary of metrics to add/update.
        """
        if step_name not in self.results['steps']:
            self.results['steps'][step_name] = {'metrics': {}}

        if 'metrics' not in self.results['steps'][step_name]:
            self.results['steps'][step_name]['metrics'] = {}

        self.results['steps'][step_name]['metrics'].update(metrics)
        self._save()

    def log_error(self, step_name: str, error_message: str):
        """Log an error for a step.

        Args:
            step_name: Name of the step.
            error_message: Error message to log.
        """
        if step_name not in self.results['steps']:
            self.results['steps'][step_name] = {}

        cleaned_error = self._strip_progress_lines(error_message)
        self.results['steps'][step_name]['status'] = 'failed'
        self.results['steps'][step_name]['error'] = cleaned_error
        self.results['steps'][step_name]['end_time'] = datetime.now().isoformat()
        self._save()

    @classmethod
    def _strip_progress_lines(cls, text: Optional[str]) -> str:
        """Remove tqdm-style progress lines from log text."""
        if not text:
            return ""
        normalized = text.replace('\r', '\n')
        filtered = [
            line for line in normalized.splitlines()
            if not cls._is_progress_line(line.lstrip())
        ]
        return "\n".join(filtered).strip()

    @staticmethod
    def _is_progress_line(line: str) -> bool:
        """Detect tqdm-style progress output lines."""
        if not line:
            return False
        if line.startswith("Tokenizing:") or line.startswith("Epoch "):
            return "%|" in line
        return False

    def finalize(self):
        """Finalize the experiment and record total time."""
        self.results['experiment_end'] = datetime.now().isoformat()

        start = datetime.fromisoformat(self.results['experiment_start'])
        end = datetime.fromisoformat(self.results['experiment_end'])
        self.results['total_duration_seconds'] = (end - start).total_seconds()

        # Count completed/failed steps
        completed = sum(1 for s in self.results['steps'].values()
                       if s.get('status') == 'completed')
        failed = sum(1 for s in self.results['steps'].values()
                    if s.get('status') == 'failed')
        total = len(self.results['steps'])

        self.results['summary'] = {
            'total_steps': total,
            'completed_steps': completed,
            'failed_steps': failed,
            'success': failed == 0 and completed == total
        }

        self._save()

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Model size: {self.results['model_size']}")
        print(f"Parameters: {self.results['num_parameters']:,}")
        print(f"Total duration: {self._format_duration(self.results['total_duration_seconds'])}")
        print(f"Steps: {completed}/{total} completed, {failed} failed")
        print(f"Results saved to: {self.results_file}")
        print(f"{'='*60}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def get_step_duration(self, step_name: str) -> Optional[float]:
        """Get duration of a step in seconds.

        Args:
            step_name: Name of the step.

        Returns:
            Duration in seconds, or None if not available.
        """
        if step_name in self.results['steps']:
            return self.results['steps'][step_name].get('duration_seconds')
        return None

    def get_step_metrics(self, step_name: str) -> Dict[str, Any]:
        """Get metrics for a step.

        Args:
            step_name: Name of the step.

        Returns:
            Dictionary of metrics.
        """
        if step_name in self.results['steps']:
            return self.results['steps'][step_name].get('metrics', {})
        return {}

    @classmethod
    def load(cls, results_file: Path) -> 'ScalingLogger':
        """Load an existing scaling logger from file.

        Args:
            results_file: Path to scaling_results.json.

        Returns:
            ScalingLogger instance with loaded data.
        """
        results_dir = results_file.parent
        with open(results_file, 'r') as f:
            data = json.load(f)

        logger = cls.__new__(cls)
        logger.results_dir = results_dir
        logger.results_file = results_file
        logger.results = data
        return logger
