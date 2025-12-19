"""Generative evaluation metrics for polymer generation."""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
from tqdm import tqdm

from ..utils.chemistry import (
    check_validity,
    count_stars,
    compute_sa_score,
    canonicalize_smiles,
    compute_fingerprint,
    compute_pairwise_diversity,
    batch_compute_fingerprints
)


class GenerativeEvaluator:
    """Evaluate generated polymer samples."""

    def __init__(
        self,
        training_smiles: Set[str],
        fp_type: str = "morgan",
        fp_radius: int = 2,
        fp_bits: int = 2048
    ):
        """Initialize evaluator.

        Args:
            training_smiles: Set of training SMILES for novelty computation.
            fp_type: Fingerprint type for diversity.
            fp_radius: Fingerprint radius.
            fp_bits: Number of fingerprint bits.
        """
        self.training_smiles = {canonicalize_smiles(s) or s for s in training_smiles}
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

    def evaluate(
        self,
        generated_smiles: List[str],
        sample_id: str = "sample",
        show_progress: bool = True
    ) -> Dict:
        """Evaluate generated SMILES.

        Args:
            generated_smiles: List of generated SMILES strings.
            sample_id: Identifier for this sample set.
            show_progress: Whether to show progress.

        Returns:
            Dictionary of metrics.
        """
        n_total = len(generated_smiles)

        # Filter valid molecules
        valid_smiles = []
        validity_results = []
        star_counts = []

        iterator = generated_smiles
        if show_progress:
            iterator = tqdm(generated_smiles, desc="Checking validity")

        for smiles in iterator:
            is_valid = check_validity(smiles)
            validity_results.append(is_valid)

            if is_valid:
                valid_smiles.append(smiles)
                star_counts.append(count_stars(smiles))

        n_valid = len(valid_smiles)
        validity = n_valid / n_total if n_total > 0 else 0.0

        # Canonicalize for uniqueness/novelty
        canonical_valid = [canonicalize_smiles(s) or s for s in valid_smiles]

        # Uniqueness
        unique_smiles = list(set(canonical_valid))
        n_unique = len(unique_smiles)
        uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

        # Novelty (unique SMILES not in training)
        novel_smiles = [s for s in unique_smiles if s not in self.training_smiles]
        n_novel = len(novel_smiles)
        novelty = n_novel / n_unique if n_unique > 0 else 0.0

        # Star count distribution
        star_counter = Counter(star_counts)
        frac_star_eq_2 = star_counter.get(2, 0) / n_valid if n_valid > 0 else 0.0

        # SA scores
        sa_scores = []
        if show_progress:
            print("Computing SA scores...")
        for smiles in tqdm(valid_smiles, desc="SA scores", disable=not show_progress):
            sa = compute_sa_score(smiles)
            if sa is not None:
                sa_scores.append(sa)

        sa_stats = self._compute_stats(sa_scores, "sa")

        # Diversity (fingerprint-based)
        if show_progress:
            print("Computing diversity...")
        fingerprints, _ = batch_compute_fingerprints(
            unique_smiles[:min(5000, len(unique_smiles))],  # Limit for speed
            self.fp_type,
            self.fp_radius,
            self.fp_bits
        )

        avg_diversity = 0.0
        if len(fingerprints) >= 2:
            # Sample pairs for efficiency
            if len(fingerprints) > 1000:
                indices = np.random.choice(len(fingerprints), 1000, replace=False)
                sampled_fps = [fingerprints[i] for i in indices]
            else:
                sampled_fps = fingerprints
            avg_diversity = compute_pairwise_diversity(sampled_fps)

        # Length statistics
        lengths = [len(s) for s in valid_smiles]
        length_stats = self._compute_stats(lengths, "length")

        # Compile metrics (round floats to 4 decimal places)
        metrics = {
            "sample_id": sample_id,
            "n_total": n_total,
            "n_valid": n_valid,
            "validity": round(validity, 4),
            "uniqueness": round(uniqueness, 4),
            "novelty": round(novelty, 4),
            "avg_diversity": round(avg_diversity, 4),
            "frac_star_eq_2": round(frac_star_eq_2, 4),
            **sa_stats,
            **length_stats,
            "star_count_distribution": dict(star_counter)
        }

        return metrics

    def _compute_stats(self, values: List[float], prefix: str) -> Dict[str, float]:
        """Compute statistics for a list of values.

        Args:
            values: List of numeric values.
            prefix: Prefix for stat names.

        Returns:
            Dictionary of statistics (rounded to 4 decimal places).
        """
        if not values:
            return {
                f"mean_{prefix}": 0.0,
                f"std_{prefix}": 0.0,
                f"min_{prefix}": 0.0,
                f"max_{prefix}": 0.0
            }

        return {
            f"mean_{prefix}": round(float(np.mean(values)), 4),
            f"std_{prefix}": round(float(np.std(values)), 4),
            f"min_{prefix}": round(float(np.min(values)), 4),
            f"max_{prefix}": round(float(np.max(values)), 4)
        }

    def get_valid_samples(
        self,
        generated_smiles: List[str],
        require_two_stars: bool = True
    ) -> List[str]:
        """Filter to valid samples.

        Args:
            generated_smiles: List of generated SMILES.
            require_two_stars: Whether to require exactly 2 stars.

        Returns:
            List of valid SMILES.
        """
        valid = []
        for smiles in generated_smiles:
            if not check_validity(smiles):
                continue
            if require_two_stars and count_stars(smiles) != 2:
                continue
            valid.append(smiles)
        return valid

    def compute_sa_stats(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute SA score statistics.

        Args:
            smiles_list: List of SMILES.

        Returns:
            SA statistics.
        """
        sa_scores = []
        for smiles in smiles_list:
            sa = compute_sa_score(smiles)
            if sa is not None:
                sa_scores.append(sa)
        return self._compute_stats(sa_scores, "sa")

    def compute_length_stats(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute length statistics.

        Args:
            smiles_list: List of SMILES.

        Returns:
            Length statistics.
        """
        lengths = [len(s) for s in smiles_list]
        return self._compute_stats(lengths, "length")

    def format_metrics_csv(self, metrics: Dict) -> pd.DataFrame:
        """Format metrics as a DataFrame for CSV output.

        Args:
            metrics: Metrics dictionary.

        Returns:
            DataFrame with metrics.
        """
        # Remove non-scalar values
        scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        return pd.DataFrame([scalar_metrics])

    def get_training_sa_scores(self) -> List[float]:
        """Compute SA scores for training set.

        Returns:
            List of SA scores.
        """
        sa_scores = []
        for smiles in tqdm(list(self.training_smiles), desc="Training SA scores"):
            sa = compute_sa_score(smiles)
            if sa is not None:
                sa_scores.append(sa)
        return sa_scores

    def get_training_lengths(self) -> List[int]:
        """Get lengths of training SMILES.

        Returns:
            List of lengths.
        """
        return [len(s) for s in self.training_smiles]
