"""Property-guided inverse design (generate-then-filter).

Generates SELFIES strings, converts to p-SMILES for validation and metrics.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from ..utils.chemistry import check_validity, count_stars, compute_sa_score
from ..utils.selfies_utils import selfies_to_psmiles
from .generative_metrics import GenerativeEvaluator


class InverseDesigner:
    """Inverse design via generate-then-filter approach.

    Generates SELFIES strings, converts to p-SMILES for validation and metrics.
    """

    def __init__(
        self,
        sampler,
        property_predictor,
        tokenizer,
        training_smiles: set,
        device: str = 'cuda',
        normalization_params: Optional[Dict] = None
    ):
        """Initialize inverse designer.

        Args:
            sampler: Constrained sampler for generation (generates SELFIES).
            property_predictor: Property prediction model.
            tokenizer: Tokenizer instance (SelfiesTokenizer).
            training_smiles: Set of training p-SMILES for novelty.
            device: Device for computation.
            normalization_params: Dict with 'mean' and 'std' for denormalizing predictions.
        """
        self.sampler = sampler
        self.property_predictor = property_predictor
        self.tokenizer = tokenizer
        self.training_smiles = training_smiles
        self.device = device
        self.normalization_params = normalization_params or {'mean': 0.0, 'std': 1.0}
        # Use input_format="selfies" since sampler generates SELFIES
        self.evaluator = GenerativeEvaluator(training_smiles, input_format="selfies")

    def design(
        self,
        target_value: float,
        epsilon: float,
        num_candidates: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> Dict:
        """Run inverse design for a target property value.

        Args:
            target_value: Target property value.
            epsilon: Tolerance for property matching.
            num_candidates: Number of candidates to generate.
            seq_length: Sequence length.
            batch_size: Batch size for generation.
            show_progress: Whether to show progress.
            lengths: Optional list of sequence lengths for each sample.
                    If provided, samples use training length distribution.

        Returns:
            Dictionary with results.
        """
        # Generate candidates (sampler returns SELFIES)
        if show_progress:
            print(f"Generating {num_candidates} candidates...")

        _, all_selfies = self.sampler.sample_batch(
            num_candidates, seq_length, batch_size, show_progress,
            lengths=lengths
        )

        # Convert SELFIES to p-SMILES and filter to valid molecules with exactly 2 stars
        valid_selfies = []
        valid_psmiles = []
        for selfies in all_selfies:
            psmiles = selfies_to_psmiles(selfies)
            if psmiles is not None and check_validity(psmiles) and count_stars(psmiles) == 2:
                valid_selfies.append(selfies)
                valid_psmiles.append(psmiles)

        if show_progress:
            print(f"Valid candidates: {len(valid_selfies)} / {num_candidates}")

        if len(valid_selfies) == 0:
            return self._empty_results(target_value, epsilon, num_candidates)

        # Predict properties (use SELFIES for tokenization)
        predictions = self._predict_batch(valid_selfies, batch_size, show_progress)

        # Find hits
        hits_mask = np.abs(predictions - target_value) < epsilon
        hits_selfies = [s for s, h in zip(valid_selfies, hits_mask) if h]
        hits_psmiles = [s for s, h in zip(valid_psmiles, hits_mask) if h]
        hits_predictions = predictions[hits_mask]

        # Compute metrics (round floats to 4 decimal places)
        results = {
            "target_value": round(target_value, 4),
            "epsilon": round(epsilon, 4),
            "n_generated": num_candidates,
            "n_valid": len(valid_selfies),
            "n_hits": len(hits_selfies),
            "success_rate": round(len(hits_selfies) / len(valid_selfies), 4) if valid_selfies else 0.0,
            "pred_mean_valid": round(float(np.mean(predictions)), 4),
            "pred_std_valid": round(float(np.std(predictions)), 4),
            "pred_mean_hits": round(float(np.mean(hits_predictions)), 4) if len(hits_predictions) > 0 else 0.0,
            "pred_std_hits": round(float(np.std(hits_predictions)), 4) if len(hits_predictions) > 0 else 0.0,
        }

        # Generative metrics for all generated SELFIES (evaluator handles conversion internally)
        gen_metrics = self.evaluator.evaluate(all_selfies, "all_generated", show_progress=False)
        results.update({
            "validity": gen_metrics["validity"],
            "uniqueness": gen_metrics["uniqueness"],
            "novelty": gen_metrics["novelty"],
            "avg_diversity": gen_metrics["avg_diversity"],
        })

        # SA statistics (use p-SMILES for SA computation)
        sa_valid = self._compute_sa_scores(valid_psmiles)
        results.update({
            "sa_mean_valid": round(float(np.mean(sa_valid)), 4) if sa_valid else 0.0,
            "sa_std_valid": round(float(np.std(sa_valid)), 4) if sa_valid else 0.0,
        })

        if hits_psmiles:
            sa_hits = self._compute_sa_scores(hits_psmiles)
            results.update({
                "sa_mean_hits": round(float(np.mean(sa_hits)), 4) if sa_hits else 0.0,
                "sa_std_hits": round(float(np.std(sa_hits)), 4) if sa_hits else 0.0,
            })
        else:
            results.update({"sa_mean_hits": 0.0, "sa_std_hits": 0.0})

        # Store samples (both SELFIES and p-SMILES)
        results["valid_selfies"] = valid_selfies
        results["valid_smiles"] = valid_psmiles  # p-SMILES for backwards compatibility
        results["predictions"] = [round(p, 4) for p in predictions.tolist()]
        results["hits_selfies"] = hits_selfies
        results["hits_smiles"] = hits_psmiles  # p-SMILES for backwards compatibility
        results["hits_predictions"] = [round(p, 4) for p in hits_predictions.tolist()] if len(hits_predictions) > 0 else []

        return results

    def design_multiple_targets(
        self,
        target_values: List[float],
        epsilon: float,
        num_candidates_per_target: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Run inverse design for multiple target values.

        Args:
            target_values: List of target values.
            epsilon: Tolerance.
            num_candidates_per_target: Candidates per target.
            seq_length: Sequence length.
            batch_size: Batch size.
            show_progress: Whether to show progress.
            lengths: Optional list of sequence lengths for all samples.
                    If provided, will be split across targets.

        Returns:
            DataFrame with results for all targets.
        """
        all_results = []

        for i, target in enumerate(tqdm(target_values, desc="Targets", disable=not show_progress)):
            # Get lengths for this target's candidates
            target_lengths = None
            if lengths is not None:
                start_idx = i * num_candidates_per_target
                end_idx = start_idx + num_candidates_per_target
                target_lengths = lengths[start_idx:end_idx]

            results = self.design(
                target_value=target,
                epsilon=epsilon,
                num_candidates=num_candidates_per_target,
                seq_length=seq_length,
                batch_size=batch_size,
                show_progress=False,
                lengths=target_lengths
            )

            # Remove large lists for CSV
            csv_results = {k: v for k, v in results.items()
                          if k not in ["valid_smiles", "predictions", "hits_smiles", "hits_predictions"]}
            all_results.append(csv_results)

        return pd.DataFrame(all_results)

    def _predict_batch(
        self,
        selfies_list: List[str],
        batch_size: int,
        show_progress: bool = True
    ) -> np.ndarray:
        """Predict properties for a batch of SELFIES.

        Args:
            selfies_list: List of SELFIES strings.
            batch_size: Batch size.
            show_progress: Whether to show progress.

        Returns:
            Array of predictions.
        """
        self.property_predictor.eval()
        predictions = []

        num_batches = (len(selfies_list) + batch_size - 1) // batch_size
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for i in iterator:
            start = i * batch_size
            end = min(start + batch_size, len(selfies_list))
            batch_selfies = selfies_list[start:end]

            # Encode SELFIES
            encoded = self.tokenizer.batch_encode(batch_selfies)
            input_ids = torch.tensor(encoded['input_ids'], device=self.device)
            attention_mask = torch.tensor(encoded['attention_mask'], device=self.device)

            # Predict
            with torch.no_grad():
                preds = self.property_predictor.predict(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy().tolist())

        # Denormalize predictions to original scale
        predictions = np.array(predictions)
        predictions = predictions * self.normalization_params['std'] + self.normalization_params['mean']
        return predictions

    def _compute_sa_scores(self, smiles_list: List[str]) -> List[float]:
        """Compute SA scores.

        Args:
            smiles_list: List of SMILES.

        Returns:
            List of SA scores.
        """
        scores = []
        for smiles in smiles_list:
            sa = compute_sa_score(smiles)
            if sa is not None:
                scores.append(sa)
        return scores

    def _empty_results(
        self,
        target_value: float,
        epsilon: float,
        n_generated: int
    ) -> Dict:
        """Return empty results when no valid samples.

        Args:
            target_value: Target value.
            epsilon: Tolerance.
            n_generated: Number generated.

        Returns:
            Empty results dictionary.
        """
        return {
            "target_value": target_value,
            "epsilon": epsilon,
            "n_generated": n_generated,
            "n_valid": 0,
            "n_hits": 0,
            "success_rate": 0.0,
            "pred_mean_valid": 0.0,
            "pred_std_valid": 0.0,
            "pred_mean_hits": 0.0,
            "pred_std_hits": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
            "avg_diversity": 0.0,
            "sa_mean_valid": 0.0,
            "sa_std_valid": 0.0,
            "sa_mean_hits": 0.0,
            "sa_std_hits": 0.0,
            "valid_selfies": [],
            "valid_smiles": [],  # p-SMILES for backwards compatibility
            "predictions": [],
            "hits_selfies": [],
            "hits_smiles": [],  # p-SMILES for backwards compatibility
            "hits_predictions": []
        }
