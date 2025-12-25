"""Polymer class detection and class-guided design.

Generates SELFIES strings, converts to p-SMILES for validation and classification.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from tqdm import tqdm

from ..utils.chemistry import check_validity, count_stars, compute_sa_score
from ..utils.selfies_utils import selfies_to_psmiles
from .generative_metrics import GenerativeEvaluator


class PolymerClassifier:
    """Classify polymers into families using SMARTS patterns."""

    # Default SMARTS patterns for polymer classes
    DEFAULT_PATTERNS = {
        "polyimide": "[#6](=O)-[#7]-[#6](=O)",
        "polyester": "[#6](=O)-[#8]-[#6]",
        "polyamide": "[#6](=O)-[#7]-[#6]",
        "polyurethane": "[#8]-[#6](=O)-[#7]",
        "polyether": "[#6]-[#8]-[#6]",
        "polysiloxane": "[Si]-[#8]-[Si]",
        "polycarbonate": "[#8]-[#6](=O)-[#8]",
        "polysulfone": "[#6]-[S](=O)(=O)-[#6]",
        "polyacrylate": "[#6]-[#6](=O)-[#8]",
        "polystyrene": "[#6]-[#6](c1ccccc1)-[#6]"
    }

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        """Initialize classifier.

        Args:
            patterns: Dictionary of class name -> SMARTS pattern.
        """
        self.patterns = patterns if patterns else self.DEFAULT_PATTERNS
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile SMARTS patterns for efficient matching."""
        self.compiled_patterns = {}
        for name, smarts in self.patterns.items():
            try:
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    self.compiled_patterns[name] = mol
            except Exception:
                print(f"Warning: Could not compile SMARTS pattern for {name}: {smarts}")

    def classify(self, smiles: str) -> Dict[str, bool]:
        """Classify a single SMILES.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary of class name -> match boolean.
        """
        # Replace * with dummy atom for matching
        smiles_clean = smiles.replace('*', '[*]')

        try:
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: False for name in self.patterns}
        except Exception:
            return {name: False for name in self.patterns}

        results = {}
        for name, pattern in self.compiled_patterns.items():
            try:
                results[name] = mol.HasSubstructMatch(pattern)
            except Exception:
                results[name] = False

        return results

    def get_class_label(self, smiles: str) -> Optional[str]:
        """Get primary class label for a SMILES.

        Args:
            smiles: SMILES string.

        Returns:
            Class name or None if no match.
        """
        classification = self.classify(smiles)
        for name, matches in classification.items():
            if matches:
                return name
        return None

    def batch_classify(
        self,
        smiles_list: List[str],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Classify a batch of SMILES.

        Args:
            smiles_list: List of SMILES.
            show_progress: Whether to show progress.

        Returns:
            DataFrame with classification results.
        """
        results = []
        iterator = smiles_list
        if show_progress:
            iterator = tqdm(smiles_list, desc="Classifying")

        for smiles in iterator:
            row = {"smiles": smiles}
            row.update(self.classify(smiles))
            results.append(row)

        return pd.DataFrame(results)

    def filter_by_class(
        self,
        smiles_list: List[str],
        target_class: str
    ) -> List[str]:
        """Filter SMILES by polymer class.

        Args:
            smiles_list: List of SMILES.
            target_class: Target class name.

        Returns:
            List of SMILES matching the class.
        """
        if target_class not in self.patterns:
            raise ValueError(f"Unknown class: {target_class}")

        matched = []
        for smiles in smiles_list:
            classification = self.classify(smiles)
            if classification.get(target_class, False):
                matched.append(smiles)

        return matched


class ClassGuidedDesigner:
    """Class-guided polymer design.

    Generates SELFIES strings, converts to p-SMILES for validation and classification.
    """

    def __init__(
        self,
        sampler,
        tokenizer,
        classifier: PolymerClassifier,
        training_smiles: set,
        property_predictor=None,
        device: str = 'cuda',
        normalization_params: Optional[Dict] = None
    ):
        """Initialize class-guided designer.

        Args:
            sampler: Constrained sampler (generates SELFIES).
            tokenizer: Tokenizer (SelfiesTokenizer).
            classifier: Polymer classifier (uses p-SMILES).
            training_smiles: Training p-SMILES set.
            property_predictor: Optional property predictor.
            device: Device.
            normalization_params: Dict with 'mean' and 'std' for denormalizing predictions.
        """
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.training_smiles = training_smiles
        self.property_predictor = property_predictor
        self.device = device
        self.normalization_params = normalization_params or {'mean': 0.0, 'std': 1.0}
        # Use input_format="selfies" since sampler generates SELFIES
        self.evaluator = GenerativeEvaluator(training_smiles, input_format="selfies")

    def design_by_class(
        self,
        target_class: str,
        num_candidates: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> Dict:
        """Design polymers for a specific class.

        Args:
            target_class: Target polymer class.
            num_candidates: Number of candidates.
            seq_length: Sequence length.
            batch_size: Batch size.
            show_progress: Whether to show progress.
            lengths: Optional list of sequence lengths for each sample.

        Returns:
            Dictionary with results.
        """
        if show_progress:
            print(f"Generating {num_candidates} candidates for class: {target_class}")

        # Generate candidates (sampler returns SELFIES)
        _, all_selfies = self.sampler.sample_batch(
            num_candidates, seq_length, batch_size, show_progress,
            lengths=lengths
        )

        # Convert SELFIES to p-SMILES and filter valid
        valid_selfies = []
        valid_psmiles = []
        for selfies in all_selfies:
            psmiles = selfies_to_psmiles(selfies)
            if psmiles is not None and check_validity(psmiles) and count_stars(psmiles) == 2:
                valid_selfies.append(selfies)
                valid_psmiles.append(psmiles)

        if show_progress:
            print(f"Valid candidates: {len(valid_selfies)}")

        if len(valid_selfies) == 0:
            return self._empty_class_results(target_class, num_candidates)

        # Filter by class (using p-SMILES)
        class_matches_psmiles = self.classifier.filter_by_class(valid_psmiles, target_class)
        # Get corresponding SELFIES
        class_matches_selfies = [valid_selfies[valid_psmiles.index(s)] for s in class_matches_psmiles]

        if show_progress:
            print(f"Class matches: {len(class_matches_psmiles)}")

        # Compute metrics (round floats to 4 decimal places)
        results = {
            "target_class": target_class,
            "n_generated": num_candidates,
            "n_valid": len(valid_selfies),
            "n_class_matches": len(class_matches_psmiles),
            "class_success_rate": round(len(class_matches_psmiles) / len(valid_selfies), 4) if valid_selfies else 0.0,
        }

        # Generative metrics on ALL generated SELFIES (evaluator handles conversion internally)
        gen_metrics = self.evaluator.evaluate(all_selfies, "all_generated", show_progress=False)
        results.update({
            "validity": gen_metrics["validity"],
            "uniqueness": gen_metrics["uniqueness"],
            "novelty": gen_metrics["novelty"],
            "avg_diversity": gen_metrics["avg_diversity"],
            "mean_sa": gen_metrics["mean_sa"],
            "std_sa": gen_metrics["std_sa"],
        })

        # Additional metrics for class matches specifically (use SELFIES for evaluation)
        if class_matches_selfies:
            class_gen_metrics = self.evaluator.evaluate(class_matches_selfies, target_class, show_progress=False)
            results.update({
                "uniqueness_class": class_gen_metrics["uniqueness"],
                "novelty_class": class_gen_metrics["novelty"],
                "avg_diversity_class": class_gen_metrics["avg_diversity"],
                "mean_sa_class": class_gen_metrics["mean_sa"],
                "std_sa_class": class_gen_metrics["std_sa"],
            })
        else:
            results.update({
                "uniqueness_class": 0.0,
                "novelty_class": 0.0,
                "avg_diversity_class": 0.0,
                "mean_sa_class": 0.0,
                "std_sa_class": 0.0,
            })

        # Store both SELFIES and p-SMILES
        results["class_matches_selfies"] = class_matches_selfies
        results["class_matches_smiles"] = class_matches_psmiles  # p-SMILES for backwards compatibility

        return results

    def design_joint(
        self,
        target_class: str,
        target_value: float,
        epsilon: float,
        num_candidates: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> Dict:
        """Joint design: class + property.

        Args:
            target_class: Target polymer class.
            target_value: Target property value.
            epsilon: Property tolerance.
            num_candidates: Number of candidates.
            seq_length: Sequence length.
            batch_size: Batch size.
            show_progress: Whether to show progress.
            lengths: Optional list of sequence lengths for each sample.

        Returns:
            Dictionary with results.
        """
        if self.property_predictor is None:
            raise ValueError("Property predictor required for joint design")

        if show_progress:
            print(f"Joint design: {target_class} with property={target_value}+/-{epsilon}")

        # Generate candidates (sampler returns SELFIES)
        _, all_selfies = self.sampler.sample_batch(
            num_candidates, seq_length, batch_size, show_progress,
            lengths=lengths
        )

        # Convert SELFIES to p-SMILES and filter valid
        valid_selfies = []
        valid_psmiles = []
        for selfies in all_selfies:
            psmiles = selfies_to_psmiles(selfies)
            if psmiles is not None and check_validity(psmiles) and count_stars(psmiles) == 2:
                valid_selfies.append(selfies)
                valid_psmiles.append(psmiles)

        if len(valid_selfies) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates)

        # Filter by class (using p-SMILES)
        class_matches_psmiles = self.classifier.filter_by_class(valid_psmiles, target_class)
        # Get corresponding SELFIES
        class_matches_selfies = [valid_selfies[valid_psmiles.index(s)] for s in class_matches_psmiles]

        if len(class_matches_psmiles) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates,
                                             n_valid=len(valid_selfies))

        # Predict properties for class matches (use SELFIES for tokenization)
        predictions = self._predict_batch(class_matches_selfies, batch_size)

        # Find joint hits
        joint_hits_mask = np.abs(predictions - target_value) < epsilon
        joint_hits_selfies = [s for s, h in zip(class_matches_selfies, joint_hits_mask) if h]
        joint_hits_psmiles = [s for s, h in zip(class_matches_psmiles, joint_hits_mask) if h]
        joint_hits_predictions = predictions[joint_hits_mask]

        # Compute metrics (round floats to 4 decimal places)
        results = {
            "target_class": target_class,
            "target_value": round(target_value, 4),
            "epsilon": round(epsilon, 4),
            "n_generated": num_candidates,
            "n_valid": len(valid_selfies),
            "n_class_matches": len(class_matches_psmiles),
            "n_joint_hits": len(joint_hits_psmiles),
            "class_success_rate": round(len(class_matches_psmiles) / len(valid_selfies), 4) if valid_selfies else 0.0,
            "joint_success_rate": round(len(joint_hits_psmiles) / len(valid_selfies), 4) if valid_selfies else 0.0,
            "pred_mean_class": round(float(np.mean(predictions)), 4),
            "pred_std_class": round(float(np.std(predictions)), 4),
            "pred_mean_joint": round(float(np.mean(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
            "pred_std_joint": round(float(np.std(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
        }

        # SA statistics (use p-SMILES for SA computation)
        if class_matches_psmiles:
            sa_class = [compute_sa_score(s) for s in class_matches_psmiles]
            sa_class = [s for s in sa_class if s is not None]
            results["sa_mean_class"] = round(float(np.mean(sa_class)), 4) if sa_class else 0.0
            results["sa_std_class"] = round(float(np.std(sa_class)), 4) if sa_class else 0.0

        if joint_hits_psmiles:
            sa_joint = [compute_sa_score(s) for s in joint_hits_psmiles]
            sa_joint = [s for s in sa_joint if s is not None]
            results["sa_mean_joint"] = round(float(np.mean(sa_joint)), 4) if sa_joint else 0.0
            results["sa_std_joint"] = round(float(np.std(sa_joint)), 4) if sa_joint else 0.0
        else:
            results["sa_mean_joint"] = 0.0
            results["sa_std_joint"] = 0.0

        # Store both SELFIES and p-SMILES
        results["class_matches_selfies"] = class_matches_selfies
        results["class_matches_smiles"] = class_matches_psmiles  # p-SMILES for backwards compatibility
        results["joint_hits_selfies"] = joint_hits_selfies
        results["joint_hits_smiles"] = joint_hits_psmiles  # p-SMILES for backwards compatibility

        return results

    def _predict_batch(self, selfies_list: List[str], batch_size: int) -> np.ndarray:
        """Predict properties for SELFIES list."""
        self.property_predictor.eval()
        predictions = []

        for i in range(0, len(selfies_list), batch_size):
            batch = selfies_list[i:i+batch_size]
            encoded = self.tokenizer.batch_encode(batch)
            input_ids = torch.tensor(encoded['input_ids'], device=self.device)
            attention_mask = torch.tensor(encoded['attention_mask'], device=self.device)

            with torch.no_grad():
                preds = self.property_predictor.predict(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy().tolist())

        # Denormalize predictions to original scale
        predictions = np.array(predictions)
        predictions = predictions * self.normalization_params['std'] + self.normalization_params['mean']
        return predictions

    def _empty_class_results(self, target_class: str, n_generated: int) -> Dict:
        """Return empty results for class design."""
        return {
            "target_class": target_class,
            "n_generated": n_generated,
            "n_valid": 0,
            "n_class_matches": 0,
            "class_success_rate": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
            "avg_diversity": 0.0,
            "mean_sa": 0.0,
            "std_sa": 0.0,
            "uniqueness_class": 0.0,
            "novelty_class": 0.0,
            "avg_diversity_class": 0.0,
            "mean_sa_class": 0.0,
            "std_sa_class": 0.0,
            "class_matches_smiles": []
        }

    def _empty_joint_results(
        self,
        target_class: str,
        target_value: float,
        epsilon: float,
        n_generated: int,
        n_valid: int = 0
    ) -> Dict:
        """Return empty results for joint design."""
        return {
            "target_class": target_class,
            "target_value": target_value,
            "epsilon": epsilon,
            "n_generated": n_generated,
            "n_valid": n_valid,
            "n_class_matches": 0,
            "n_joint_hits": 0,
            "class_success_rate": 0.0,
            "joint_success_rate": 0.0,
            "pred_mean_class": 0.0,
            "pred_std_class": 0.0,
            "pred_mean_joint": 0.0,
            "pred_std_joint": 0.0,
            "sa_mean_class": 0.0,
            "sa_std_class": 0.0,
            "sa_mean_joint": 0.0,
            "sa_std_joint": 0.0,
            "class_matches_smiles": [],
            "joint_hits_smiles": []
        }
