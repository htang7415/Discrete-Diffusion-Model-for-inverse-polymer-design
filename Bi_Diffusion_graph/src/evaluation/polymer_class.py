"""Polymer class detection and class-guided design."""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from tqdm import tqdm

from ..utils.chemistry import check_validity, count_stars, compute_sa_score
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
    """Class-guided polymer design."""

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
            sampler: Constrained sampler.
            tokenizer: Tokenizer.
            classifier: Polymer classifier.
            training_smiles: Training SMILES set.
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
        self.evaluator = GenerativeEvaluator(training_smiles)

    def design_by_class(
        self,
        target_class: str,
        num_candidates: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True
    ) -> Dict:
        """Design polymers for a specific class.

        Args:
            target_class: Target polymer class.
            num_candidates: Number of candidates.
            seq_length: Sequence length.
            batch_size: Batch size.
            show_progress: Whether to show progress.

        Returns:
            Dictionary with results.
        """
        if show_progress:
            print(f"Generating {num_candidates} candidates for class: {target_class}")

        # Generate candidates
        _, all_smiles = self.sampler.sample_batch(
            num_candidates, seq_length, batch_size, show_progress
        )

        # Filter valid
        n_total = len(all_smiles)
        n_valid_any = 0
        valid_smiles = []
        for smiles in all_smiles:
            if check_validity(smiles):
                n_valid_any += 1
                if count_stars(smiles) == 2:
                    valid_smiles.append(smiles)

        n_valid_two_stars = len(valid_smiles)
        validity = n_valid_any / n_total if n_total > 0 else 0.0
        validity_two_stars = n_valid_two_stars / n_total if n_total > 0 else 0.0

        if show_progress:
            print(f"Valid candidates: {len(valid_smiles)}")

        if len(valid_smiles) == 0:
            return self._empty_class_results(target_class, num_candidates, validity, validity_two_stars)

        # Filter by class
        class_matches = self.classifier.filter_by_class(valid_smiles, target_class)

        if show_progress:
            print(f"Class matches: {len(class_matches)}")

        # Compute metrics (round floats to 4 decimal places)
        results = {
            "target_class": target_class,
            "n_generated": num_candidates,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": len(valid_smiles),
            "n_class_matches": len(class_matches),
            "class_success_rate": round(len(class_matches) / len(valid_smiles), 4) if valid_smiles else 0.0,
        }

        # Generative metrics on ALL generated samples (star=2 fraction across all)
        gen_metrics = self.evaluator.evaluate(all_smiles, "all_generated", show_progress=False)
        results.update({
            "frac_star_eq_2": gen_metrics["frac_star_eq_2"],
            "uniqueness": gen_metrics["uniqueness"],
            "novelty": gen_metrics["novelty"],
            "avg_diversity": gen_metrics["avg_diversity"],
            "mean_sa": gen_metrics["mean_sa"],
            "std_sa": gen_metrics["std_sa"],
        })

        # Additional metrics for class matches specifically
        if class_matches:
            class_gen_metrics = self.evaluator.evaluate(class_matches, target_class, show_progress=False)
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

        results["class_matches_smiles"] = class_matches

        return results

    def design_joint(
        self,
        target_class: str,
        target_value: float,
        epsilon: float,
        num_candidates: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True
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

        Returns:
            Dictionary with results.
        """
        if self.property_predictor is None:
            raise ValueError("Property predictor required for joint design")

        if show_progress:
            print(f"Joint design: {target_class} with property={target_value}+/-{epsilon}")

        # Generate candidates
        _, all_smiles = self.sampler.sample_batch(
            num_candidates, seq_length, batch_size, show_progress
        )

        # Filter valid
        n_total = len(all_smiles)
        n_valid_any = 0
        valid_smiles = []
        for smiles in all_smiles:
            if check_validity(smiles):
                n_valid_any += 1
                if count_stars(smiles) == 2:
                    valid_smiles.append(smiles)

        n_valid_two_stars = len(valid_smiles)
        validity = n_valid_any / n_total if n_total > 0 else 0.0
        validity_two_stars = n_valid_two_stars / n_total if n_total > 0 else 0.0

        if len(valid_smiles) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates, validity, validity_two_stars)

        # Filter by class
        class_matches = self.classifier.filter_by_class(valid_smiles, target_class)

        if len(class_matches) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates, validity, validity_two_stars,
                                             n_valid=len(valid_smiles))

        # Predict properties for class matches
        predictions = self._predict_batch(class_matches, batch_size)

        # Find joint hits
        joint_hits_mask = np.abs(predictions - target_value) < epsilon
        joint_hits_smiles = [s for s, h in zip(class_matches, joint_hits_mask) if h]
        joint_hits_predictions = predictions[joint_hits_mask]

        # Compute metrics (round floats to 4 decimal places)
        results = {
            "target_class": target_class,
            "target_value": round(target_value, 4),
            "epsilon": round(epsilon, 4),
            "n_generated": num_candidates,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": len(valid_smiles),
            "n_class_matches": len(class_matches),
            "n_joint_hits": len(joint_hits_smiles),
            "class_success_rate": round(len(class_matches) / len(valid_smiles), 4) if valid_smiles else 0.0,
            "joint_success_rate": round(len(joint_hits_smiles) / len(valid_smiles), 4) if valid_smiles else 0.0,
            "pred_mean_class": round(float(np.mean(predictions)), 4),
            "pred_std_class": round(float(np.std(predictions)), 4),
            "pred_mean_joint": round(float(np.mean(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
            "pred_std_joint": round(float(np.std(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
        }

        # SA statistics (round to 4 decimal places)
        if class_matches:
            sa_class = [compute_sa_score(s) for s in class_matches]
            sa_class = [s for s in sa_class if s is not None]
            results["sa_mean_class"] = round(float(np.mean(sa_class)), 4) if sa_class else 0.0
            results["sa_std_class"] = round(float(np.std(sa_class)), 4) if sa_class else 0.0

        if joint_hits_smiles:
            sa_joint = [compute_sa_score(s) for s in joint_hits_smiles]
            sa_joint = [s for s in sa_joint if s is not None]
            results["sa_mean_joint"] = round(float(np.mean(sa_joint)), 4) if sa_joint else 0.0
            results["sa_std_joint"] = round(float(np.std(sa_joint)), 4) if sa_joint else 0.0
        else:
            results["sa_mean_joint"] = 0.0
            results["sa_std_joint"] = 0.0

        results["class_matches_smiles"] = class_matches
        results["joint_hits_smiles"] = joint_hits_smiles

        return results

    def _predict_batch(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Predict properties for SMILES list."""
        self.property_predictor.eval()
        predictions = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
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

    def _empty_class_results(self, target_class: str, n_generated: int, validity: float = 0.0, validity_two_stars: float = 0.0) -> Dict:
        """Return empty results for class design."""
        return {
            "target_class": target_class,
            "n_generated": n_generated,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": 0,
            "n_class_matches": 0,
            "class_success_rate": 0.0,
            "frac_star_eq_2": 0.0,
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
        validity: float = 0.0,
        validity_two_stars: float = 0.0,
        n_valid: int = 0
    ) -> Dict:
        """Return empty results for joint design."""
        return {
            "target_class": target_class,
            "target_value": target_value,
            "epsilon": epsilon,
            "n_generated": n_generated,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
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


class GraphClassGuidedDesigner:
    """Class-guided polymer design for graph models."""

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
        """Initialize graph class-guided designer.

        Args:
            sampler: GraphSampler.
            tokenizer: GraphTokenizer.
            classifier: Polymer classifier.
            training_smiles: Training SMILES set.
            property_predictor: Optional GraphPropertyPredictor.
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
        self.evaluator = GenerativeEvaluator(training_smiles)

    def design_by_class(
        self,
        target_class: str,
        num_candidates: int,
        batch_size: int = 256,
        show_progress: bool = True
    ) -> Dict:
        """Design polymers for a specific class using graph model.

        Args:
            target_class: Target polymer class.
            num_candidates: Number of candidates.
            batch_size: Batch size.
            show_progress: Whether to show progress.

        Returns:
            Dictionary with results.
        """
        if show_progress:
            print(f"Generating {num_candidates} candidates for class: {target_class}")

        # Generate candidates using graph sampler
        all_smiles = self.sampler.sample_batch(
            num_samples=num_candidates,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Filter valid
        n_total = len(all_smiles)
        n_valid_any = 0
        valid_smiles = []
        for smiles in all_smiles:
            if smiles and check_validity(smiles):
                n_valid_any += 1
                if count_stars(smiles) == 2:
                    valid_smiles.append(smiles)

        n_valid_two_stars = len(valid_smiles)
        validity = n_valid_any / n_total if n_total > 0 else 0.0
        validity_two_stars = n_valid_two_stars / n_total if n_total > 0 else 0.0

        if show_progress:
            print(f"Valid candidates: {len(valid_smiles)}")

        if len(valid_smiles) == 0:
            return self._empty_class_results(target_class, num_candidates, validity, validity_two_stars)

        # Filter by class
        class_matches = self.classifier.filter_by_class(valid_smiles, target_class)

        if show_progress:
            print(f"Class matches: {len(class_matches)}")

        # Compute metrics
        results = {
            "target_class": target_class,
            "n_generated": num_candidates,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": len(valid_smiles),
            "n_class_matches": len(class_matches),
            "class_success_rate": round(len(class_matches) / len(valid_smiles), 4) if valid_smiles else 0.0,
        }

        # Generative metrics on ALL generated samples (star=2 fraction across all)
        gen_metrics = self.evaluator.evaluate(all_smiles, "all_generated", show_progress=False)
        results.update({
            "frac_star_eq_2": gen_metrics["frac_star_eq_2"],
            "uniqueness": gen_metrics["uniqueness"],
            "novelty": gen_metrics["novelty"],
            "avg_diversity": gen_metrics["avg_diversity"],
            "mean_sa": gen_metrics["mean_sa"],
            "std_sa": gen_metrics["std_sa"],
        })

        # Additional metrics for class matches
        if class_matches:
            class_gen_metrics = self.evaluator.evaluate(class_matches, target_class, show_progress=False)
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

        results["class_matches_smiles"] = class_matches

        return results

    def design_joint(
        self,
        target_class: str,
        target_value: float,
        epsilon: float,
        num_candidates: int,
        batch_size: int = 256,
        show_progress: bool = True
    ) -> Dict:
        """Joint design: class + property using graph model.

        Args:
            target_class: Target polymer class.
            target_value: Target property value.
            epsilon: Property tolerance.
            num_candidates: Number of candidates.
            batch_size: Batch size.
            show_progress: Whether to show progress.

        Returns:
            Dictionary with results.
        """
        if self.property_predictor is None:
            raise ValueError("Property predictor required for joint design")

        if show_progress:
            print(f"Joint design: {target_class} with property={target_value}+/-{epsilon}")

        # Generate candidates
        all_smiles = self.sampler.sample_batch(
            num_samples=num_candidates,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Filter valid
        n_total = len(all_smiles)
        n_valid_any = 0
        valid_smiles = []
        for smiles in all_smiles:
            if smiles and check_validity(smiles):
                n_valid_any += 1
                if count_stars(smiles) == 2:
                    valid_smiles.append(smiles)

        n_valid_two_stars = len(valid_smiles)
        validity = n_valid_any / n_total if n_total > 0 else 0.0
        validity_two_stars = n_valid_two_stars / n_total if n_total > 0 else 0.0

        if len(valid_smiles) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates, validity, validity_two_stars)

        # Filter by class
        class_matches = self.classifier.filter_by_class(valid_smiles, target_class)

        if len(class_matches) == 0:
            return self._empty_joint_results(target_class, target_value, epsilon, num_candidates, validity, validity_two_stars,
                                             n_valid=len(valid_smiles))

        # Predict properties for class matches
        predictions = self._predict_batch(class_matches, batch_size)

        # Find joint hits
        joint_hits_mask = np.abs(predictions - target_value) < epsilon
        joint_hits_smiles = [s for s, h in zip(class_matches, joint_hits_mask) if h]
        joint_hits_predictions = predictions[joint_hits_mask]

        # Compute metrics
        results = {
            "target_class": target_class,
            "target_value": round(target_value, 4),
            "epsilon": round(epsilon, 4),
            "n_generated": num_candidates,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": len(valid_smiles),
            "n_class_matches": len(class_matches),
            "n_joint_hits": len(joint_hits_smiles),
            "class_success_rate": round(len(class_matches) / len(valid_smiles), 4) if valid_smiles else 0.0,
            "joint_success_rate": round(len(joint_hits_smiles) / len(valid_smiles), 4) if valid_smiles else 0.0,
            "pred_mean_class": round(float(np.mean(predictions)), 4),
            "pred_std_class": round(float(np.std(predictions)), 4),
            "pred_mean_joint": round(float(np.mean(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
            "pred_std_joint": round(float(np.std(joint_hits_predictions)), 4) if len(joint_hits_predictions) > 0 else 0.0,
        }

        # SA statistics
        if class_matches:
            sa_class = [compute_sa_score(s) for s in class_matches]
            sa_class = [s for s in sa_class if s is not None]
            results["sa_mean_class"] = round(float(np.mean(sa_class)), 4) if sa_class else 0.0
            results["sa_std_class"] = round(float(np.std(sa_class)), 4) if sa_class else 0.0

        if joint_hits_smiles:
            sa_joint = [compute_sa_score(s) for s in joint_hits_smiles]
            sa_joint = [s for s in sa_joint if s is not None]
            results["sa_mean_joint"] = round(float(np.mean(sa_joint)), 4) if sa_joint else 0.0
            results["sa_std_joint"] = round(float(np.std(sa_joint)), 4) if sa_joint else 0.0
        else:
            results["sa_mean_joint"] = 0.0
            results["sa_std_joint"] = 0.0

        results["class_matches_smiles"] = class_matches
        results["joint_hits_smiles"] = joint_hits_smiles

        return results

    def _predict_batch(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Predict properties for SMILES list using graph model."""
        self.property_predictor.eval()
        predictions = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]

            # Encode as graphs
            X_list, E_list, M_list = [], [], []
            for smiles in batch:
                try:
                    graph_data = self.tokenizer.encode(smiles)
                    X_list.append(torch.tensor(graph_data['X'], dtype=torch.long))
                    E_list.append(torch.tensor(graph_data['E'], dtype=torch.long))
                    M_list.append(torch.tensor(graph_data['M'], dtype=torch.float))
                except Exception:
                    # Use padding for failed encodes
                    Nmax = self.tokenizer.Nmax
                    pad_id = self.tokenizer.pad_id
                    X_list.append(torch.full((Nmax,), pad_id, dtype=torch.long))
                    E_list.append(torch.zeros((Nmax, Nmax), dtype=torch.long))
                    M_list.append(torch.zeros(Nmax, dtype=torch.float))

            X = torch.stack(X_list).to(self.device)
            E = torch.stack(E_list).to(self.device)
            M = torch.stack(M_list).to(self.device)

            with torch.no_grad():
                preds = self.property_predictor.predict(X, E, M)
            predictions.extend(preds.cpu().numpy().tolist())

        # Denormalize predictions
        predictions = np.array(predictions)
        predictions = predictions * self.normalization_params['std'] + self.normalization_params['mean']
        return predictions

    def _empty_class_results(self, target_class: str, n_generated: int, validity: float = 0.0, validity_two_stars: float = 0.0) -> Dict:
        """Return empty results for class design."""
        return {
            "target_class": target_class,
            "n_generated": n_generated,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
            "n_valid": 0,
            "n_class_matches": 0,
            "class_success_rate": 0.0,
            "frac_star_eq_2": 0.0,
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
        validity: float = 0.0,
        validity_two_stars: float = 0.0,
        n_valid: int = 0
    ) -> Dict:
        """Return empty results for joint design."""
        return {
            "target_class": target_class,
            "target_value": target_value,
            "epsilon": epsilon,
            "n_generated": n_generated,
            "validity": round(validity, 4),
            "validity_two_stars": round(validity_two_stars, 4),
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
