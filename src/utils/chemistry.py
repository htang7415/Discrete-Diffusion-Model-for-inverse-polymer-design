"""Chemistry utilities using RDKit."""

from typing import Optional, List, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

# Import SA score calculator
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError:
    sascorer = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES or None if parsing fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def check_validity(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit.

    Args:
        smiles: Input SMILES string.

    Returns:
        True if valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def count_stars(smiles: str) -> int:
    """Count the number of '*' characters in a SMILES string.

    Args:
        smiles: Input SMILES string.

    Returns:
        Number of '*' characters.
    """
    return smiles.count('*')


def is_valid_psmiles(smiles: str) -> bool:
    """Check if a p-SMILES is valid (parseable and has exactly 2 stars).

    Args:
        smiles: Input p-SMILES string.

    Returns:
        True if valid p-SMILES with exactly 2 stars.
    """
    if not check_validity(smiles):
        return False
    return count_stars(smiles) == 2


def compute_sa_score(smiles: str) -> Optional[float]:
    """Compute RDKit synthetic accessibility score.

    Args:
        smiles: Input SMILES string.

    Returns:
        SA score (1-10, lower is better) or None if computation fails.
    """
    if sascorer is None:
        # Fallback: use a simple approximation based on molecular properties
        return compute_sa_score_fallback(smiles)

    try:
        # Replace * with H for SA score calculation
        smiles_clean = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            # Try original SMILES
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def compute_sa_score_fallback(smiles: str) -> Optional[float]:
    """Fallback SA score calculation using molecular descriptors.

    Args:
        smiles: Input SMILES string.

    Returns:
        Approximate SA score or None if computation fails.
    """
    try:
        # Replace * with H for calculation
        smiles_clean = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Simple approximation based on molecular weight and complexity
        mw = Descriptors.MolWt(mol)
        num_rings = Descriptors.RingCount(mol)
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        num_heavy = mol.GetNumHeavyAtoms()

        # Normalize to 1-10 scale (rough approximation)
        score = 1.0 + (mw / 500.0) + (num_rings * 0.5) + (num_rotatable * 0.1)
        score = min(max(score, 1.0), 10.0)
        return score
    except Exception:
        return None


def compute_fingerprint(
    smiles: str,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048
) -> Optional[np.ndarray]:
    """Compute molecular fingerprint.

    Args:
        smiles: Input SMILES string.
        fp_type: Type of fingerprint ('morgan' or 'maccs').
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits for fingerprint.

    Returns:
        Fingerprint as numpy array or None if computation fails.
    """
    try:
        # Replace * with H for fingerprint calculation
        smiles_clean = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.

    Returns:
        Tanimoto similarity (0-1).
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return intersection / union


def compute_pairwise_diversity(fingerprints: List[np.ndarray]) -> float:
    """Compute average pairwise Tanimoto distance.

    Args:
        fingerprints: List of fingerprints.

    Returns:
        Average pairwise distance (1 - similarity).
    """
    n = len(fingerprints)
    if n < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_tanimoto_similarity(fingerprints[i], fingerprints[j])
            total_distance += (1.0 - sim)
            count += 1

    return total_distance / count if count > 0 else 0.0


def batch_compute_fingerprints(
    smiles_list: List[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048
) -> Tuple[List[np.ndarray], List[int]]:
    """Compute fingerprints for a batch of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        fp_type: Type of fingerprint.
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits.

    Returns:
        Tuple of (valid fingerprints, valid indices).
    """
    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(smiles_list):
        fp = compute_fingerprint(smiles, fp_type, radius, n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    return fingerprints, valid_indices
