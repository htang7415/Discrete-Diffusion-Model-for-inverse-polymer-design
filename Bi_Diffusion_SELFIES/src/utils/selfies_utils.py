"""SELFIES utilities for p-SMILES ↔ SELFIES conversion.

This module provides functions to convert between p-SMILES (polymer SMILES with '*'
attachment points) and SELFIES representation using a placeholder atom.

The conversion uses [I+3] as a placeholder for the '*' character, as validated in
smiles_selfies.ipynb with 100% round-trip accuracy on the PolyInfo dataset.
"""

import selfies as sf
from rdkit import Chem
from typing import Optional, Tuple, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Placeholder for polymer attachment point '*'
PLACEHOLDER_SMILES = "[I+3]"


def psmiles_to_selfies(psmiles: str) -> Optional[str]:
    """Convert p-SMILES to SELFIES using placeholder for '*'.

    Args:
        psmiles: Polymer SMILES string with '*' attachment points.

    Returns:
        SELFIES string, or None if conversion fails.

    Example:
        >>> psmiles_to_selfies("*C*")
        '[I+3][C][I+3]'
        >>> psmiles_to_selfies("*CC(C)C*")
        '[I+3][C][C][Branch1][C][C][C][I+3]'
    """
    if psmiles is None or not isinstance(psmiles, str):
        return None

    try:
        # Replace '*' with placeholder
        smiles_with_placeholder = psmiles.replace("*", PLACEHOLDER_SMILES)

        # Encode to SELFIES
        selfies = sf.encoder(smiles_with_placeholder)

        if selfies is None:
            logger.warning(f"SELFIES encoder returned None for: {psmiles}")
            return None

        return selfies

    except Exception as e:
        logger.warning(f"Failed to convert p-SMILES to SELFIES: {psmiles}, Error: {e}")
        return None


def selfies_to_psmiles(selfies: str) -> Optional[str]:
    """Convert SELFIES to p-SMILES, replacing placeholder with '*'.

    Args:
        selfies: SELFIES string with [I+3] placeholders.

    Returns:
        Polymer SMILES string with '*' attachment points, or None if conversion fails.

    Example:
        >>> selfies_to_psmiles("[I+3][C][I+3]")
        '*C*'
        >>> selfies_to_psmiles("[I+3][C][C][Branch1][C][C][C][I+3]")
        '*CC(C)C*'
    """
    if selfies is None or not isinstance(selfies, str):
        return None

    try:
        # Decode SELFIES to SMILES
        smiles_with_placeholder = sf.decoder(selfies)

        if smiles_with_placeholder is None:
            logger.warning(f"SELFIES decoder returned None for: {selfies}")
            return None

        # Replace placeholder with '*'
        psmiles = smiles_with_placeholder.replace(PLACEHOLDER_SMILES, "*")

        return psmiles

    except Exception as e:
        logger.warning(f"Failed to convert SELFIES to p-SMILES: {selfies}, Error: {e}")
        return None


def canonical_psmiles(psmiles: str) -> Optional[str]:
    """Canonicalize p-SMILES for comparison.

    Args:
        psmiles: Polymer SMILES string with '*' attachment points.

    Returns:
        Canonical p-SMILES, or None if invalid.
    """
    if psmiles is None:
        return None

    try:
        # Replace '*' with placeholder for RDKit processing
        smiles_with_placeholder = psmiles.replace("*", PLACEHOLDER_SMILES)

        # Parse and canonicalize
        mol = Chem.MolFromSmiles(smiles_with_placeholder)
        if mol is None:
            return None

        canonical = Chem.MolToSmiles(mol, canonical=True)

        # Replace placeholder back to '*'
        canonical_psmiles = canonical.replace(PLACEHOLDER_SMILES, "*")

        return canonical_psmiles

    except Exception as e:
        logger.warning(f"Failed to canonicalize p-SMILES: {psmiles}, Error: {e}")
        return None


def verify_roundtrip(psmiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Verify p-SMILES → SELFIES → p-SMILES round-trip conversion.

    Tests that:
    1. p-SMILES converts to SELFIES
    2. SELFIES converts back to p-SMILES
    3. Original and reconstructed p-SMILES are chemically equivalent (canonical forms match)

    Args:
        psmiles: Polymer SMILES string with '*' attachment points.

    Returns:
        Tuple of (success, selfies, reconstructed_psmiles):
        - success: True if round-trip preserves chemical structure
        - selfies: SELFIES representation (or None if conversion failed)
        - reconstructed_psmiles: Reconstructed p-SMILES (or None if conversion failed)

    Example:
        >>> verify_roundtrip("*CC(C)C*")
        (True, '[I+3][C][C][Branch1][C][C][C][I+3]', '*CC(C)C*')
    """
    try:
        # Step 1: p-SMILES → SELFIES
        selfies = psmiles_to_selfies(psmiles)
        if selfies is None:
            return (False, None, None)

        # Step 2: SELFIES → p-SMILES
        reconstructed_psmiles = selfies_to_psmiles(selfies)
        if reconstructed_psmiles is None:
            return (False, selfies, None)

        # Step 3: Compare canonical forms
        original_canonical = canonical_psmiles(psmiles)
        reconstructed_canonical = canonical_psmiles(reconstructed_psmiles)

        if original_canonical is None or reconstructed_canonical is None:
            return (False, selfies, reconstructed_psmiles)

        # Check if canonical forms match
        success = (original_canonical == reconstructed_canonical)

        return (success, selfies, reconstructed_psmiles)

    except Exception as e:
        logger.warning(f"Round-trip verification failed for {psmiles}: {e}")
        return (False, None, None)


def count_placeholder_in_selfies(selfies: str) -> int:
    """Count occurrences of [I+3] placeholder in SELFIES string.

    Args:
        selfies: SELFIES string.

    Returns:
        Number of [I+3] tokens.

    Example:
        >>> count_placeholder_in_selfies("[I+3][C][I+3]")
        2
        >>> count_placeholder_in_selfies("[I+3][C][C][I+3]")
        2
        >>> count_placeholder_in_selfies("[C][C][C]")
        0
    """
    if selfies is None or not isinstance(selfies, str):
        return 0

    try:
        # Split SELFIES into tokens
        tokens = list(sf.split_selfies(selfies))

        # Count occurrences of placeholder token
        count = tokens.count("[I+3]")

        return count

    except Exception as e:
        logger.warning(f"Failed to count placeholders in SELFIES: {selfies}, Error: {e}")
        return 0


def batch_psmiles_to_selfies(psmiles_list: List[str]) -> List[Optional[str]]:
    """Convert batch of p-SMILES to SELFIES.

    Args:
        psmiles_list: List of p-SMILES strings.

    Returns:
        List of SELFIES strings (None for failed conversions).
    """
    return [psmiles_to_selfies(psmiles) for psmiles in psmiles_list]


def batch_selfies_to_psmiles(selfies_list: List[str]) -> List[Optional[str]]:
    """Convert batch of SELFIES to p-SMILES.

    Args:
        selfies_list: List of SELFIES strings.

    Returns:
        List of p-SMILES strings (None for failed conversions).
    """
    return [selfies_to_psmiles(selfies) for selfies in selfies_list]


def parallel_selfies_to_psmiles(
    selfies_list: List[str],
    num_workers: int = 8,
    chunksize: int = 100
) -> List[Optional[str]]:
    """Convert batch of SELFIES to p-SMILES in parallel.

    Args:
        selfies_list: List of SELFIES strings.
        num_workers: Number of parallel workers.
        chunksize: Chunk size for multiprocessing.

    Returns:
        List of p-SMILES strings (None for failed conversions).
    """
    from multiprocessing import Pool

    if len(selfies_list) == 0:
        return []

    # For small lists, use sequential processing (multiprocessing overhead)
    if len(selfies_list) < 100 or num_workers <= 1:
        return batch_selfies_to_psmiles(selfies_list)

    with Pool(processes=num_workers) as pool:
        results = list(pool.imap(selfies_to_psmiles, selfies_list, chunksize=chunksize))

    return results


def parallel_psmiles_to_selfies(
    psmiles_list: List[str],
    num_workers: int = 8,
    chunksize: int = 100
) -> List[Optional[str]]:
    """Convert batch of p-SMILES to SELFIES in parallel.

    Args:
        psmiles_list: List of p-SMILES strings.
        num_workers: Number of parallel workers.
        chunksize: Chunk size for multiprocessing.

    Returns:
        List of SELFIES strings (None for failed conversions).
    """
    from multiprocessing import Pool

    if len(psmiles_list) == 0:
        return []

    # For small lists, use sequential processing (multiprocessing overhead)
    if len(psmiles_list) < 100 or num_workers <= 1:
        return batch_psmiles_to_selfies(psmiles_list)

    with Pool(processes=num_workers) as pool:
        results = list(pool.imap(psmiles_to_selfies, psmiles_list, chunksize=chunksize))

    return results


def validate_selfies_placeholder_count(selfies: str, expected_count: int = 2) -> bool:
    """Validate that SELFIES has expected number of placeholders.

    Args:
        selfies: SELFIES string.
        expected_count: Expected number of [I+3] tokens (default: 2 for polymer repeat units).

    Returns:
        True if placeholder count matches expected count.
    """
    count = count_placeholder_in_selfies(selfies)
    return count == expected_count
