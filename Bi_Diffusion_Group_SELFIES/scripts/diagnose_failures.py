"""
Diagnostic script to analyze Group-SELFIES roundtrip failures.

This script categorizes failures by molecular features to understand
why ~4.5% of molecules fail the roundtrip conversion.

Usage:
    python scripts/diagnose_failures.py --sample_size 100000
"""

import argparse
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import GroupSELFIESTokenizer, _suppress_stdout_stderr

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')


def has_star_in_ring(smiles: str) -> bool:
    """Check if * is part of a ring system."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 53 and atom.GetFormalCharge() == 3:  # [I+3]
            atom_idx = atom.GetIdx()
            for ring in ring_info.AtomRings():
                if atom_idx in ring:
                    return True
    return False


def has_star_adjacent_to_ring(smiles: str) -> bool:
    """Check if * is directly bonded to a ring atom."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 53 and atom.GetFormalCharge() == 3:  # [I+3]
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in ring_atoms:
                    return True
    return False


def count_rings(smiles: str) -> int:
    """Count the number of rings in the molecule."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


def get_ring_sizes(smiles: str) -> list:
    """Get list of ring sizes in the molecule."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return []
    return sorted([len(ring) for ring in mol.GetRingInfo().AtomRings()])


def has_fused_rings(smiles: str) -> bool:
    """Check if molecule has fused ring systems."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    rings = list(ring_info.AtomRings())
    if len(rings) < 2:
        return False
    # Check if any two rings share 2+ atoms (fused)
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = set(rings[i]) & set(rings[j])
            if len(shared) >= 2:
                return True
    return False


def has_spiro(smiles: str) -> bool:
    """Check if molecule has a spiro center (single atom shared between rings)."""
    mol = Chem.MolFromSmiles(smiles.replace("*", "[I+3]"))
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    rings = list(ring_info.AtomRings())
    if len(rings) < 2:
        return False
    # Check if any two rings share exactly 1 atom (spiro)
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = set(rings[i]) & set(rings[j])
            if len(shared) == 1:
                return True
    return False


def has_stereochemistry(smiles: str) -> bool:
    """Check if molecule has stereochemistry (cis/trans or chiral)."""
    return ('/' in smiles or '\\' in smiles or '@' in smiles)


def ring_size_changed(canon_orig: str, canon_dec: str) -> bool:
    """Check if ring sizes changed during roundtrip."""
    orig_sizes = get_ring_sizes(canon_orig)
    dec_sizes = get_ring_sizes(canon_dec)
    return orig_sizes != dec_sizes


def ring_count_changed(canon_orig: str, canon_dec: str) -> bool:
    """Check if number of rings changed during roundtrip."""
    orig_count = count_rings(canon_orig)
    dec_count = count_rings(canon_dec)
    return orig_count != dec_count


def analyze_failure(smiles: str, decoded: str, canon_orig: str, canon_dec: str) -> dict:
    """Analyze a single failure and return categorization."""
    return {
        'star_in_ring': has_star_in_ring(smiles),
        'star_adjacent_to_ring': has_star_adjacent_to_ring(smiles),
        'fused_rings': has_fused_rings(smiles),
        'spiro': has_spiro(smiles),
        'stereochemistry': has_stereochemistry(smiles),
        'ring_count': count_rings(smiles),
        'ring_sizes': get_ring_sizes(smiles),
        'ring_size_changed': ring_size_changed(canon_orig, canon_dec),
        'ring_count_changed': ring_count_changed(canon_orig, canon_dec),
    }


def collect_failures(tokenizer, smiles_list, max_failures=10000, verbose=True):
    """Collect failures with full molecular analysis.

    Args:
        tokenizer: GroupSELFIESTokenizer instance
        smiles_list: List of p-SMILES to test
        max_failures: Maximum number of failures to collect
        verbose: Whether to show progress

    Returns:
        List of failure dicts with analysis
    """
    failures = []
    successes = 0
    total = 0

    iterator = tqdm(smiles_list, desc="Collecting failures") if verbose else smiles_list

    for smiles in iterator:
        total += 1
        try:
            # Tokenize
            tokens = tokenizer.tokenize(smiles)
            if tokens == ['[UNK]']:
                continue

            # Detokenize
            decoded = tokenizer.detokenize(tokens)
            if not decoded:
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': smiles,
                        'error': 'decode_failed',
                        'analysis': analyze_failure(smiles, '', '', '')
                    })
                continue

            # Compare canonical
            smiles_ph = tokenizer._star_to_placeholder(smiles)
            decoded_ph = tokenizer._star_to_placeholder(decoded)

            mol_orig = Chem.MolFromSmiles(smiles_ph)
            mol_dec = Chem.MolFromSmiles(decoded_ph)

            if mol_orig is None or mol_dec is None:
                continue

            canon_orig = Chem.MolToSmiles(mol_orig)
            canon_dec = Chem.MolToSmiles(mol_dec)

            if canon_orig == canon_dec:
                successes += 1
            else:
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': smiles,
                        'decoded': decoded,
                        'canon_orig': canon_orig,
                        'canon_dec': canon_dec,
                        'error': 'canonical_mismatch',
                        'analysis': analyze_failure(smiles, decoded, canon_orig, canon_dec)
                    })

        except Exception as e:
            if len(failures) < max_failures:
                failures.append({
                    'smiles': smiles,
                    'error': f'exception: {str(e)}',
                    'analysis': {}
                })

    return failures, successes, total


def analyze_failures(failures: list) -> dict:
    """Compute statistics on failure patterns."""
    stats = {
        'total_failures': len(failures),
        'by_error_type': Counter(),
        'by_feature': defaultdict(int),
        'ring_size_distribution': Counter(),
        'failures_with_feature': defaultdict(list),
    }

    for f in failures:
        stats['by_error_type'][f.get('error', 'unknown')] += 1

        analysis = f.get('analysis', {})
        for key, value in analysis.items():
            if isinstance(value, bool) and value:
                stats['by_feature'][key] += 1
            elif key == 'ring_count':
                stats['ring_size_distribution'][value] += 1

    # Compute percentages
    total = len(failures) if failures else 1
    stats['feature_percentages'] = {
        k: v / total * 100 for k, v in stats['by_feature'].items()
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Diagnose Group-SELFIES failures")
    parser.add_argument("--sample_size", type=int, default=100000,
                       help="Number of molecules to sample for analysis")
    parser.add_argument("--max_failures", type=int, default=10000,
                       help="Maximum failures to collect")
    parser.add_argument("--output_dir", type=str, default="results/diagnosis",
                       help="Output directory for results")
    parser.add_argument("--tokenizer_path", type=str,
                       default="results/tokenizer.pkl",
                       help="Path to saved tokenizer")
    parser.add_argument("--data_path", type=str,
                       default="Data/Polymer/SMiPoly_polymers.gz",
                       help="Path to data file")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Group-SELFIES Failure Diagnosis")
    print("="*60)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}...")
    tokenizer = GroupSELFIESTokenizer.load(args.tokenizer_path)
    print(f"Vocabulary has {len(tokenizer.vocab)} tokens")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, compression='gzip')
    # Check for SMILES column (may be 'SMILES', 'p-SMILES', or 'p_smiles')
    smiles_col = None
    for col in ['SMILES', 'p-SMILES', 'p_smiles']:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        smiles_col = df.columns[0]
    all_smiles = df[smiles_col].dropna().tolist()
    print(f"Total molecules: {len(all_smiles):,}")
    print(f"SMILES column: {smiles_col}")

    # Filter to p-SMILES (exactly 2 * attachment points)
    p_smiles = [s for s in all_smiles if s.count('*') == 2]
    print(f"p-SMILES (with 2 *): {len(p_smiles):,}")

    # Sample
    import random
    random.seed(42)
    sample_size = min(args.sample_size, len(p_smiles))
    sample_smiles = random.sample(p_smiles, sample_size)
    print(f"Sampled {sample_size:,} p-SMILES for analysis")

    # Collect failures
    print(f"\nCollecting failures (max {args.max_failures:,})...")
    failures, successes, total = collect_failures(
        tokenizer, sample_smiles,
        max_failures=args.max_failures,
        verbose=True
    )

    accuracy = successes / total * 100 if total > 0 else 0
    print(f"\nRoundtrip accuracy: {accuracy:.2f}% ({successes:,}/{total:,})")
    print(f"Failures collected: {len(failures):,}")

    # Analyze failures
    print("\n" + "="*60)
    print("Failure Analysis")
    print("="*60)

    stats = analyze_failures(failures)

    print("\n--- Error Types ---")
    for error_type, count in stats['by_error_type'].most_common():
        pct = count / len(failures) * 100
        print(f"  {error_type}: {count:,} ({pct:.1f}%)")

    print("\n--- Molecular Features in Failures ---")
    for feature, pct in sorted(stats['feature_percentages'].items(), key=lambda x: -x[1]):
        count = stats['by_feature'][feature]
        print(f"  {feature}: {count:,} ({pct:.1f}%)")

    print("\n--- Ring Count Distribution in Failures ---")
    for ring_count, count in sorted(stats['ring_size_distribution'].items()):
        pct = count / len(failures) * 100
        print(f"  {ring_count} rings: {count:,} ({pct:.1f}%)")

    # Save detailed failure data
    failure_df = pd.DataFrame([
        {
            'smiles': f['smiles'],
            'decoded': f.get('decoded', ''),
            'canon_orig': f.get('canon_orig', ''),
            'canon_dec': f.get('canon_dec', ''),
            'error': f.get('error', ''),
            **{f'feat_{k}': v for k, v in f.get('analysis', {}).items()
               if not isinstance(v, list)}
        }
        for f in failures
    ])
    failure_path = output_dir / "failures_detailed.csv"
    failure_df.to_csv(failure_path, index=False)
    print(f"\nDetailed failures saved to: {failure_path}")

    # Save summary statistics
    summary = {
        'sample_size': sample_size,
        'accuracy': accuracy,
        'total_failures': len(failures),
        'error_types': dict(stats['by_error_type']),
        'feature_percentages': stats['feature_percentages'],
        'ring_distribution': dict(stats['ring_size_distribution']),
    }

    import json
    summary_path = output_dir / "diagnosis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "="*60)
    print("Key Findings")
    print("="*60)

    # Identify dominant failure modes
    dominant_features = [
        (f, p) for f, p in stats['feature_percentages'].items() if p > 20
    ]
    if dominant_features:
        print("\nDominant failure patterns (>20% of failures):")
        for feature, pct in sorted(dominant_features, key=lambda x: -x[1]):
            print(f"  - {feature}: {pct:.1f}%")

    # Recommendations
    print("\n--- Recommendations ---")
    if stats['feature_percentages'].get('ring_size_changed', 0) > 50:
        print("  HIGH PRIORITY: Ring size changes are the main issue.")
        print("  -> Focus on ring-aware grammar building")

    if stats['feature_percentages'].get('star_in_ring', 0) > 30:
        print("  HIGH PRIORITY: * inside rings causes many failures.")
        print("  -> Test alternative placeholder atoms")

    if stats['feature_percentages'].get('star_adjacent_to_ring', 0) > 50:
        print("  MEDIUM PRIORITY: * adjacent to rings contributes to failures.")
        print("  -> Pre-canonicalization may help")

    if stats['feature_percentages'].get('stereochemistry', 0) > 20:
        print("  MEDIUM PRIORITY: Stereochemistry loss detected.")
        print("  -> Consider stereo-preserving encoding")


if __name__ == "__main__":
    main()
