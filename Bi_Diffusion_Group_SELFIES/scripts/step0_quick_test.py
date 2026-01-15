#!/usr/bin/env python
"""
Quick test for Group-SELFIES tokenizer fixes on sampled data.

This script validates the tokenizer fixes on a subset before running the full pipeline:
1. Sample molecules from full dataset
2. Build grammar with fixes (no placeholder filtering, frequency-based capping)
3. Test roundtrip on held-out sample
4. Report failure modes and accuracy

Usage:
    python scripts/step0_quick_test.py --sample-size 500000 --test-size 50000 --max-groups 20000
"""

import os
import sys
import argparse
import random
import gzip
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
import selfies as sf
from group_selfies import GroupGrammar, fragment_mols, Group
from group_selfies.utils import fragment_utils as fu

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Patches and helpers from notebook
# ============================================================================

def select_diverse_set_simple(l, k, weights=None):
    """Simple non-recursive replacement for fragment_utils.select_diverse_set."""
    if not l:
        return []
    if k >= len(l):
        return list(l)
    if weights is not None:
        items = list(zip(l, weights))
        items.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in items[:k]]
    return list(l)[:k]


# Patch fragment_utils
fu.select_diverse_set = select_diverse_set_simple


# Placeholder config (from notebook)
PLACEHOLDER_SMILES = "[I+3]"


def star_to_placeholder(s: str) -> str:
    """Replace '*' with placeholder SMILES atom."""
    return s.replace("*", PLACEHOLDER_SMILES)


def placeholder_to_star(s: str) -> str:
    """Replace placeholder atom back to '*'."""
    return s.replace(PLACEHOLDER_SMILES, "*")


def canonical_star_smiles(s_star: str) -> Optional[str]:
    """Canonicalize a polymer SMILES containing '*'."""
    s_ph = star_to_placeholder(s_star)
    m = Chem.MolFromSmiles(s_ph)
    if m is None:
        return None
    s_ph_canon = Chem.MolToSmiles(m, canonical=True)
    return placeholder_to_star(s_ph_canon)


def verify_placeholder_compatibility():
    """Verify placeholder works with SELFIES and RDKit (from notebook)."""
    print(f"Verifying placeholder '{PLACEHOLDER_SMILES}' compatibility...")

    # Check SELFIES can handle it
    ph_sf = sf.encoder(PLACEHOLDER_SMILES)
    if ph_sf is None:
        raise ValueError(f"SELFIES encoder cannot handle {PLACEHOLDER_SMILES}")

    # Check roundtrip
    try:
        ph_dec = sf.decoder(ph_sf)
        if Chem.MolFromSmiles(ph_dec) is None:
            raise ValueError(f"Decoded placeholder {ph_dec} is not RDKit-parseable")
    except Exception as e:
        raise ValueError(f"SELFIES round-trip failed for placeholder {PLACEHOLDER_SMILES}: {e}")

    print(f"  Placeholder OK: {PLACEHOLDER_SMILES} -> {ph_sf} -> {ph_dec}")
    return True


# ============================================================================
# Data loading
# ============================================================================

def load_polymer_data(data_path: str) -> pd.DataFrame:
    """Load polymer SMILES data from gzipped or regular CSV."""
    data_path = Path(data_path)

    if data_path.suffix == '.gz':
        df = pd.read_csv(data_path, compression='gzip')
    else:
        df = pd.read_csv(data_path)

    # Expect 'p_smiles' column or 'SMILES' column
    if 'p_smiles' in df.columns:
        smiles_col = 'p_smiles'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        raise ValueError(f"No SMILES column found in {data_path}")

    return df, smiles_col


def sample_smiles(smiles_list: List[str], n: int, seed: int = 42) -> List[str]:
    """Random sample of n SMILES from list."""
    random.seed(seed)
    if n >= len(smiles_list):
        return smiles_list
    return random.sample(smiles_list, n)


# ============================================================================
# Grammar building (with fixes)
# ============================================================================

def build_grammar_with_fixes(
    smiles_list: List[str],
    max_groups: int = 20000,
    use_frequency_capping: bool = True,
    filter_placeholder_groups: bool = False,  # KEY FIX: default False
    verbose: bool = True
) -> Tuple[GroupGrammar, List[str], Dict]:
    """
    Build grammar with fixes applied:
    1. No placeholder filtering (like notebook)
    2. Frequency-based group capping (keep most common groups)

    Returns:
        grammar: GroupGrammar object
        groups_used: List of group SMILES used
        stats: Dict with statistics
    """
    stats = {
        'input_smiles': len(smiles_list),
        'valid_mols': 0,
        'total_fragments': 0,
        'unique_fragments': 0,
        'placeholder_fragments': 0,
        'groups_after_cap': 0,
    }

    # Convert to placeholder SMILES and parse with RDKit
    mols = []
    valid_smiles = []

    if verbose:
        print(f"Parsing {len(smiles_list)} SMILES...")

    for s in tqdm(smiles_list, disable=not verbose, desc="Parsing SMILES"):
        s_ph = star_to_placeholder(s)
        m = Chem.MolFromSmiles(s_ph)
        if m is not None:
            mols.append(m)
            valid_smiles.append(s)

    stats['valid_mols'] = len(mols)

    if not mols:
        raise ValueError("No valid molecules found!")

    if verbose:
        print(f"Valid molecules: {len(mols)}/{len(smiles_list)}")

    # Fragment molecules
    if verbose:
        print("Fragmenting molecules...")

    raw_groups = fragment_mols(mols)

    if not raw_groups:
        raise RuntimeError("fragment_mols returned no groups!")

    stats['total_fragments'] = len(raw_groups)

    # Count placeholder-containing groups
    placeholder_groups = [g for g in raw_groups if PLACEHOLDER_SMILES in g]
    stats['placeholder_fragments'] = len(placeholder_groups)

    if verbose:
        print(f"Total fragments: {len(raw_groups)}")
        print(f"Fragments with placeholder: {len(placeholder_groups)}")

    # Optionally filter placeholder groups (NOT recommended - this is the bug!)
    if filter_placeholder_groups:
        if verbose:
            print("WARNING: Filtering placeholder-containing groups (this may cause decode failures)")
        raw_groups = [g for g in raw_groups if PLACEHOLDER_SMILES not in g]
    else:
        if verbose:
            print("Keeping placeholder-containing groups (like notebook)")

    # Get unique groups
    unique_groups = list(dict.fromkeys(raw_groups))  # Preserve order, remove duplicates
    stats['unique_fragments'] = len(unique_groups)

    if verbose:
        print(f"Unique fragments: {len(unique_groups)}")

    # Apply frequency-based capping
    if use_frequency_capping and max_groups and len(unique_groups) > max_groups:
        if verbose:
            print(f"Applying frequency-based capping to {max_groups} groups...")

        # Count frequency of each group
        group_counts = Counter(raw_groups)

        # Sort by frequency (most common first)
        sorted_groups = [g for g, _ in group_counts.most_common()]

        # Take top max_groups
        capped_groups = sorted_groups[:max_groups]

        if verbose:
            print(f"Top group frequencies: {[group_counts[g] for g in capped_groups[:5]]}")
    else:
        capped_groups = unique_groups[:max_groups] if max_groups else unique_groups

    stats['groups_after_cap'] = len(capped_groups)

    if verbose:
        print(f"Groups after capping: {len(capped_groups)}")

    # Build grammar
    groups = [Group(name=f"G{i}", canonsmiles=g) for i, g in enumerate(capped_groups)]
    grammar = GroupGrammar(groups)

    if verbose:
        print(f"Grammar built with {len(groups)} groups")

    return grammar, capped_groups, stats


# ============================================================================
# Roundtrip testing with failure instrumentation
# ============================================================================

def test_roundtrip(
    smiles_list: List[str],
    grammar: GroupGrammar,
    verbose: bool = True,
    max_failures: int = 20
) -> Tuple[Dict, List[Dict]]:
    """
    Test roundtrip on SMILES list with detailed failure tracking.

    Returns:
        stats: Dict with counts for each failure mode
        failures: List of failure examples
    """
    stats = {
        'total': len(smiles_list),
        'success': 0,
        'rdkit_parse_fail': 0,
        'encode_fail': 0,
        'decode_none': 0,
        'canonical_mismatch': 0,
        'exception': 0,
    }

    failures = []

    for s_star in tqdm(smiles_list, disable=not verbose, desc="Testing roundtrip"):
        try:
            # Step 1: Parse original
            s_ph = star_to_placeholder(s_star)
            m_orig = Chem.MolFromSmiles(s_ph)

            if m_orig is None:
                stats['rdkit_parse_fail'] += 1
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': s_star,
                        'error': 'rdkit_parse_fail',
                        'detail': 'RDKit could not parse placeholder SMILES'
                    })
                continue

            # Step 2: Encode to Group SELFIES
            try:
                gsf = grammar.full_encoder(m_orig)
            except Exception as e:
                stats['encode_fail'] += 1
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': s_star,
                        'error': 'encode_fail',
                        'detail': str(e)
                    })
                continue

            # Step 3: Decode back
            m_dec = grammar.decoder(gsf)

            if m_dec is None:
                stats['decode_none'] += 1
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': s_star,
                        'gsf': gsf,
                        'error': 'decode_none',
                        'detail': 'grammar.decoder() returned None'
                    })
                continue

            # Step 4: Compare canonical forms
            s_ph_dec = Chem.MolToSmiles(m_dec, canonical=True)
            s_star_dec = placeholder_to_star(s_ph_dec)

            s_orig_canon = canonical_star_smiles(s_star)
            s_dec_canon = canonical_star_smiles(s_star_dec)

            if s_orig_canon is None or s_dec_canon is None:
                stats['canonical_mismatch'] += 1
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': s_star,
                        'decoded': s_star_dec,
                        'error': 'canonical_mismatch',
                        'detail': 'Could not canonicalize one or both SMILES'
                    })
                continue

            if s_orig_canon == s_dec_canon:
                stats['success'] += 1
            else:
                stats['canonical_mismatch'] += 1
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': s_star,
                        'decoded': s_star_dec,
                        'canon_orig': s_orig_canon,
                        'canon_dec': s_dec_canon,
                        'gsf': gsf,
                        'error': 'canonical_mismatch',
                        'detail': 'Canonical SMILES do not match'
                    })

        except Exception as e:
            stats['exception'] += 1
            if len(failures) < max_failures:
                failures.append({
                    'smiles': s_star,
                    'error': 'exception',
                    'detail': str(e)
                })

    return stats, failures


# ============================================================================
# Main
# ============================================================================

def main(args):
    print("=" * 60)
    print("Group-SELFIES Quick Test")
    print("=" * 60)
    print(f"Sample size for grammar: {args.sample_size:,}")
    print(f"Test size for roundtrip: {args.test_size:,}")
    print(f"Max groups: {args.max_groups:,}")
    print(f"Seed: {args.seed}")
    print(f"Filter placeholder groups: {args.filter_placeholder}")
    print("=" * 60)

    # Set random seed
    random.seed(args.seed)

    # Verify placeholder compatibility
    verify_placeholder_compatibility()

    # Load data
    print(f"\n1. Loading data from {args.data_path}...")
    df, smiles_col = load_polymer_data(args.data_path)
    all_smiles = df[smiles_col].dropna().astype(str).unique().tolist()
    print(f"   Total unique SMILES: {len(all_smiles):,}")

    # Quick check: placeholder shouldn't appear in data (sample-based check)
    sample_for_check = all_smiles[:10000]  # Check first 10K only
    placeholder_count = sum(1 for s in sample_for_check if PLACEHOLDER_SMILES in s)
    if placeholder_count > 0:
        print(f"   WARNING: {placeholder_count}/10000 sampled SMILES contain placeholder!")

    # Sample for grammar building and testing using index-based sampling
    # This avoids the O(n*m) set membership check
    print(f"\n2. Sampling {args.sample_size:,} molecules for grammar...")

    total_needed = args.sample_size + args.test_size
    if total_needed > len(all_smiles):
        print(f"   WARNING: Requested {total_needed:,} but only {len(all_smiles):,} available")
        total_needed = len(all_smiles)

    # Sample indices instead of checking set membership (MUCH faster)
    random.seed(args.seed)
    sampled_indices = random.sample(range(len(all_smiles)), total_needed)

    # Split into grammar and test samples
    grammar_indices = sampled_indices[:args.sample_size]
    test_indices = sampled_indices[args.sample_size:args.sample_size + args.test_size]

    grammar_sample = [all_smiles[i] for i in grammar_indices]
    test_sample = [all_smiles[i] for i in test_indices]

    print(f"   Grammar sample: {len(grammar_sample):,}")
    print(f"   Test sample: {len(test_sample):,}")

    # Build grammar
    print(f"\n3. Building grammar...")
    grammar, groups_used, grammar_stats = build_grammar_with_fixes(
        grammar_sample,
        max_groups=args.max_groups,
        use_frequency_capping=True,
        filter_placeholder_groups=args.filter_placeholder,
        verbose=True
    )

    print(f"\n   Grammar stats:")
    for k, v in grammar_stats.items():
        print(f"     {k}: {v:,}")

    # Test roundtrip on grammar sample (should be high)
    print(f"\n4. Testing roundtrip on GRAMMAR sample (sanity check)...")
    grammar_test_sample = sample_smiles(grammar_sample, min(10000, len(grammar_sample)), args.seed + 2)
    grammar_rt_stats, grammar_rt_failures = test_roundtrip(
        grammar_test_sample, grammar, verbose=True, max_failures=10
    )

    grammar_accuracy = 100 * grammar_rt_stats['success'] / grammar_rt_stats['total']
    print(f"\n   Grammar sample roundtrip: {grammar_rt_stats['success']:,}/{grammar_rt_stats['total']:,} = {grammar_accuracy:.2f}%")

    # Test roundtrip on held-out test sample
    print(f"\n5. Testing roundtrip on HELD-OUT test sample...")
    test_rt_stats, test_rt_failures = test_roundtrip(
        test_sample, grammar, verbose=True, max_failures=20
    )

    test_accuracy = 100 * test_rt_stats['success'] / test_rt_stats['total']
    print(f"\n   Test sample roundtrip: {test_rt_stats['success']:,}/{test_rt_stats['total']:,} = {test_accuracy:.2f}%")

    # Print failure breakdown
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nGrammar sample accuracy: {grammar_accuracy:.2f}%")
    print(f"Test sample accuracy:    {test_accuracy:.2f}%")

    print(f"\nTest sample failure breakdown:")
    for k, v in test_rt_stats.items():
        if k not in ['total', 'success']:
            pct = 100 * v / test_rt_stats['total']
            print(f"  {k}: {v:,} ({pct:.2f}%)")

    print(f"\nFirst {min(5, len(test_rt_failures))} failures:")
    for f in test_rt_failures[:5]:
        print(f"  [{f['error']}] {f['smiles'][:60]}...")
        if 'gsf' in f:
            print(f"    GSF: {f['gsf'][:80]}...")
        if 'detail' in f:
            print(f"    Detail: {f['detail'][:80]}")

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save stats
        stats_df = pd.DataFrame([{
            'grammar_sample_size': len(grammar_sample),
            'test_sample_size': len(test_sample),
            'max_groups': args.max_groups,
            'filter_placeholder': args.filter_placeholder,
            'grammar_accuracy': grammar_accuracy,
            'test_accuracy': test_accuracy,
            **{f'test_{k}': v for k, v in test_rt_stats.items()},
            **{f'grammar_stat_{k}': v for k, v in grammar_stats.items()},
        }])
        stats_df.to_csv(output_dir / 'quick_test_stats.csv', index=False)

        # Save failures
        failures_df = pd.DataFrame(test_rt_failures)
        failures_df.to_csv(output_dir / 'quick_test_failures.csv', index=False)

        print(f"\nResults saved to {output_dir}")

    # Success criteria
    print("\n" + "=" * 60)
    if test_accuracy >= 80:
        print("SUCCESS: Test accuracy >= 80%")
        print("Fixes are working! Proceed with full pipeline.")
    elif test_accuracy >= 50:
        print("PARTIAL: Test accuracy 50-80%")
        print("Fixes help but need further tuning (try increasing max_groups)")
    else:
        print("FAILED: Test accuracy < 50%")
        print("Fixes not sufficient. Need to investigate failure modes.")
    print("=" * 60)

    return test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick test for Group-SELFIES tokenizer fixes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-path', type=str,
        default='Data/Polymer/SMiPoly_polymers.gz',
        help='Path to polymer SMILES data'
    )
    parser.add_argument(
        '--sample-size', type=int, default=500000,
        help='Number of molecules to sample for grammar building'
    )
    parser.add_argument(
        '--test-size', type=int, default=50000,
        help='Number of held-out molecules for roundtrip testing'
    )
    parser.add_argument(
        '--max-groups', type=int, default=20000,
        help='Maximum number of groups in grammar'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--filter-placeholder', action='store_true',
        help='Filter out placeholder-containing groups (NOT recommended)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/quick_test',
        help='Directory to save results'
    )

    args = parser.parse_args()
    main(args)
