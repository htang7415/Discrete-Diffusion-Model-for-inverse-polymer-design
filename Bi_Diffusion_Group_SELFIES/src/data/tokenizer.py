"""Group SELFIES Tokenizer with grammar-based, invertible tokenization."""

import re
import pickle
import contextlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from group_selfies import GroupGrammar, fragment_mols, Group
from group_selfies.utils import fragment_utils as fu

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Suppress noisy third-party stdout/stderr (e.g., attachment point warnings)."""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _select_diverse_set_simple(l, k, weights=None):
    """Simple non-recursive replacement for fragment_utils.select_diverse_set.

    Avoids RDKit Tanimoto distance computation and recursion issues.
    """
    if not l:
        return []
    if k >= len(l):
        return list(l)
    if weights is not None:
        items = list(zip(l, weights))
        items.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in items[:k]]
    return list(l)[:k]


def _unique_in_order(items):
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# Patch the fragment_utils to use simple selection
fu.select_diverse_set = _select_diverse_set_simple


def _fragment_batch_worker(smiles_batch, placeholder_smiles):
    """Worker function to fragment a batch of SMILES in parallel.

    This function runs in a separate process and cannot access class methods directly.

    Args:
        smiles_batch: List of p-SMILES strings to process
        placeholder_smiles: Placeholder string to replace '*'

    Returns:
        List of fragmented groups (canonical SMILES strings)
    """
    from rdkit import Chem, RDLogger
    from group_selfies import fragment_mols

    # Silence RDKit warnings in worker
    RDLogger.DisableLog('rdApp.*')

    # Convert SMILES to Mol objects
    mols = []
    for smiles in smiles_batch:
        smiles_ph = smiles.replace("*", placeholder_smiles)
        try:
            mol = Chem.MolFromSmiles(smiles_ph)
            if mol is not None:
                mols.append(mol)
        except Exception:
            continue

    # Fragment this batch
    if not mols:
        return []

    try:
        with _suppress_stdout_stderr():
            groups = fragment_mols(mols)
        return groups if groups else []
    except Exception as e:
        # Log error but don't crash worker
        print(f"Warning: fragment_mols failed for batch: {e}")
        return []


def _tokenize_batch_worker(smiles_batch, grammar, placeholder_smiles):
    """Worker function to tokenize a batch of SMILES in parallel.

    This function runs in a separate process and tokenizes molecules to collect vocabulary.

    Args:
        smiles_batch: List of p-SMILES strings to tokenize
        grammar: GroupGrammar object for encoding
        placeholder_smiles: Placeholder string to replace '*'

    Returns:
        Set of unique tokens from this batch
    """
    import re
    from rdkit import Chem, RDLogger

    # Silence RDKit warnings in worker
    RDLogger.DisableLog('rdApp.*')

    batch_tokens = set()

    for smiles in smiles_batch:
        # Replace * with placeholder
        smiles_ph = smiles.replace("*", placeholder_smiles)

        # Convert to RDKit Mol
        try:
            mol = Chem.MolFromSmiles(smiles_ph)
            if mol is None:
                continue
        except Exception:
            continue

        # Encode to Group SELFIES
        try:
            with _suppress_stdout_stderr():
                gsf_string = grammar.full_encoder(mol)

            # Parse Group SELFIES string into tokens
            # Group SELFIES tokens are bracket-enclosed: [token1][token2]...
            i = 0
            while i < len(gsf_string):
                # Check for bracket token
                if gsf_string[i] == '[':
                    match = re.match(r'\[[^\[\]]+\]', gsf_string[i:])
                    if match:
                        batch_tokens.add(match.group())
                        i += len(match.group())
                        continue

                # Check for group reference (e.g., :0G10)
                if gsf_string[i] == ':':
                    match = re.match(r':[0-9]+[A-Za-z0-9]+', gsf_string[i:])
                    if match:
                        batch_tokens.add(match.group())
                        i += len(match.group())
                        continue

                # Single character
                batch_tokens.add(gsf_string[i])
                i += 1

        except Exception:
            continue

    return batch_tokens


def _get_lengths_worker(smiles_batch, grammar, placeholder_smiles):
    """Worker function to get token lengths for a batch of SMILES in parallel.

    Args:
        smiles_batch: List of p-SMILES strings to tokenize
        grammar: GroupGrammar object for encoding
        placeholder_smiles: Placeholder string to replace '*'

    Returns:
        List of token lengths (one per input SMILES)
    """
    import re
    from rdkit import Chem, RDLogger

    # Silence RDKit warnings in worker
    RDLogger.DisableLog('rdApp.*')

    lengths = []
    for smiles in smiles_batch:
        smiles_ph = smiles.replace("*", placeholder_smiles)
        try:
            mol = Chem.MolFromSmiles(smiles_ph)
            if mol is None:
                lengths.append(0)
                continue

            with _suppress_stdout_stderr():
                gsf_string = grammar.full_encoder(mol)

            # Parse and count tokens
            token_count = 0
            i = 0
            while i < len(gsf_string):
                if gsf_string[i] == '[':
                    match = re.match(r'\[[^\[\]]+\]', gsf_string[i:])
                    if match:
                        token_count += 1
                        i += len(match.group())
                        continue
                if gsf_string[i] == ':':
                    match = re.match(r':[0-9]+[A-Za-z0-9]+', gsf_string[i:])
                    if match:
                        token_count += 1
                        i += len(match.group())
                        continue
                token_count += 1
                i += 1

            lengths.append(token_count)
        except Exception:
            lengths.append(0)

    return lengths


def _verify_roundtrip_worker(smiles_batch, group_smiles, placeholder_smiles, max_failures=5):
    """Worker function to verify roundtrip for a batch of SMILES in parallel.

    Args:
        smiles_batch: List of p-SMILES strings to verify
        group_smiles: List of canonical SMILES for grammar groups (serializable)
        placeholder_smiles: Placeholder string to replace '*'
        max_failures: Maximum number of failures to collect for diagnostics

    Returns:
        Tuple of (valid_count, total_count, failures)
    """
    import re
    from rdkit import Chem, RDLogger
    from group_selfies import GroupGrammar, Group

    # Silence RDKit warnings in worker
    RDLogger.DisableLog('rdApp.*')

    # Recreate grammar in worker (avoids pickling issues with GroupGrammar)
    groups = [Group(name=f"G{i}", canonsmiles=g) for i, g in enumerate(group_smiles)]
    grammar = GroupGrammar(groups)

    valid_count = 0
    total_count = len(smiles_batch)
    failures = []  # Collect first N failures for diagnostics

    for smiles in smiles_batch:
        try:
            # Step 1: Tokenize (encode)
            smiles_ph = smiles.replace("*", placeholder_smiles)
            mol_orig = Chem.MolFromSmiles(smiles_ph)
            if mol_orig is None:
                continue

            # Copy mol and get canonical BEFORE encoder mutates it
            canon_orig = Chem.MolToSmiles(Chem.Mol(mol_orig))

            # Encode to Group SELFIES
            with _suppress_stdout_stderr():
                gsf_string = grammar.full_encoder(mol_orig)

            # Parse into tokens (simplified from _parse_gsf_string)
            tokens = []
            i = 0
            while i < len(gsf_string):
                if gsf_string[i] == '[':
                    match = re.match(r'\[[^\[\]]+\]', gsf_string[i:])
                    if match:
                        tokens.append(match.group())
                        i += len(match.group())
                        continue
                if gsf_string[i] == ':':
                    match = re.match(r':[0-9]+[A-Za-z0-9]+', gsf_string[i:])
                    if match:
                        tokens.append(match.group())
                        i += len(match.group())
                        continue
                tokens.append(gsf_string[i])
                i += 1

            # Step 2: Detokenize (decode)
            decoded_gsf_string = ''.join(tokens)
            with _suppress_stdout_stderr():
                mol_decoded = grammar.decoder(decoded_gsf_string)
            if mol_decoded is None:
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': smiles,
                        'gsf_string': gsf_string,
                        'error': 'decode_returned_none'
                    })
                continue

            smiles_decoded_ph = Chem.MolToSmiles(mol_decoded)
            if smiles_decoded_ph is None:
                continue

            smiles_decoded = smiles_decoded_ph.replace(placeholder_smiles, "*")

            # Step 3: Compare canonical forms
            mol_dec = Chem.MolFromSmiles(smiles_decoded.replace("*", placeholder_smiles))
            if mol_dec is None:
                continue
            canon_dec = Chem.MolToSmiles(mol_dec)

            if canon_orig == canon_dec:
                valid_count += 1
            else:
                if len(failures) < max_failures:
                    failures.append({
                        'smiles': smiles,
                        'gsf_string': gsf_string,
                        'decoded': smiles_decoded,
                        'canon_orig': canon_orig,
                        'canon_dec': canon_dec,
                        'error': 'canonical_mismatch'
                    })

        except Exception as e:
            if len(failures) < max_failures:
                failures.append({
                    'smiles': smiles,
                    'error': f'exception: {str(e)}'
                })
            continue

    return (valid_count, total_count, failures)


class GroupSELFIESTokenizer:
    """Grammar-based tokenizer for Group SELFIES representation.

    Converts p-SMILES to Group SELFIES tokens using a data-dependent grammar.
    The grammar is built from training molecules and must be saved/loaded with
    the tokenizer.

    Tokenization flow:
        p-SMILES (with *) -> p-SMILES (with [I+3]) -> RDKit Mol -> Group SELFIES tokens

    Detokenization flow:
        Group SELFIES tokens -> RDKit Mol -> p-SMILES (with [I+3]) -> p-SMILES (with *)
    """

    # Special tokens (same as p-SMILES tokenizer for compatibility)
    SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[BOS]', '[EOS]', '[UNK]']

    # Placeholder for '*' (polymer connection point)
    # Using [I+3] as it's unlikely to appear in real molecules
    PLACEHOLDER_SMILES = "[I+3]"

    def __init__(
        self,
        grammar: Optional[GroupGrammar] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128
    ):
        """Initialize tokenizer.

        Args:
            grammar: Pre-built GroupGrammar for encoding/decoding.
            vocab: Pre-built vocabulary (token -> id mapping).
            max_length: Maximum sequence length.
        """
        self.grammar = grammar
        self.max_length = max_length
        self.vocab = vocab if vocab else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()} if vocab else {}

        # Cache placeholder token info
        self._placeholder_token = None
        self._placeholder_token_id = None

    def _star_to_placeholder(self, smiles: str) -> str:
        """Replace '*' with placeholder SMILES atom."""
        return smiles.replace("*", self.PLACEHOLDER_SMILES)

    def _placeholder_to_star(self, smiles: str) -> str:
        """Replace placeholder atom back to '*'.

        Handles multiple iodine placeholder variants that RDKit may produce:
        - [I+3] - basic placeholder
        - [IH+3] - with explicit hydrogen
        - [IH0+3] - with explicit H count
        """
        # Use regex to handle all variants: [I+3], [IH+3], [IH0+3], etc.
        return re.sub(r'\[IH?\d*\+3\]', '*', smiles)

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit Mol object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception:
            return None

    def _mol_to_smiles(self, mol: Chem.Mol, canonical: bool = True) -> Optional[str]:
        """Convert RDKit Mol to SMILES string."""
        try:
            return Chem.MolToSmiles(mol, canonical=canonical)
        except Exception:
            return None

    def tokenize(self, smiles: str, canonicalize: bool = True) -> List[str]:
        """Tokenize a p-SMILES string to Group SELFIES tokens.

        Args:
            smiles: Input p-SMILES string (with '*' for polymer connections).
            canonicalize: If True, canonicalize the molecule before encoding.
                This improves roundtrip consistency by ensuring the same
                canonical representation regardless of input SMILES order.

        Returns:
            List of Group SELFIES tokens.
        """
        if self.grammar is None:
            raise ValueError("Grammar not initialized. Call build_vocab_and_grammar first.")

        # Replace * with placeholder
        smiles_ph = self._star_to_placeholder(smiles)

        # Convert to RDKit Mol
        mol = self._smiles_to_mol(smiles_ph)
        if mol is None:
            # Return UNK token for invalid SMILES
            return ['[UNK]']

        try:
            # Pre-canonicalize for consistent encoding
            # This helps improve roundtrip accuracy by ensuring the grammar
            # always sees molecules in a consistent canonical form
            if canonicalize:
                canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                mol = Chem.MolFromSmiles(canon_smiles)
                if mol is None:
                    return ['[UNK]']

            # Encode to Group SELFIES
            with _suppress_stdout_stderr():
                gsf_string = self.grammar.full_encoder(mol)

            # Parse Group SELFIES string into tokens
            # Group SELFIES format: [token1][token2][token3]...
            tokens = self._parse_gsf_string(gsf_string)
            return tokens
        except Exception:
            return ['[UNK]']

    def _parse_gsf_string(self, gsf_string: str) -> List[str]:
        """Parse a Group SELFIES string into individual tokens.

        Group SELFIES tokens are bracket-enclosed: [token1][token2]...
        """
        if not gsf_string:
            return []

        tokens = []
        # Match all bracket-enclosed tokens
        pattern = re.compile(r'\[[^\[\]]+\]|:[0-9]+[A-Za-z0-9]+')

        i = 0
        while i < len(gsf_string):
            # Check for bracket token
            if gsf_string[i] == '[':
                match = re.match(r'\[[^\[\]]+\]', gsf_string[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
                    continue

            # Check for group reference (e.g., :0G10)
            if gsf_string[i] == ':':
                match = re.match(r':[0-9]+[A-Za-z0-9]+', gsf_string[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
                    continue

            # Single character (should be rare in Group SELFIES)
            tokens.append(gsf_string[i])
            i += 1

        return tokens

    def _tokens_to_gsf_string(self, tokens: List[str]) -> str:
        """Convert tokens back to Group SELFIES string."""
        # Filter out special tokens
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return ''.join(filtered)

    def detokenize(self, tokens: List[str]) -> str:
        """Convert Group SELFIES tokens back to p-SMILES string.

        Args:
            tokens: List of Group SELFIES tokens.

        Returns:
            Reconstructed p-SMILES string (with '*').
        """
        if self.grammar is None:
            raise ValueError("Grammar not initialized. Call build_vocab_and_grammar first.")

        # Filter out special tokens
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        if not filtered:
            return ""

        # Reconstruct Group SELFIES string
        gsf_string = ''.join(filtered)

        try:
            # Decode to RDKit Mol
            with _suppress_stdout_stderr():
                mol = self.grammar.decoder(gsf_string)
            if mol is None:
                return ""

            # Convert to SMILES (with placeholder)
            smiles_ph = self._mol_to_smiles(mol)
            if smiles_ph is None:
                return ""

            # Replace placeholder back to *
            return self._placeholder_to_star(smiles_ph)
        except Exception:
            return ""

    def build_vocab_and_grammar(
        self,
        smiles_list: List[str],
        max_groups: int = 10000,
        num_workers: int = 1,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> Tuple[Dict[str, int], GroupGrammar]:
        """Build vocabulary and grammar from a list of SMILES strings.

        Args:
            smiles_list: List of p-SMILES strings.
            max_groups: Maximum number of groups in grammar.
            num_workers: Number of parallel workers for vocabulary building (1 = sequential).
                         NOTE: Fragmentation is ALWAYS sequential (parallel breaks grammar coherence).
            chunk_size: Number of molecules per chunk for parallel processing.
            verbose: Whether to show progress bars.

        Returns:
            Tuple of (vocabulary dict, GroupGrammar).
        """
        import random

        # Filter to valid molecules first
        valid_smiles = []
        iterator = tqdm(smiles_list, desc="Validating molecules") if verbose else smiles_list
        for smiles in iterator:
            smiles_ph = self._star_to_placeholder(smiles)
            mol = self._smiles_to_mol(smiles_ph)
            if mol is not None:
                valid_smiles.append(smiles)

        if not valid_smiles:
            raise ValueError("No valid molecules found for grammar building.")

        print(f"Building grammar from {len(valid_smiles)} valid molecules...")

        # Fragment molecules - MUST be sequential on all molecules at once
        # CRITICAL: Parallel fragmentation breaks grammar coherence because each chunk
        # gets different group assignments. The fragment_mols() function builds GROUP
        # RELATIONSHIPS internally across all molecules. When called on chunks separately,
        # groups from chunk A may not exist in chunk B, resulting in an incomplete grammar.
        # Quick test proved: sequential = 99.39% accuracy, parallel = 0.01% accuracy.

        # Convert all valid SMILES to RDKit mols
        mols_for_grammar = []
        iterator = tqdm(valid_smiles, desc="Converting to RDKit mols") if verbose else valid_smiles
        for smiles in iterator:
            smiles_ph = self._star_to_placeholder(smiles)
            mol = self._smiles_to_mol(smiles_ph)
            if mol is not None:
                mols_for_grammar.append(mol)

        if verbose:
            print(f"Fragmenting {len(mols_for_grammar)} molecules (sequential, required for grammar coherence)...")

        with _suppress_stdout_stderr():
            raw_groups = fragment_mols(mols_for_grammar)

        if not raw_groups:
            raise RuntimeError("fragment_mols returned no groups; cannot build GroupGrammar.")

        raw_groups = _unique_in_order(raw_groups)

        # Count placeholder-containing groups (for logging only)
        placeholder_groups = [g for g in raw_groups if self.PLACEHOLDER_SMILES in g]
        if verbose:
            print(f"Total unique groups: {len(raw_groups)}")
            print(f"Groups containing placeholder: {len(placeholder_groups)} (keeping them)")

        # NOTE: We do NOT filter out placeholder-containing groups anymore.
        # The quick test showed 99.4% accuracy when keeping them (vs 0.05% when filtering).
        # The notebook also keeps them and achieves 100% roundtrip.

        # Limit number of groups using frequency-based selection (most common first)
        if max_groups and len(raw_groups) > max_groups:
            from collections import Counter
            # Count frequency of each group in the original (non-unique) list
            # Since raw_groups is already unique, we use the order as a proxy
            # In production with parallel fragmentation, consider tracking counts
            if verbose:
                print(f"Capping groups from {len(raw_groups)} to {max_groups}")
            raw_groups = raw_groups[:max_groups]

        # Create Group objects
        groups = [Group(name=f"G{i}", canonsmiles=g) for i, g in enumerate(raw_groups)]
        self.grammar = GroupGrammar(groups)
        self.group_smiles = raw_groups  # Store for parallel verification (serializable)

        print(f"Grammar built with {len(groups)} groups.")

        # Build vocabulary from tokenized training data
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        current_id = len(self.SPECIAL_TOKENS)

        # Collect all unique tokens (PARALLELIZED)
        if num_workers > 1:
            all_tokens = self._parallel_build_vocabulary(
                valid_smiles,
                self.grammar,
                num_workers=num_workers,
                chunk_size=chunk_size,
                verbose=verbose
            )
        else:
            # Sequential fallback
            all_tokens = set()
            iterator = tqdm(valid_smiles, desc="Building vocabulary") if verbose else valid_smiles
            for smiles in iterator:
                tokens = self.tokenize(smiles)
                all_tokens.update(tokens)

        # Remove special tokens that might have been added
        all_tokens -= set(self.SPECIAL_TOKENS)

        # Sort tokens for deterministic ordering
        sorted_tokens = sorted(all_tokens)

        # Add to vocabulary
        for token in sorted_tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

        print(f"Vocabulary built with {len(vocab)} tokens.")

        # Find placeholder token ID
        self._find_placeholder_token()

        return vocab, self.grammar

    def _parallel_fragment_mols(
        self,
        smiles_list: List[str],
        num_workers: int,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> List[str]:
        """Fragment molecules in parallel using multiprocessing.

        Args:
            smiles_list: List of p-SMILES strings
            num_workers: Number of parallel workers
            chunk_size: Number of SMILES per chunk
            verbose: Show progress bars

        Returns:
            List of unique fragment SMILES
        """
        # Split into chunks
        chunks = [
            smiles_list[i:i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]

        if verbose:
            print(f"Fragmenting {len(smiles_list)} molecules using {num_workers} workers...")
            print(f"Split into {len(chunks)} chunks of ~{chunk_size} molecules each")

        # Create worker function with placeholder bound
        worker_func = partial(_fragment_batch_worker, placeholder_smiles=self.PLACEHOLDER_SMILES)

        # Process chunks in parallel
        all_groups = []

        if num_workers <= 1:
            # Sequential fallback for debugging
            iterator = tqdm(chunks, desc="Fragmenting chunks") if verbose else chunks
            for chunk in iterator:
                groups = worker_func(chunk)
                all_groups.extend(groups)
        else:
            # Parallel processing
            with Pool(processes=num_workers) as pool:
                if verbose:
                    # Use imap for ordered, deterministic progress tracking
                    results = list(tqdm(
                        pool.imap(worker_func, chunks),
                        total=len(chunks),
                        desc="Fragmenting chunks"
                    ))
                else:
                    results = pool.map(worker_func, chunks)

                # Flatten results
                for groups in results:
                    all_groups.extend(groups)

        unique_groups = _unique_in_order(all_groups)

        if verbose:
            print(f"Found {len(all_groups)} total fragments ({len(unique_groups)} unique)")

        return unique_groups

    def _parallel_build_vocabulary(
        self,
        smiles_list: List[str],
        grammar,
        num_workers: int,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> set:
        """Build vocabulary in parallel by tokenizing molecules.

        Args:
            smiles_list: List of p-SMILES strings
            grammar: GroupGrammar object for encoding
            num_workers: Number of parallel workers
            chunk_size: Number of SMILES per chunk
            verbose: Show progress bars

        Returns:
            Set of unique tokens
        """
        # Split into chunks
        chunks = [
            smiles_list[i:i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]

        if verbose:
            print(f"Building vocabulary from {len(smiles_list)} molecules using {num_workers} workers...")
            print(f"Split into {len(chunks)} chunks of ~{chunk_size} molecules each")

        # Create worker function with grammar and placeholder bound
        worker_func = partial(_tokenize_batch_worker, grammar=grammar, placeholder_smiles=self.PLACEHOLDER_SMILES)

        # Process chunks in parallel
        all_tokens = set()

        if num_workers <= 1:
            # Sequential fallback for debugging
            iterator = tqdm(chunks, desc="Building vocabulary") if verbose else chunks
            for chunk in iterator:
                batch_tokens = worker_func(chunk)
                all_tokens.update(batch_tokens)
        else:
            # Parallel processing
            with Pool(processes=num_workers) as pool:
                if verbose:
                    # Use imap_unordered for progress tracking
                    results = list(tqdm(
                        pool.imap_unordered(worker_func, chunks),
                        total=len(chunks),
                        desc="Building vocabulary"
                    ))
                else:
                    results = pool.map(worker_func, chunks)

                # Merge all token sets
                for batch_tokens in results:
                    all_tokens.update(batch_tokens)

        if verbose:
            print(f"Found {len(all_tokens)} unique tokens")

        return all_tokens

    def parallel_verify_roundtrip(
        self,
        smiles_list: List[str],
        num_workers: int,
        chunk_size: int = 1000,
        verbose: bool = True,
        max_failures: int = 20
    ) -> Tuple[int, int, List[dict]]:
        """Verify roundtrip invertibility in parallel.

        Args:
            smiles_list: List of p-SMILES strings to verify
            num_workers: Number of parallel workers
            chunk_size: Number of SMILES per chunk
            verbose: Show progress bars
            max_failures: Maximum number of failures to collect for diagnostics

        Returns:
            Tuple of (valid_count, total_count, failures)
        """
        if self.grammar is None:
            raise ValueError("Grammar not initialized. Call build_vocab_and_grammar first.")

        # Split into chunks
        chunks = [
            smiles_list[i:i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]

        if verbose:
            print(f"Verifying {len(smiles_list)} molecules using {num_workers} workers...")
            print(f"Split into {len(chunks)} chunks of ~{chunk_size} molecules each")

        # Create worker function with group_smiles (serializable) and placeholder bound
        # Note: We pass group_smiles instead of grammar to avoid pickling issues
        worker_func = partial(_verify_roundtrip_worker, group_smiles=self.group_smiles, placeholder_smiles=self.PLACEHOLDER_SMILES)

        # Process chunks in parallel
        total_valid = 0
        total_count = 0
        all_failures = []

        if num_workers <= 1:
            # Sequential fallback
            iterator = tqdm(chunks, desc="Verifying roundtrip") if verbose else chunks
            for chunk in iterator:
                valid, count, failures = worker_func(chunk)
                total_valid += valid
                total_count += count
                if len(all_failures) < max_failures:
                    all_failures.extend(failures[:max_failures - len(all_failures)])
        else:
            # Parallel processing
            with Pool(processes=num_workers) as pool:
                if verbose:
                    results = list(tqdm(
                        pool.imap_unordered(worker_func, chunks),
                        total=len(chunks),
                        desc="Verifying roundtrip"
                    ))
                else:
                    results = pool.map(worker_func, chunks)

                # Aggregate results
                for valid, count, failures in results:
                    total_valid += valid
                    total_count += count
                    if len(all_failures) < max_failures:
                        all_failures.extend(failures[:max_failures - len(all_failures)])

        if verbose:
            accuracy = 100 * total_valid / total_count if total_count > 0 else 0
            print(f"Roundtrip accuracy: {total_valid}/{total_count} ({accuracy:.2f}%)")

            if all_failures:
                print(f"\nFirst {len(all_failures)} failures:")
                for f in all_failures[:5]:
                    error_type = f.get('error', 'unknown')
                    smiles = f.get('smiles', 'N/A')[:50]
                    print(f"  {error_type}: {smiles}")

        return (total_valid, total_count, all_failures)

    def parallel_get_lengths(
        self,
        smiles_list: List[str],
        num_workers: int,
        chunk_size: int = 1000,
        verbose: bool = True
    ) -> List[int]:
        """Get token lengths for a list of SMILES in parallel.

        Args:
            smiles_list: List of p-SMILES strings
            num_workers: Number of parallel workers (1 = sequential)
            chunk_size: Number of SMILES per chunk
            verbose: Show progress bars

        Returns:
            List of token lengths (one per input SMILES)
        """
        if self.grammar is None:
            raise ValueError("Grammar not initialized. Call build_vocab_and_grammar first.")

        # Split into chunks
        chunks = [
            smiles_list[i:i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]

        if verbose:
            print(f"Computing lengths for {len(smiles_list)} molecules using {num_workers} workers...")

        # Create worker function with grammar and placeholder bound
        worker_func = partial(_get_lengths_worker, grammar=self.grammar, placeholder_smiles=self.PLACEHOLDER_SMILES)

        # Process chunks
        all_lengths = []

        if num_workers <= 1:
            # Sequential fallback
            iterator = tqdm(chunks, desc="Computing lengths") if verbose else chunks
            for chunk in iterator:
                lengths = worker_func(chunk)
                all_lengths.extend(lengths)
        else:
            # Parallel processing
            with Pool(processes=num_workers) as pool:
                if verbose:
                    results = list(tqdm(
                        pool.imap(worker_func, chunks),
                        total=len(chunks),
                        desc="Computing lengths"
                    ))
                else:
                    results = pool.map(worker_func, chunks)

                # Flatten results
                for lengths in results:
                    all_lengths.extend(lengths)

        return all_lengths

    def _find_placeholder_token(self):
        """Find the token(s) representing the placeholder atom."""
        # Tokenize a simple molecule with placeholder
        test_smiles = "*C*"
        tokens = self.tokenize(test_smiles)

        # Find tokens that represent the placeholder
        # The placeholder [I+3] typically becomes [IH0+3] or similar in Group SELFIES
        for token in tokens:
            if 'I' in token and '+3' in token:
                self._placeholder_token = token
                self._placeholder_token_id = self.vocab.get(token)
                break

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, List[int]]:
        """Encode a p-SMILES string to token IDs.

        Args:
            smiles: Input p-SMILES string.
            add_special_tokens: Whether to add BOS/EOS tokens.
            padding: Whether to pad to max_length.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        tokens = self.tokenize(smiles)

        # Convert to IDs
        unk_id = self.vocab.get('[UNK]', 0)
        ids = [self.vocab.get(token, unk_id) for token in tokens]

        # Add special tokens
        if add_special_tokens:
            bos_id = self.vocab['[BOS]']
            eos_id = self.vocab['[EOS]']
            ids = [bos_id] + ids + [eos_id]

        # Truncate if needed
        if len(ids) > self.max_length:
            ids = ids[:self.max_length - 1] + [self.vocab['[EOS]']]

        # Create attention mask before padding
        attention_mask = [1] * len(ids)

        # Padding
        if padding:
            pad_id = self.vocab['[PAD]']
            pad_length = self.max_length - len(ids)
            if pad_length > 0:
                ids = ids + [pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

        result = {'input_ids': ids}
        if return_attention_mask:
            result['attention_mask'] = attention_mask

        return result

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to p-SMILES string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded p-SMILES string.
        """
        tokens = []
        for id_ in ids:
            token = self.id_to_token.get(id_, '[UNK]')
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)

        return self.detokenize(tokens)

    def batch_encode(
        self,
        smiles_list: List[str],
        add_special_tokens: bool = True,
        padding: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            add_special_tokens: Whether to add BOS/EOS tokens.
            padding: Whether to pad sequences.

        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'.
        """
        results = [
            self.encode(smiles, add_special_tokens, padding)
            for smiles in smiles_list
        ]

        return {
            'input_ids': [r['input_ids'] for r in results],
            'attention_mask': [r['attention_mask'] for r in results]
        }

    def batch_decode(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token IDs.

        Args:
            ids_list: List of token ID lists.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded SMILES strings.
        """
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def verify_roundtrip(self, smiles: str) -> bool:
        """Verify that tokenization is invertible for a given string.

        Uses canonical SMILES comparison to check molecular identity.

        Args:
            smiles: Input p-SMILES string.

        Returns:
            True if the decoded molecule matches the original.
        """
        try:
            # Tokenize and detokenize
            tokens = self.tokenize(smiles)
            decoded = self.detokenize(tokens)

            if not decoded:
                return False

            # Compare canonical forms
            smiles_ph_orig = self._star_to_placeholder(smiles)
            smiles_ph_dec = self._star_to_placeholder(decoded)

            mol_orig = self._smiles_to_mol(smiles_ph_orig)
            mol_dec = self._smiles_to_mol(smiles_ph_dec)

            if mol_orig is None or mol_dec is None:
                return False

            canon_orig = self._mol_to_smiles(mol_orig)
            canon_dec = self._mol_to_smiles(mol_dec)

            return canon_orig == canon_dec
        except Exception:
            return False

    def save(self, path: str) -> None:
        """Save tokenizer to file (pickle format for grammar).

        Args:
            path: Path to save the tokenizer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract group SMILES from grammar for reliable reconstruction
        # The grammar object itself can have pickling issues, so we store
        # the group SMILES and recreate the grammar on load
        group_smiles = None
        if self.grammar is not None:
            group_smiles = [g.canonsmiles for _, g in self.grammar.vocab.items()]

        data = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'grammar': self.grammar,  # Keep for backwards compatibility
            'group_smiles': group_smiles,  # New: for reliable grammar reconstruction
            'placeholder_token': self._placeholder_token,
            'placeholder_token_id': self._placeholder_token_id
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'GroupSELFIESTokenizer':
        """Load tokenizer from file.

        Args:
            path: Path to the tokenizer file.

        Returns:
            Loaded tokenizer instance.
        """
        from group_selfies import GroupGrammar, Group

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Prefer reconstructing grammar from group_smiles (more reliable)
        # The pickled grammar object can have internal state issues
        grammar = None
        group_smiles = data.get('group_smiles')
        if group_smiles is not None:
            groups = [Group(name=f"G{i}", canonsmiles=g) for i, g in enumerate(group_smiles)]
            grammar = GroupGrammar(groups)
        else:
            # Fallback to pickled grammar for backwards compatibility
            grammar = data.get('grammar')

        tokenizer = cls(
            grammar=grammar,
            vocab=data['vocab'],
            max_length=data['max_length']
        )
        tokenizer._placeholder_token = data.get('placeholder_token')
        tokenizer._placeholder_token_id = data.get('placeholder_token_id')
        tokenizer.group_smiles = group_smiles  # Store for parallel verification

        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Return PAD token ID."""
        return self.vocab['[PAD]']

    @property
    def mask_token_id(self) -> int:
        """Return MASK token ID."""
        return self.vocab['[MASK]']

    @property
    def bos_token_id(self) -> int:
        """Return BOS token ID."""
        return self.vocab['[BOS]']

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self.vocab['[EOS]']

    @property
    def unk_token_id(self) -> int:
        """Return UNK token ID."""
        return self.vocab['[UNK]']

    def get_placeholder_token_id(self) -> Optional[int]:
        """Return the token ID for the placeholder (represents '*')."""
        return self._placeholder_token_id

    def get_placeholder_token(self) -> Optional[str]:
        """Return the placeholder token string."""
        return self._placeholder_token

    def get_star_token_id(self) -> int:
        """Return the token ID for '*' (placeholder token).

        For backward compatibility with sampler.
        """
        if self._placeholder_token_id is not None:
            return self._placeholder_token_id
        return self.unk_token_id
