"""Group SELFIES Tokenizer with grammar-based, invertible tokenization."""

import re
import pickle
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
        """Replace placeholder atom back to '*'."""
        return smiles.replace(self.PLACEHOLDER_SMILES, "*")

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

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a p-SMILES string to Group SELFIES tokens.

        Args:
            smiles: Input p-SMILES string (with '*' for polymer connections).

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
            # Encode to Group SELFIES
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
            num_workers: Number of parallel workers for fragmentation (1 = sequential).
            chunk_size: Number of molecules per chunk for parallel processing.
            verbose: Whether to show progress bars.

        Returns:
            Tuple of (vocabulary dict, GroupGrammar).
        """
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

        # Fragment molecules (PARALLELIZED)
        if num_workers > 1:
            raw_groups = self._parallel_fragment_mols(
                valid_smiles,
                num_workers=num_workers,
                chunk_size=chunk_size,
                verbose=verbose
            )
        else:
            # Original sequential path
            mols_for_grammar = []
            for smiles in valid_smiles:
                smiles_ph = self._star_to_placeholder(smiles)
                mol = self._smiles_to_mol(smiles_ph)
                if mol is not None:
                    mols_for_grammar.append(mol)

            raw_groups = fragment_mols(mols_for_grammar)

        if not raw_groups:
            raise RuntimeError("fragment_mols returned no groups; cannot build GroupGrammar.")

        # Limit number of groups
        if len(raw_groups) > max_groups:
            raw_groups = raw_groups[:max_groups]

        # Create Group objects
        groups = [Group(name=f"G{i}", canonsmiles=g) for i, g in enumerate(raw_groups)]
        self.grammar = GroupGrammar(groups)

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
                    # Use imap_unordered for progress tracking
                    results = list(tqdm(
                        pool.imap_unordered(worker_func, chunks),
                        total=len(chunks),
                        desc="Fragmenting chunks"
                    ))
                else:
                    results = pool.map(worker_func, chunks)

                # Flatten results
                for groups in results:
                    all_groups.extend(groups)

        # Remove duplicates while preserving order (deterministic)
        unique_groups = []
        seen = set()
        for group in all_groups:
            if group not in seen:
                seen.add(group)
                unique_groups.append(group)

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

        data = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'grammar': self.grammar,
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
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(
            grammar=data['grammar'],
            vocab=data['vocab'],
            max_length=data['max_length']
        )
        tokenizer._placeholder_token = data.get('placeholder_token')
        tokenizer._placeholder_token_id = data.get('placeholder_token_id')

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
