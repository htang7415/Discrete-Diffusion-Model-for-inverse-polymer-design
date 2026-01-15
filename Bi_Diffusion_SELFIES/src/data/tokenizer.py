"""p-SMILES Tokenizer with deterministic, invertible tokenization."""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class PSmilesTokenizer:
    """Deterministic, invertible tokenizer for p-SMILES strings.

    Tokenization rules (priority order):
    1. Bracket tokens: [...] blocks -> one token
    2. Ring indices with %: %10, %11, etc. -> one token
    3. Multi-character atoms: Cl, Br, Si, etc. -> one token
    4. Single-character tokens: atoms, digits, symbols
    """

    # Multi-character atoms (must be matched before single chars)
    MULTI_CHAR_ATOMS = [
        'Cl', 'Br', 'Si', 'Na', 'Li', 'Ca', 'Mg', 'Al', 'Sn', 'Sb', 'Se',
        'Fe', 'Cu', 'Zn', 'Ni', 'Co', 'Mn', 'Cr', 'Ti', 'Pt', 'Pd', 'Au',
        'Ag', 'Hg', 'Pb', 'Bi', 'As', 'Te', 'Ge', 'Ga', 'In', 'Tl'
    ]

    # Single-character atoms
    SINGLE_ATOMS = list('BCNOFPSIHcnosp')

    # Symbols
    SYMBOLS = list('=-#/\\().@+')

    # Digits
    DIGITS = list('0123456789')

    # Special tokens
    SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[BOS]', '[EOS]', '[UNK]']

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128
    ):
        """Initialize tokenizer.

        Args:
            vocab: Pre-built vocabulary (token -> id mapping).
            max_length: Maximum sequence length.
        """
        self.max_length = max_length
        self.vocab = vocab if vocab else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()} if vocab else {}

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for tokenization."""
        # Pattern for bracket tokens: [...]
        self.bracket_pattern = re.compile(r'\[[^\[\]]+\]')
        # Pattern for ring indices with %
        self.ring_pattern = re.compile(r'%\d{2}')
        # Pattern for multi-character atoms
        self.multi_atom_pattern = re.compile(
            '|'.join(sorted(self.MULTI_CHAR_ATOMS, key=len, reverse=True))
        )

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a p-SMILES string.

        Args:
            smiles: Input p-SMILES string.

        Returns:
            List of tokens.
        """
        tokens = []
        i = 0
        n = len(smiles)

        while i < n:
            # Try bracket token first
            if smiles[i] == '[':
                match = self.bracket_pattern.match(smiles, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            # Try ring index with %
            if smiles[i] == '%':
                match = self.ring_pattern.match(smiles, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            # Try multi-character atoms
            match = self.multi_atom_pattern.match(smiles, i)
            if match:
                tokens.append(match.group())
                i = match.end()
                continue

            # Single character token
            tokens.append(smiles[i])
            i += 1

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to p-SMILES string.

        Args:
            tokens: List of tokens.

        Returns:
            Reconstructed p-SMILES string.
        """
        # Filter out special tokens
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return ''.join(filtered)

    def build_vocab(self, smiles_list: List[str]) -> Dict[str, int]:
        """Build vocabulary from a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Vocabulary dictionary (token -> id).
        """
        # Start with special tokens
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        current_id = len(self.SPECIAL_TOKENS)

        # Collect all unique tokens
        all_tokens = set()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            all_tokens.update(tokens)

        # Sort tokens for deterministic ordering
        sorted_tokens = sorted(all_tokens)

        # Add to vocabulary
        for token in sorted_tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

        return vocab

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

        Args:
            smiles: Input SMILES string.

        Returns:
            True if detokenize(tokenize(smiles)) == smiles.
        """
        tokens = self.tokenize(smiles)
        reconstructed = self.detokenize(tokens)
        return reconstructed == smiles

    def save(self, path: str) -> None:
        """Save tokenizer to file.

        Args:
            path: Path to save the tokenizer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'vocab': self.vocab,
            'max_length': self.max_length
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PSmilesTokenizer':
        """Load tokenizer from file.

        Args:
            path: Path to the tokenizer file.

        Returns:
            Loaded tokenizer instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(vocab=data['vocab'], max_length=data['max_length'])

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

    def get_star_token_id(self) -> int:
        """Return the token ID for '*'."""
        return self.vocab.get('*', self.unk_token_id)
