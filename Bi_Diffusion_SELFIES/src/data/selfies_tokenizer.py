"""SELFIES Tokenizer with deterministic, invertible tokenization."""

import json
import selfies as sf
from pathlib import Path
from typing import List, Dict, Optional


class SelfiesTokenizer:
    """Deterministic, invertible tokenizer for SELFIES strings.

    SELFIES tokens are naturally bracket-delimited (e.g., [C], [Branch1], [=N], [I+3]),
    making tokenization much simpler than p-SMILES. We use selfies.split_selfies()
    for tokenization.

    Special placeholder: [I+3] represents polymer attachment points (replaces '*' in p-SMILES).
    """

    # Special tokens (same as PSmilesTokenizer for compatibility)
    SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[BOS]', '[EOS]', '[UNK]']

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 256
    ):
        """Initialize tokenizer.

        Args:
            vocab: Pre-built vocabulary (token -> id mapping).
            max_length: Maximum sequence length (default 256 for SELFIES, vs 128 for p-SMILES).
        """
        self.max_length = max_length
        self.vocab = vocab if vocab else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()} if vocab else {}

    def tokenize(self, selfies: str) -> List[str]:
        """Tokenize a SELFIES string.

        Args:
            selfies: Input SELFIES string.

        Returns:
            List of tokens.

        Example:
            >>> tokenizer.tokenize("[I+3][C][C][Branch1][C][C][C][I+3]")
            ['[I+3]', '[C]', '[C]', '[Branch1]', '[C]', '[C]', '[C]', '[I+3]']
        """
        if not selfies:
            return []

        try:
            # Use SELFIES library to split into tokens
            tokens = list(sf.split_selfies(selfies))
            return tokens
        except Exception:
            # If split_selfies fails, return empty list
            return []

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to SELFIES string.

        Args:
            tokens: List of tokens.

        Returns:
            Reconstructed SELFIES string.

        Example:
            >>> tokenizer.detokenize(['[I+3]', '[C]', '[I+3]'])
            '[I+3][C][I+3]'
        """
        # Filter out special tokens
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        # Simply concatenate - SELFIES tokens are already bracket-delimited
        return ''.join(filtered)

    def build_vocab(self, selfies_list: List[str]) -> Dict[str, int]:
        """Build vocabulary from a list of SELFIES strings.

        Args:
            selfies_list: List of SELFIES strings.

        Returns:
            Vocabulary dictionary (token -> id).
        """
        # Start with special tokens
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        current_id = len(self.SPECIAL_TOKENS)

        # Collect all unique tokens
        all_tokens = set()
        for selfies in selfies_list:
            tokens = self.tokenize(selfies)
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
        selfies: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, List[int]]:
        """Encode a SELFIES string to token IDs.

        Args:
            selfies: Input SELFIES string.
            add_special_tokens: Whether to add BOS/EOS tokens.
            padding: Whether to pad to max_length.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        tokens = self.tokenize(selfies)

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
        """Decode token IDs back to SELFIES string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded SELFIES string.
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
        selfies_list: List[str],
        add_special_tokens: bool = True,
        padding: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Encode a batch of SELFIES strings.

        Args:
            selfies_list: List of SELFIES strings.
            add_special_tokens: Whether to add BOS/EOS tokens.
            padding: Whether to pad sequences.

        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'.
        """
        results = [
            self.encode(selfies, add_special_tokens, padding)
            for selfies in selfies_list
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
            List of decoded SELFIES strings.
        """
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def verify_roundtrip(self, selfies: str) -> bool:
        """Verify that tokenization is invertible for a given string.

        Args:
            selfies: Input SELFIES string.

        Returns:
            True if detokenize(tokenize(selfies)) == selfies.
        """
        tokens = self.tokenize(selfies)
        reconstructed = self.detokenize(tokens)
        return reconstructed == selfies

    def save(self, path: str) -> None:
        """Save tokenizer to file.

        Args:
            path: Path to save the tokenizer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'tokenizer_type': 'selfies'  # Add type identifier
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SelfiesTokenizer':
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

    def get_placeholder_token_id(self) -> int:
        """Return the token ID for '[I+3]' placeholder.

        This is used by the sampler to enforce placeholder count constraints
        during generation (max 2 placeholders for polymer repeat units).

        Returns:
            Token ID for '[I+3]', or unk_token_id if not in vocabulary.
        """
        return self.vocab.get('[I+3]', self.unk_token_id)

    def get_star_token_id(self) -> int:
        """Return the token ID for placeholder (compatibility with PSmilesTokenizer).

        This method exists for API compatibility. In SELFIES, we use '[I+3]' instead of '*'.

        Returns:
            Token ID for '[I+3]'.
        """
        return self.get_placeholder_token_id()
