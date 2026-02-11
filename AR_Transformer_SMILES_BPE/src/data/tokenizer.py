"""p-SMILES tokenizer with deterministic BPE over SMILES base tokens."""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PSmilesTokenizer:
    """Deterministic p-SMILES tokenizer with learned BPE merges.

    Base tokenization rules (priority order):
    1. Bracket tokens: [...] blocks -> one token
    2. Ring indices with %: %10, %11, etc. -> one token
    3. Multi-character atoms: Cl, Br, Si, etc. -> one token
    4. Single-character tokens: atoms, digits, symbols

    BPE is learned on top of this deterministic base tokenization.
    """

    MULTI_CHAR_ATOMS = [
        "Cl", "Br", "Si", "Na", "Li", "Ca", "Mg", "Al", "Sn", "Sb", "Se",
        "Fe", "Cu", "Zn", "Ni", "Co", "Mn", "Cr", "Ti", "Pt", "Pd", "Au",
        "Ag", "Hg", "Pb", "Bi", "As", "Te", "Ge", "Ga", "In", "Tl",
    ]

    SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]
    STRUCTURAL_CHARS = set("*()=#/\\")

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        merges: Optional[List[Tuple[str, str]]] = None,
        bpe_vocab_size: int = 512,
        min_frequency: int = 2,
        max_corpus_size: int = 200000,
        random_seed: int = 42,
    ):
        self.max_length = int(max_length)
        self.bpe_vocab_size = int(bpe_vocab_size)
        self.min_frequency = int(min_frequency)
        self.max_corpus_size = int(max_corpus_size)
        self.random_seed = int(random_seed)

        self.vocab = vocab if vocab else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()} if vocab else {}
        self.merges: List[Tuple[str, str]] = list(merges) if merges else []

        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.bracket_pattern = re.compile(r"\[[^\[\]]+\]")
        self.ring_pattern = re.compile(r"%\d{2}")
        self.ring_token_pattern = re.compile(r"(?:\d|%\d{2})")
        self.multi_atom_pattern = re.compile(
            "|".join(sorted(self.MULTI_CHAR_ATOMS, key=len, reverse=True))
        )

    def _tokenize_base(self, smiles: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(smiles)

        while i < n:
            if smiles[i] == "[":
                match = self.bracket_pattern.match(smiles, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            if smiles[i] == "%":
                match = self.ring_pattern.match(smiles, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            match = self.multi_atom_pattern.match(smiles, i)
            if match:
                tokens.append(match.group())
                i = match.end()
                continue

            tokens.append(smiles[i])
            i += 1

        return tokens

    @staticmethod
    def _merge_pair(tokens: List[str], left: str, right: str, merged: str) -> List[str]:
        out: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i + 1 < n and tokens[i] == left and tokens[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        merged_tokens = tokens
        for left, right in self.merges:
            merged_tokens = self._merge_pair(merged_tokens, left, right, left + right)
        return merged_tokens

    def tokenize(self, smiles: str) -> List[str]:
        base_tokens = self._tokenize_base(smiles)
        if not self.merges:
            return base_tokens
        return self._apply_merges(base_tokens)

    def detokenize(self, tokens: List[str]) -> str:
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return "".join(filtered)

    @staticmethod
    def _count_pair_frequencies(sequences: List[List[str]]) -> Counter:
        counts: Counter = Counter()
        for seq in sequences:
            if len(seq) < 2:
                continue
            counts.update((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        return counts

    def _touches_structural_symbols(self, token: str) -> bool:
        """Return True when token should remain atomic for constraints."""
        if any(ch in token for ch in self.STRUCTURAL_CHARS):
            return True
        return self.ring_token_pattern.fullmatch(token) is not None

    def _select_best_pair(
        self,
        pair_counts: Counter,
        symbol_set: set,
    ) -> Optional[Tuple[str, str]]:
        best_pair: Optional[Tuple[str, str]] = None
        best_key: Optional[Tuple[int, str, str]] = None

        for (left, right), count in pair_counts.items():
            if count < self.min_frequency:
                continue
            if self._touches_structural_symbols(left) or self._touches_structural_symbols(right):
                continue
            merged = left + right
            if merged in symbol_set:
                continue
            key = (-count, left, right)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (left, right)

        return best_pair

    def build_vocab(self, smiles_list: List[str]) -> Dict[str, int]:
        """Learn BPE merges and build final token vocabulary.

        Args:
            smiles_list: Training p-SMILES strings.

        Returns:
            Token -> id vocabulary dictionary.
        """
        if not smiles_list:
            raise ValueError("smiles_list is empty; cannot build tokenizer vocabulary")

        train_smiles = smiles_list
        if self.max_corpus_size > 0 and len(smiles_list) > self.max_corpus_size:
            rng = random.Random(self.random_seed)
            idxs = rng.sample(range(len(smiles_list)), self.max_corpus_size)
            idxs.sort()
            train_smiles = [smiles_list[i] for i in idxs]

        sequences: List[List[str]] = [self._tokenize_base(s) for s in train_smiles]

        symbol_set = set()
        for seq in sequences:
            symbol_set.update(seq)

        target_symbol_count = max(
            len(symbol_set),
            max(self.bpe_vocab_size - len(self.SPECIAL_TOKENS), 0),
        )

        merges: List[Tuple[str, str]] = []
        while len(symbol_set) < target_symbol_count:
            pair_counts = self._count_pair_frequencies(sequences)
            pair = self._select_best_pair(pair_counts, symbol_set)
            if pair is None:
                break

            left, right = pair
            merged = left + right
            merges.append((left, right))
            symbol_set.add(merged)

            sequences = [self._merge_pair(seq, left, right, merged) for seq in sequences]

        self.merges = merges

        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        current_id = len(self.SPECIAL_TOKENS)
        for token in sorted(symbol_set):
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
        return_attention_mask: bool = True,
    ) -> Dict[str, List[int]]:
        tokens = self.tokenize(smiles)

        unk_id = self.vocab.get("[UNK]", 0)
        ids = [self.vocab.get(token, unk_id) for token in tokens]

        if add_special_tokens:
            ids = [self.vocab["[BOS]"]] + ids + [self.vocab["[EOS]"]]

        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.vocab["[EOS]"]]

        attention_mask = [1] * len(ids)

        if padding:
            pad_id = self.vocab["[PAD]"]
            pad_length = self.max_length - len(ids)
            if pad_length > 0:
                ids = ids + [pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

        result = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for id_ in ids:
            token = self.id_to_token.get(id_, "[UNK]")
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return self.detokenize(tokens)

    def batch_encode(
        self,
        smiles_list: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
    ) -> Dict[str, List[List[int]]]:
        results = [
            self.encode(smiles, add_special_tokens, padding)
            for smiles in smiles_list
        ]

        return {
            "input_ids": [r["input_ids"] for r in results],
            "attention_mask": [r["attention_mask"] for r in results],
        }

    def batch_decode(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def verify_roundtrip(self, smiles: str) -> bool:
        return self.detokenize(self.tokenize(smiles)) == smiles

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "token_to_id": self.vocab,
            "max_length": self.max_length,
            "merges": [[a, b] for a, b in self.merges],
            "tokenizer_type": "smiles_bpe",
            "bpe_vocab_size": self.bpe_vocab_size,
            "min_frequency": self.min_frequency,
            "max_corpus_size": self.max_corpus_size,
            "random_seed": self.random_seed,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PSmilesTokenizer":
        with open(path, "r") as f:
            data = json.load(f)

        vocab = data.get("vocab", data.get("token_to_id", {}))
        merges = [tuple(p) for p in data.get("merges", [])]

        return cls(
            vocab=vocab,
            max_length=int(data.get("max_length", 128)),
            merges=merges,
            bpe_vocab_size=int(data.get("bpe_vocab_size", 512)),
            min_frequency=int(data.get("min_frequency", 2)),
            max_corpus_size=int(data.get("max_corpus_size", 200000)),
            random_seed=int(data.get("random_seed", 42)),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab["[PAD]"]

    @property
    def mask_token_id(self) -> int:
        return self.vocab["[MASK]"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["[BOS]"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["[EOS]"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["[UNK]"]

    def get_star_token_id(self) -> int:
        return self.vocab.get("*", self.unk_token_id)
