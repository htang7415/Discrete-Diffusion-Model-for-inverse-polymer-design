"""Hugging Face compatible p-SMILES tokenizer implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer


class HFPSmilesTokenizer(PreTrainedTokenizer):
    """HF tokenizer preserving deterministic, invertible p-SMILES tokenization."""

    # Multi-character atoms (must be matched before single chars)
    MULTI_CHAR_ATOMS = [
        "Cl", "Br", "Si", "Na", "Li", "Ca", "Mg", "Al", "Sn", "Sb", "Se",
        "Fe", "Cu", "Zn", "Ni", "Co", "Mn", "Cr", "Ti", "Pt", "Pd", "Au",
        "Ag", "Hg", "Pb", "Bi", "As", "Te", "Ge", "Ga", "In", "Tl",
    ]

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "legacy_tokenizer_file": "tokenizer.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        vocab_file: Optional[str] = None,
        legacy_tokenizer_file: Optional[str] = None,
        **kwargs,
    ):
        loaded_vocab = vocab
        loaded_max_length = max_length

        if legacy_tokenizer_file:
            legacy_path = Path(legacy_tokenizer_file)
            if legacy_path.exists():
                with open(legacy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded_vocab = data.get("vocab", loaded_vocab)
                loaded_max_length = int(data.get("max_length", loaded_max_length))

        if loaded_vocab is None and vocab_file:
            vocab_path = Path(vocab_file)
            if vocab_path.exists():
                with open(vocab_path, "r", encoding="utf-8") as f:
                    loaded_vocab = json.load(f)

        if loaded_vocab is None:
            loaded_vocab = {}

        self.max_length = int(loaded_max_length)
        self.vocab: Dict[str, int] = dict(loaded_vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._compile_patterns()

        pad_token = kwargs.pop("pad_token", "[PAD]")
        mask_token = kwargs.pop("mask_token", "[MASK]")
        bos_token = kwargs.pop("bos_token", "[BOS]")
        eos_token = kwargs.pop("eos_token", "[EOS]")
        unk_token = kwargs.pop("unk_token", "[UNK]")
        model_max_length = kwargs.pop("model_max_length", self.max_length)

        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    def _compile_patterns(self) -> None:
        """Compile regex patterns used in tokenizer scanning."""
        self.bracket_pattern = re.compile(r"\[[^\[\]]+\]")
        self.ring_pattern = re.compile(r"%\d{2}")
        self.multi_atom_pattern = re.compile(
            "|".join(sorted(self.MULTI_CHAR_ATOMS, key=len, reverse=True))
        )

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            if text[i] == "[":
                match = self.bracket_pattern.match(text, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            if text[i] == "%":
                match = self.ring_pattern.match(text, i)
                if match:
                    tokens.append(match.group())
                    i = match.end()
                    continue

            match = self.multi_atom_pattern.match(text, i)
            if match:
                tokens.append(match.group())
                i = match.end()
                continue

            tokens.append(text[i])
            i += 1

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("[UNK]", 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.detokenize(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for p-SMILES tokenizer.")
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            special_ids = {self.pad_token_id, self.mask_token_id, self.bos_token_id, self.eos_token_id}
            return [1 if tok in special_ids else 0 for tok in token_ids_0]
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for p-SMILES tokenizer.")
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is not None:
            raise NotImplementedError("Pair sequences are not supported for p-SMILES tokenizer.")
        return [0] * (len(token_ids_0) + 2)

    # ------------------------------------------------------------------
    # Pipeline APIs (kept stable to avoid behavioral changes in scripts).
    # ------------------------------------------------------------------
    def tokenize(self, smiles: str) -> List[str]:  # type: ignore[override]
        return self._tokenize(smiles)

    def detokenize(self, tokens: List[str]) -> str:
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return "".join(filtered)

    def build_vocab(self, smiles_list: List[str]) -> Dict[str, int]:
        vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        current_id = len(self.SPECIAL_TOKENS)

        all_tokens = set()
        for smiles in smiles_list:
            all_tokens.update(self.tokenize(smiles))

        for token in sorted(all_tokens):
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
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.eos_token_id]

        attention_mask = [1] * len(ids)

        if padding:
            pad_length = self.max_length - len(ids)
            if pad_length > 0:
                ids = ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

        result: Dict[str, List[int]] = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result

    def decode(self, ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:  # type: ignore[override]
        tokens: List[str] = []
        for token_id in ids:
            token = self.id_to_token.get(token_id, "[UNK]")
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
        encoded = [
            self.encode(
                smiles,
                add_special_tokens=add_special_tokens,
                padding=padding,
                return_attention_mask=True,
            )
            for smiles in smiles_list
        ]
        return {
            "input_ids": [row["input_ids"] for row in encoded],
            "attention_mask": [row["attention_mask"] for row in encoded],
        }

    def batch_decode(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_list]

    def verify_roundtrip(self, smiles: str) -> bool:
        return self.detokenize(self.tokenize(smiles)) == smiles

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer JSON for pipeline compatibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vocab": self.vocab, "max_length": self.max_length}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HFPSmilesTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], max_length=int(data["max_length"]))

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_path = save_dir / f"{prefix}vocab.json"
        legacy_path = save_dir / f"{prefix}tokenizer.json"

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "max_length": self.max_length}, f, indent=2)

        return str(vocab_path), str(legacy_path)

    def get_star_token_id(self) -> int:
        return self.vocab.get("*", self.unk_token_id)


def load_polymer_tokenizer(
    results_dir: Union[str, Path],
    fallback_results_dir: Optional[Union[str, Path]] = None,
) -> HFPSmilesTokenizer:
    """Load tokenizer, preferring HF-pretrained assets with legacy fallback."""
    candidates: List[Path] = []
    for root in [results_dir, fallback_results_dir]:
        if root is None:
            continue
        root_path = Path(root)
        candidates.append(root_path / "tokenizer_hf")
        candidates.append(root_path / "tokenizer.json")

    for candidate in candidates:
        if candidate.is_dir():
            return HFPSmilesTokenizer.from_pretrained(str(candidate), local_files_only=True)
        if candidate.is_file():
            return HFPSmilesTokenizer.load(candidate)

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Tokenizer not found. Checked: {searched}")
