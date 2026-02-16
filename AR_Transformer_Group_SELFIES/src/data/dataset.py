"""PyTorch Dataset classes for polymer data with Group SELFIES tokenization."""

import math
import warnings
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from .tokenizer import GroupSELFIESTokenizer


class PolymerDataset(Dataset):
    """Dataset for unlabeled polymer data (diffusion training).

    Uses Group SELFIES tokenization internally while storing p-SMILES.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: GroupSELFIESTokenizer,
        smiles_col: str = 'p_smiles',
        max_length: Optional[int] = None,
        cache_tokenization: bool = False,
        pad_to_max_length: bool = True,
        canonicalize: bool = True,
        pretokenized_group_selfies: bool = False
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with SMILES data.
            tokenizer: Tokenizer instance.
            smiles_col: Name of SMILES column.
            max_length: Maximum sequence length (overrides tokenizer).
            cache_tokenization: Whether to pre-tokenize and cache all samples.
            pad_to_max_length: Whether to pad each sample to tokenizer.max_length.
            canonicalize: Whether to canonicalize before grammar encoding.
            pretokenized_group_selfies: Whether smiles_col already stores Group SELFIES
                strings from Step0 cache (bypasses RDKit/grammar encoding in __getitem__).
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.cache_tokenization = cache_tokenization
        self.pad_to_max_length = pad_to_max_length
        self.canonicalize = canonicalize
        self.pretokenized_group_selfies = pretokenized_group_selfies
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if max_length:
            self.tokenizer.max_length = max_length

        if cache_tokenization:
            self._pretokenize()

    def _encode_group_selfies_value(self, gsf_value: str) -> Dict[str, List[int]]:
        """Encode precomputed Group SELFIES string via lightweight parsing path."""
        return self.tokenizer.encode_group_selfies(
            gsf_value,
            add_special_tokens=True,
            padding=self.pad_to_max_length,
            return_attention_mask=True
        )

    def _pretokenize(self):
        """Pre-tokenize all samples and cache them."""
        print(f"Pre-tokenizing {len(self)} samples...")
        for idx in tqdm(range(len(self)), desc="Tokenizing"):
            smiles = self.df.iloc[idx][self.smiles_col]
            if self.pretokenized_group_selfies:
                encoded = self._encode_group_selfies_value(smiles)
            else:
                encoded = self.tokenizer.encode(
                    smiles,
                    add_special_tokens=True,
                    padding=self.pad_to_max_length,
                    return_attention_mask=True,
                    canonicalize=self.canonicalize
                )
            self._cache[idx] = {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
            }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return cached if available
        if self.cache_tokenization and idx in self._cache:
            return self._cache[idx]

        smiles = self.df.iloc[idx][self.smiles_col]

        # Encode SMILES
        if self.pretokenized_group_selfies:
            encoded = self._encode_group_selfies_value(smiles)
        else:
            encoded = self.tokenizer.encode(
                smiles,
                add_special_tokens=True,
                padding=self.pad_to_max_length,
                return_attention_mask=True,
                canonicalize=self.canonicalize
            )

        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
        }


class PropertyDataset(Dataset):
    """Dataset for property prediction (supervised training).

    Uses Group SELFIES tokenization internally while storing p-SMILES.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: GroupSELFIESTokenizer,
        property_name: str,
        smiles_col: str = 'p_smiles',
        max_length: Optional[int] = None,
        normalize: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        cache_tokenization: bool = False
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with SMILES and property data.
            tokenizer: Tokenizer instance.
            property_name: Name of property column.
            smiles_col: Name of SMILES column.
            max_length: Maximum sequence length.
            normalize: Whether to normalize property values.
            mean: Mean for normalization (computed from data if not provided).
            std: Std for normalization (computed from data if not provided).
            cache_tokenization: Whether to pre-tokenize and cache all samples.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.property_name = property_name
        self.smiles_col = smiles_col
        self.normalize = normalize
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if max_length:
            self.tokenizer.max_length = max_length

        # Compute or set normalization parameters
        if normalize:
            self.mean = mean if mean is not None else df[property_name].mean()
            self.std = std if std is not None else df[property_name].std()
            if self.mean is None or not math.isfinite(self.mean):
                warnings.warn("Property mean is not finite; defaulting to 0.0.", RuntimeWarning)
                self.mean = 0.0
            if self.std is None or not math.isfinite(self.std) or self.std == 0:
                warnings.warn("Property std is not finite or zero; defaulting to 1.0.", RuntimeWarning)
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0

        if cache_tokenization:
            self._pretokenize()

    def _pretokenize(self):
        """Pre-tokenize all samples and cache them."""
        print(f"Pre-tokenizing {len(self)} samples...")
        for idx in tqdm(range(len(self)), desc="Tokenizing"):
            row = self.df.iloc[idx]
            smiles = row[self.smiles_col]
            value = row[self.property_name]

            encoded = self.tokenizer.encode(
                smiles,
                add_special_tokens=True,
                padding=True,
                return_attention_mask=True
            )

            # Normalize property value
            if self.normalize:
                value = (value - self.mean) / self.std

            self._cache[idx] = {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(value, dtype=torch.float32)
            }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return cached if available
        if self.cache_tokenization and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]
        value = row[self.property_name]

        # Encode SMILES
        encoded = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True
        )

        # Normalize property value
        if self.normalize:
            value = (value - self.mean) / self.std

        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(value, dtype=torch.float32)
        }

    def get_normalization_params(self) -> Dict[str, float]:
        """Get normalization parameters."""
        return {'mean': self.mean, 'std': self.std}

    def denormalize(self, value: float) -> float:
        """Denormalize a value."""
        return value * self.std + self.mean


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary of tensors.
    """
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result


def dynamic_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Collate with dynamic batch padding to reduce attention FLOPs.

    Args:
        batch: List of sample dictionaries.
        pad_token_id: Token ID used to pad input_ids.

    Returns:
        Batched dictionary with per-batch padded tensors.
    """
    batch_size = len(batch)
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    if 'labels' in batch[0]:
        result['labels'] = torch.stack([item['labels'] for item in batch])

    return result
