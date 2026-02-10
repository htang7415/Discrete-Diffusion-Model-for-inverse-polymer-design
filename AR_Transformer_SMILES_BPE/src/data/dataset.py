"""PyTorch Dataset classes for polymer data."""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from .tokenizer import PSmilesTokenizer


class PolymerDataset(Dataset):
    """Dataset for unlabeled polymer SMILES (diffusion training)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PSmilesTokenizer,
        smiles_col: str = 'p_smiles',
        max_length: Optional[int] = None,
        cache_tokenization: bool = False
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with SMILES data.
            tokenizer: Tokenizer instance.
            smiles_col: Name of SMILES column.
            max_length: Maximum sequence length (overrides tokenizer).
            cache_tokenization: Whether to pre-tokenize and cache all samples.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if max_length:
            self.tokenizer.max_length = max_length

        if cache_tokenization:
            self._pretokenize()

    def _pretokenize(self):
        """Pre-tokenize all samples and cache them."""
        print(f"Pre-tokenizing {len(self)} samples...")
        for idx in tqdm(range(len(self)), desc="Tokenizing"):
            smiles = self.df.iloc[idx][self.smiles_col]
            encoded = self.tokenizer.encode(
                smiles,
                add_special_tokens=True,
                padding=True,
                return_attention_mask=True
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
        encoded = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
        }


class PropertyDataset(Dataset):
    """Dataset for property prediction (supervised training)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PSmilesTokenizer,
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
