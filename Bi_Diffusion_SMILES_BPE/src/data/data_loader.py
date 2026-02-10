"""Data loading and preprocessing for polymer data."""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from ..utils.chemistry import canonicalize_smiles, is_valid_psmiles, compute_sa_score


class PolymerDataLoader:
    """Load and preprocess polymer data."""

    def __init__(self, config: Dict):
        """Initialize data loader.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.random_seed = config['data']['random_seed']
        self.repo_root = Path(__file__).resolve().parents[3]
        self.data_dir = self._resolve_data_path(config['paths']['data_dir'])
        self.results_dir = Path(config['paths']['results_dir'])

    def _resolve_data_path(self, path_str: str) -> Path:
        """Resolve data path against cwd first, then repository root."""
        path = Path(path_str)
        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        repo_path = self.repo_root / path
        if repo_path.exists():
            return repo_path

        # Keep a stable fallback for downstream error messages.
        return repo_path

    def load_unlabeled_data(self) -> pd.DataFrame:
        """Load unlabeled polymer SMILES data.

        Returns:
            DataFrame with cleaned p-SMILES.
        """
        polymer_file = self._resolve_data_path(self.config['paths']['polymer_file'])

        # Load gzipped CSV
        with gzip.open(polymer_file, 'rt') as f:
            df = pd.read_csv(f)

        # Column is named 'SMILES'
        smiles_col = 'SMILES'
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in {polymer_file}")

        return df[[smiles_col]].rename(columns={smiles_col: 'p_smiles'})

    def load_property_data(self, property_name: str) -> pd.DataFrame:
        """Load property data from CSV file.

        Args:
            property_name: Name of the property (e.g., 'Tg', 'Tm').

        Returns:
            DataFrame with p-SMILES and property values.
        """
        property_dir = self._resolve_data_path(self.config['paths']['property_dir'])
        property_file = property_dir / f"{property_name}.csv"

        if not property_file.exists():
            requested_name = f"{property_name}.csv".lower()
            for candidate in sorted(property_dir.glob("*.csv")):
                if candidate.name.lower() == requested_name:
                    property_file = candidate
                    break

        if not property_file.exists():
            available = ", ".join(sorted(p.stem for p in property_dir.glob("*.csv"))) or "none"
            raise FileNotFoundError(
                f"Property file not found: {property_file}\n"
                f"Resolved property_dir: {property_dir}\n"
                f"Available properties: {available}"
            )

        df = pd.read_csv(property_file)

        # The property column might have different names
        smiles_col = 'SMILES'
        value_col = None

        for col in df.columns:
            if col.lower() == smiles_col.lower():
                smiles_col = col
            elif col.lower() not in ['smiles']:
                value_col = col

        if value_col is None:
            # Use second column as value
            value_col = df.columns[1]

        df = df[[smiles_col, value_col]].copy()
        df.columns = ['p_smiles', property_name]

        return df

    def clean_and_filter(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'p_smiles'
    ) -> pd.DataFrame:
        """Clean and filter p-SMILES data.

        Args:
            df: DataFrame with SMILES column.
            smiles_col: Name of SMILES column.

        Returns:
            Cleaned DataFrame with only valid p-SMILES.
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=[smiles_col]).copy()

        # Remove empty strings
        df = df[df[smiles_col].notna() & (df[smiles_col] != '')]

        # Filter to valid p-SMILES (parseable and exactly 2 stars)
        valid_mask = df[smiles_col].apply(is_valid_psmiles)
        df = df[valid_mask].copy()

        # Canonicalize SMILES
        df[smiles_col] = df[smiles_col].apply(
            lambda x: canonicalize_smiles(x) or x
        )

        # Remove duplicates again after canonicalization
        df = df.drop_duplicates(subset=[smiles_col]).copy()

        return df.reset_index(drop=True)

    def compute_sa_scores(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'p_smiles'
    ) -> pd.DataFrame:
        """Compute SA scores for all molecules.

        Args:
            df: DataFrame with SMILES column.
            smiles_col: Name of SMILES column.

        Returns:
            DataFrame with SA score column added.
        """
        df = df.copy()
        df['sa_score'] = df[smiles_col].apply(compute_sa_score)
        return df

    def split_unlabeled(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split unlabeled data into train/val.

        Args:
            df: DataFrame with unlabeled data.

        Returns:
            Tuple of (train_df, val_df).
        """
        train_ratio = self.config['data']['unlabeled_train_ratio']

        train_df, val_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=self.random_seed
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def split_property(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split property data into train/val/test.

        Args:
            df: DataFrame with property data.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_ratio = self.config['data']['property_train_ratio']
        val_ratio = self.config['data']['property_val_ratio']
        test_ratio = self.config['data']['property_test_ratio']

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=self.random_seed
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=self.random_seed
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )

    def prepare_unlabeled_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare unlabeled data: load, clean, split.

        Returns:
            Dictionary with 'train' and 'val' DataFrames.
        """
        # Load data
        df = self.load_unlabeled_data()
        print(f"Loaded {len(df)} raw polymer SMILES")

        # Clean and filter
        df = self.clean_and_filter(df)
        print(f"After cleaning: {len(df)} valid p-SMILES")

        # Compute SA scores
        df = self.compute_sa_scores(df)

        # Split
        train_df, val_df = self.split_unlabeled(df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        return {'train': train_df, 'val': val_df}

    def prepare_property_data(self, property_name: str) -> Dict[str, pd.DataFrame]:
        """Prepare property data: load, clean, split.

        Args:
            property_name: Name of the property.

        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames.
        """
        # Load data
        df = self.load_property_data(property_name)
        print(f"Loaded {len(df)} samples for property {property_name}")

        # Clean and filter
        df = self.clean_and_filter(df)
        print(f"After cleaning: {len(df)} valid p-SMILES")

        # Compute SA scores
        df = self.compute_sa_scores(df)

        # Split
        train_df, val_df, test_df = self.split_property(df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return {'train': train_df, 'val': val_df, 'test': test_df}

    def get_statistics(self, df: pd.DataFrame, property_name: Optional[str] = None) -> Dict:
        """Compute statistics for a dataset.

        Args:
            df: DataFrame with data.
            property_name: Name of property column (if applicable).

        Returns:
            Dictionary of statistics.
        """
        stats = {
            'count': len(df),
            'unique_smiles': df['p_smiles'].nunique()
        }

        # Length statistics (round floats to 4 decimal places)
        lengths = df['p_smiles'].str.len()
        stats['length_mean'] = round(float(lengths.mean()), 4)
        stats['length_std'] = round(float(lengths.std()), 4)
        stats['length_min'] = int(lengths.min())
        stats['length_max'] = int(lengths.max())

        # SA score statistics (round floats to 4 decimal places)
        if 'sa_score' in df.columns:
            sa = df['sa_score'].dropna()
            stats['sa_mean'] = round(float(sa.mean()), 4)
            stats['sa_std'] = round(float(sa.std()), 4)
            stats['sa_min'] = round(float(sa.min()), 4)
            stats['sa_max'] = round(float(sa.max()), 4)

        # Property statistics (round floats to 4 decimal places)
        if property_name and property_name in df.columns:
            prop = df[property_name].dropna()
            stats[f'{property_name}_mean'] = round(float(prop.mean()), 4)
            stats[f'{property_name}_std'] = round(float(prop.std()), 4)
            stats[f'{property_name}_min'] = round(float(prop.min()), 4)
            stats[f'{property_name}_max'] = round(float(prop.max()), 4)

        return stats

    def get_available_properties(self) -> List[str]:
        """Get list of available property files.

        Returns:
            List of property names.
        """
        property_dir = self._resolve_data_path(self.config['paths']['property_dir'])
        property_files = list(property_dir.glob("*.csv"))
        return [f.stem for f in property_files]
