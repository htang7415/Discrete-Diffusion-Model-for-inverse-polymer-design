"""Graph Tokenizer for p-SMILES with deterministic, invertible transformations.

Converts p-SMILES strings to fixed-size graph representations (X, E, M) and back.
- X: node tokens (atom types)
- E: edge tokens (bond types)
- M: node mask (real vs padding)
"""

import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import rdmolops


class GraphTokenizer:
    """Bidirectional tokenizer for p-SMILES ↔ (X, E, M) graph representations.

    Design decisions:
    - Canonical atom ordering via RDKit CanonicalRankAtoms
    - Full symmetric edge matrix (Nmax × Nmax)
    - Star atoms (*) represented as RDKit dummy atoms (atomic_num=0)
    - Guaranteed invertibility for valid molecules
    """

    # Default edge vocabulary
    DEFAULT_EDGE_VOCAB = {
        'NONE': 0,
        'SINGLE': 1,
        'DOUBLE': 2,
        'TRIPLE': 3,
        'AROMATIC': 4,
        'MASK': 5
    }

    # RDKit bond type mappings
    RDKIT_BOND_TO_STR = {
        Chem.BondType.SINGLE: 'SINGLE',
        Chem.BondType.DOUBLE: 'DOUBLE',
        Chem.BondType.TRIPLE: 'TRIPLE',
        Chem.BondType.AROMATIC: 'AROMATIC'
    }

    STR_TO_RDKIT_BOND = {
        'SINGLE': Chem.BondType.SINGLE,
        'DOUBLE': Chem.BondType.DOUBLE,
        'TRIPLE': Chem.BondType.TRIPLE,
        'AROMATIC': Chem.BondType.AROMATIC
    }

    # Default stereochemistry vocabulary
    DEFAULT_STEREO_VOCAB = {
        'STEREONONE': 0,
        'STEREOANY': 1,
        'STEREOZ': 2,
        'STEREOE': 3,
        'STEREOCIS': 4,
        'STEREOTRANS': 5,
        'MASK': 6
    }

    # RDKit stereochemistry mappings
    RDKIT_STEREO_TO_STR = {
        Chem.BondStereo.STEREONONE: 'STEREONONE',
        Chem.BondStereo.STEREOANY: 'STEREOANY',
        Chem.BondStereo.STEREOZ: 'STEREOZ',
        Chem.BondStereo.STEREOE: 'STEREOE',
        Chem.BondStereo.STEREOCIS: 'STEREOCIS',
        Chem.BondStereo.STEREOTRANS: 'STEREOTRANS'
    }

    STR_TO_RDKIT_STEREO = {
        'STEREONONE': Chem.BondStereo.STEREONONE,
        'STEREOANY': Chem.BondStereo.STEREOANY,
        'STEREOZ': Chem.BondStereo.STEREOZ,
        'STEREOE': Chem.BondStereo.STEREOE,
        'STEREOCIS': Chem.BondStereo.STEREOCIS,
        'STEREOTRANS': Chem.BondStereo.STEREOTRANS
    }

    def __init__(
        self,
        atom_vocab: Dict[str, int],
        edge_vocab: Optional[Dict[str, int]] = None,
        stereo_vocab: Optional[Dict[str, int]] = None,
        Nmax: int = 64
    ):
        """Initialize GraphTokenizer.

        Args:
            atom_vocab: Mapping from atom symbols to IDs. Must include 'STAR', 'MASK', 'PAD'.
            edge_vocab: Mapping from bond types to IDs. Uses default if not provided.
            stereo_vocab: Mapping from stereochemistry types to IDs. Uses default if not provided.
            Nmax: Maximum number of atoms (fixed graph size).
        """
        self.atom_vocab = atom_vocab
        self.id_to_atom = {v: k for k, v in atom_vocab.items()}

        self.edge_vocab = edge_vocab if edge_vocab else self.DEFAULT_EDGE_VOCAB.copy()
        self.id_to_edge = {v: k for k, v in self.edge_vocab.items()}

        self.stereo_vocab = stereo_vocab if stereo_vocab else self.DEFAULT_STEREO_VOCAB.copy()
        self.id_to_stereo = {v: k for k, v in self.stereo_vocab.items()}

        self.Nmax = Nmax

        # Special token IDs
        self.star_id = atom_vocab['STAR']
        self.mask_id = atom_vocab['MASK']
        self.pad_id = atom_vocab['PAD']

        # Edge special IDs
        self.none_id = self.edge_vocab['NONE']
        self.edge_mask_id = self.edge_vocab['MASK']

        # Stereo special IDs
        self.stereo_none_id = self.stereo_vocab['STEREONONE']
        self.stereo_mask_id = self.stereo_vocab['MASK']

    @property
    def atom_vocab_size(self) -> int:
        """Return atom vocabulary size."""
        return len(self.atom_vocab)

    @property
    def edge_vocab_size(self) -> int:
        """Return edge vocabulary size."""
        return len(self.edge_vocab)

    @property
    def stereo_vocab_size(self) -> int:
        """Return stereochemistry vocabulary size."""
        return len(self.stereo_vocab)

    def _get_canonical_order(self, mol: Chem.Mol) -> List[int]:
        """Get canonical atom ordering using RDKit.

        Args:
            mol: RDKit Mol object.

        Returns:
            List of original atom indices in canonical order.
        """
        # Get canonical ranks for each atom
        ranks = list(Chem.CanonicalRankAtoms(mol))
        # Sort atom indices by their canonical rank
        sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i])
        return sorted_indices

    def encode(self, smiles: str) -> Dict[str, np.ndarray]:
        """Convert p-SMILES to (X, E, M, S) tensors.

        Args:
            smiles: p-SMILES string.

        Returns:
            Dictionary with:
                - X: (Nmax,) node token IDs
                - E: (Nmax, Nmax) edge token IDs (symmetric)
                - M: (Nmax,) node mask (1 for real, 0 for padding)
                - S: (Nmax, Nmax) stereochemistry token IDs (symmetric)

        Raises:
            ValueError: If SMILES is invalid or has more atoms than Nmax.
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.Nmax:
            raise ValueError(f"Molecule has {num_atoms} atoms, exceeds Nmax={self.Nmax}")

        # Get canonical atom ordering
        canonical_order = self._get_canonical_order(mol)

        # Build inverse mapping: original_idx -> canonical_idx
        orig_to_canonical = {orig: can for can, orig in enumerate(canonical_order)}

        # Initialize tensors
        X = np.full(self.Nmax, self.pad_id, dtype=np.int64)
        E = np.full((self.Nmax, self.Nmax), self.none_id, dtype=np.int64)
        M = np.zeros(self.Nmax, dtype=np.float32)
        S = np.full((self.Nmax, self.Nmax), self.stereo_none_id, dtype=np.int64)

        # Fill node tokens (X)
        for canonical_idx, orig_idx in enumerate(canonical_order):
            atom = mol.GetAtomWithIdx(orig_idx)

            # Handle dummy atoms (star/attachment points)
            if atom.GetAtomicNum() == 0:
                X[canonical_idx] = self.star_id
            else:
                symbol = atom.GetSymbol()
                X[canonical_idx] = self.atom_vocab.get(symbol, self.star_id)

        # Fill edge tokens (E) and stereochemistry (S)
        for bond in mol.GetBonds():
            orig_i = bond.GetBeginAtomIdx()
            orig_j = bond.GetEndAtomIdx()

            # Map to canonical indices
            can_i = orig_to_canonical[orig_i]
            can_j = orig_to_canonical[orig_j]

            # Get bond type
            bond_type = bond.GetBondType()
            bond_str = self.RDKIT_BOND_TO_STR.get(bond_type, 'SINGLE')
            bond_id = self.edge_vocab[bond_str]

            # Get bond stereochemistry
            stereo = bond.GetStereo()
            stereo_str = self.RDKIT_STEREO_TO_STR.get(stereo, 'STEREONONE')
            stereo_id = self.stereo_vocab[stereo_str]

            # Symmetric assignment
            E[can_i, can_j] = bond_id
            E[can_j, can_i] = bond_id
            S[can_i, can_j] = stereo_id
            S[can_j, can_i] = stereo_id

        # Fill node mask (M)
        M[:num_atoms] = 1.0

        return {'X': X, 'E': E, 'M': M, 'S': S}

    def decode(
        self,
        X: np.ndarray,
        E: np.ndarray,
        M: np.ndarray,
        S: Optional[np.ndarray] = None,
        sanitize: bool = True
    ) -> Optional[str]:
        """Convert (X, E, M, S) tensors back to p-SMILES.

        Args:
            X: (Nmax,) node token IDs.
            E: (Nmax, Nmax) edge token IDs.
            M: (Nmax,) node mask.
            S: (Nmax, Nmax) stereochemistry token IDs (optional for backward compatibility).
            sanitize: Whether to sanitize the molecule (may fail for invalid graphs).

        Returns:
            Canonical p-SMILES string, or None if conversion fails.
        """
        try:
            # Create editable RDKit molecule
            mol = Chem.RWMol()

            # Count real atoms
            num_atoms = int(M.sum())

            # Add atoms
            atom_idx_map = {}  # Maps our index to RDKit atom index
            for i in range(num_atoms):
                atom_id = int(X[i])
                symbol = self.id_to_atom.get(atom_id, 'C')

                if symbol == 'STAR':
                    # Dummy atom for attachment point
                    atom = Chem.Atom(0)
                elif symbol in ('PAD', 'MASK'):
                    # Skip special tokens
                    continue
                else:
                    atom = Chem.Atom(symbol)

                rdkit_idx = mol.AddAtom(atom)
                atom_idx_map[i] = rdkit_idx

            # Add bonds (only upper triangle to avoid duplicates)
            # Store atom pairs with stereochemistry to set after sanitization
            bond_stereo_to_set = []

            for i in range(num_atoms):
                if i not in atom_idx_map:
                    continue
                for j in range(i + 1, num_atoms):
                    if j not in atom_idx_map:
                        continue

                    edge_id = int(E[i, j])
                    bond_str = self.id_to_edge.get(edge_id, 'NONE')

                    if bond_str not in ('NONE', 'MASK'):
                        rdkit_bond = self.STR_TO_RDKIT_BOND.get(bond_str, Chem.BondType.SINGLE)
                        bond_idx = mol.AddBond(atom_idx_map[i], atom_idx_map[j], rdkit_bond)

                        # Record stereochemistry to set after sanitization
                        # Store atom indices (not bond index!) because bond indices change after sanitization
                        if S is not None:
                            stereo_id = int(S[i, j])
                            stereo_str = self.id_to_stereo.get(stereo_id, 'STEREONONE')
                            if stereo_str != 'STEREONONE':
                                rdkit_stereo = self.STR_TO_RDKIT_STEREO[stereo_str]
                                bond_stereo_to_set.append((atom_idx_map[i], atom_idx_map[j], rdkit_stereo))

            # Sanitize if requested (must be done BEFORE setting stereochemistry)
            if sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    # Try without full sanitization
                    pass

            # Set stereochemistry after sanitization
            # For each bond with stereochemistry, we need to set both stereo_atoms and stereo type
            for atom_i, atom_j, stereo_type in bond_stereo_to_set:
                try:
                    # Find the bond by its atom indices (not bond index, which may have changed)
                    bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
                    if bond is None:
                        continue

                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()

                    # For E/Z stereochemistry, we need to identify the stereo atoms
                    # These are the atoms adjacent to the double bond (not the double bond atoms themselves)
                    begin_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
                                      if n.GetIdx() != end_idx]
                    end_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(end_idx).GetNeighbors()
                                    if n.GetIdx() != begin_idx]

                    # Set stereo atoms (one from each side of the double bond)
                    # CRITICAL: RDKit's GetStereoAtoms() returns atoms in descending canonical rank order.
                    # To correctly reconstruct stereochemistry, we must select the neighbor with the
                    # highest canonical rank from each side of the double bond.
                    # Using neighbors[0] would select an arbitrary atom based on bond construction order,
                    # which causes E/Z stereochemistry to flip during decode.
                    if len(begin_neighbors) > 0 and len(end_neighbors) > 0:
                        canonical_ranks = list(Chem.CanonicalRankAtoms(mol))
                        begin_stereo = max(begin_neighbors, key=lambda i: canonical_ranks[i])
                        end_stereo = max(end_neighbors, key=lambda i: canonical_ranks[i])
                        bond.SetStereoAtoms(begin_stereo, end_stereo)
                        bond.SetStereo(stereo_type)
                except:
                    # If setting stereo fails, continue (molecule may not support it)
                    pass

            # CRITICAL: Must call AssignStereochemistry for RDKit to recognize the stereochemistry
            try:
                Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
            except:
                pass

            # Convert to Mol object
            mol = mol.GetMol()

            # Generate canonical SMILES
            smiles = Chem.MolToSmiles(mol)
            return smiles

        except Exception as e:
            return None

    def verify_roundtrip(self, smiles: str) -> bool:
        """Verify that encode-decode roundtrip preserves the molecule.

        Args:
            smiles: Input p-SMILES string.

        Returns:
            True if canonical(smiles) == canonical(decode(encode(smiles))).
        """
        try:
            # Get canonical input
            mol_orig = Chem.MolFromSmiles(smiles)
            if mol_orig is None:
                return False
            canonical_orig = Chem.MolToSmiles(mol_orig)

            # Encode and decode
            graph_data = self.encode(smiles)
            reconstructed = self.decode(
                graph_data['X'],
                graph_data['E'],
                graph_data['M'],
                graph_data.get('S')
            )

            if reconstructed is None:
                return False

            # Compare canonical forms
            return canonical_orig == reconstructed

        except Exception:
            return False

    def batch_encode(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Encode a batch of SMILES strings.

        Args:
            smiles_list: List of p-SMILES strings.

        Returns:
            Dictionary with batched arrays:
                - X: (B, Nmax)
                - E: (B, Nmax, Nmax)
                - M: (B, Nmax)
                - S: (B, Nmax, Nmax)
        """
        batch_X = []
        batch_E = []
        batch_M = []
        batch_S = []

        for smiles in smiles_list:
            data = self.encode(smiles)
            batch_X.append(data['X'])
            batch_E.append(data['E'])
            batch_M.append(data['M'])
            batch_S.append(data['S'])

        return {
            'X': np.stack(batch_X),
            'E': np.stack(batch_E),
            'M': np.stack(batch_M),
            'S': np.stack(batch_S)
        }

    def batch_decode(
        self,
        X: np.ndarray,
        E: np.ndarray,
        M: np.ndarray,
        S: Optional[np.ndarray] = None
    ) -> List[Optional[str]]:
        """Decode a batch of graph tensors.

        Args:
            X: (B, Nmax) node tokens.
            E: (B, Nmax, Nmax) edge tokens.
            M: (B, Nmax) node masks.
            S: (B, Nmax, Nmax) stereochemistry tokens (optional).

        Returns:
            List of p-SMILES strings (None for failed conversions).
        """
        batch_size = X.shape[0]
        return [
            self.decode(
                X[i],
                E[i],
                M[i],
                S[i] if S is not None else None
            )
            for i in range(batch_size)
        ]

    def save(self, path: str) -> None:
        """Save tokenizer configuration to JSON.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'atom_vocab': self.atom_vocab,
            'edge_vocab': self.edge_vocab,
            'stereo_vocab': self.stereo_vocab,
            'Nmax': self.Nmax
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GraphTokenizer':
        """Load tokenizer from JSON file.

        Args:
            path: Path to saved tokenizer.

        Returns:
            GraphTokenizer instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            atom_vocab=data['atom_vocab'],
            edge_vocab=data.get('edge_vocab'),
            stereo_vocab=data.get('stereo_vocab'),
            Nmax=data['Nmax']
        )


def build_atom_vocab_from_data(smiles_list: List[str]) -> Tuple[Dict[str, int], Dict]:
    """Build atom vocabulary from training data.

    Args:
        smiles_list: List of p-SMILES strings.

    Returns:
        Tuple of (atom_vocab, statistics).
    """
    atom_types: Set[str] = set()
    atom_counts = []
    bond_counts = []
    atom_type_counter = {}
    bond_type_counter = {}

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Count atoms
        num_atoms = mol.GetNumAtoms()
        atom_counts.append(num_atoms)

        # Extract atom types
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                symbol = 'STAR'
            else:
                symbol = atom.GetSymbol()
            atom_types.add(symbol)
            atom_type_counter[symbol] = atom_type_counter.get(symbol, 0) + 1

        # Count bonds
        num_bonds = mol.GetNumBonds()
        bond_counts.append(num_bonds)

        # Extract bond types
        for bond in mol.GetBonds():
            bond_type = str(bond.GetBondType())
            bond_type_counter[bond_type] = bond_type_counter.get(bond_type, 0) + 1

    # Build vocabulary (sorted for determinism)
    # Remove STAR from regular atoms as it's a special token
    regular_atoms = sorted([a for a in atom_types if a != 'STAR'])

    atom_vocab = {atom: idx for idx, atom in enumerate(regular_atoms)}
    next_id = len(atom_vocab)

    # Add special tokens
    atom_vocab['STAR'] = next_id
    atom_vocab['MASK'] = next_id + 1
    atom_vocab['PAD'] = next_id + 2

    # Compute statistics
    atom_counts_arr = np.array(atom_counts)
    atom_count_distribution = Counter(atom_counts)

    statistics = {
        'num_samples': len(smiles_list),
        'num_valid': len(atom_counts),
        'num_atoms': {
            'min': int(np.min(atom_counts_arr)),
            'max': int(np.max(atom_counts_arr)),
            'mean': float(np.mean(atom_counts_arr)),
            'std': float(np.std(atom_counts_arr)),
            'p50': float(np.percentile(atom_counts_arr, 50)),
            'p90': float(np.percentile(atom_counts_arr, 90)),
            'p95': float(np.percentile(atom_counts_arr, 95)),
            'p99': float(np.percentile(atom_counts_arr, 99)),
            'p100': int(np.percentile(atom_counts_arr, 100))
        },
        'num_bonds': {
            'min': int(np.min(bond_counts)) if bond_counts else 0,
            'max': int(np.max(bond_counts)) if bond_counts else 0,
            'mean': float(np.mean(bond_counts)) if bond_counts else 0,
        },
        'atom_type_distribution': atom_type_counter,
        'bond_type_distribution': bond_type_counter,
        'atom_count_distribution': {str(k): int(v) for k, v in atom_count_distribution.items()}
    }

    return atom_vocab, statistics
