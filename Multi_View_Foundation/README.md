# Multi_View_Foundation

This folder is reserved for the multi-view foundation model described in Outline.md.
It is intentionally independent from Bi_Diffusion_* and AR_Transformer_*.

Status: F0-F7 implemented (paired index, multi-view embeddings, cross-view retrieval, multi-view property heads, OOD, inverse reranking, OOD-aware inverse objective, chemistry-physics analysis). Graph view is optional via config.
Alignment options:
- Frozen embeddings: `scripts/step1_train_alignment.py --train_alignment`
- End-to-end (fine-tune backbones): `scripts/step1_train_alignment.py --train_alignment_e2e`

OOD-aware objective step:
- `scripts/step6_ood_aware_inverse_objective.py`

Chemistry-physics analysis step:
- `scripts/step7_chem_physics_analysis.py`

Docs in this folder:
- Pipeline.md
- technical_guide.md
- results.md
