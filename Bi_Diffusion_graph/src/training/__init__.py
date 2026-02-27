"""Training module exports with lazy loading."""

from importlib import import_module

__all__ = [
    "BackboneTrainer",
    "PropertyTrainer",
    "GraphBackboneTrainer",
    "GraphPropertyTrainer",
]


def __getattr__(name):
    if name == "BackboneTrainer":
        return import_module(".trainer_backbone", __name__).BackboneTrainer
    if name == "PropertyTrainer":
        return import_module(".trainer_property", __name__).PropertyTrainer
    if name == "GraphBackboneTrainer":
        return import_module(".graph_trainer_backbone", __name__).GraphBackboneTrainer
    if name == "GraphPropertyTrainer":
        return import_module(".graph_trainer_property", __name__).GraphPropertyTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
