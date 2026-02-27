"""
Conditional Floor Plan Generation — Full Training Pipeline.

A hybrid system combining rule-based architectural constraints with a
conditional generative model (cGAN / Diffusion) to produce dynamic CAD
floor plans from irregular plot boundaries.

Modules
-------
data/       Dataset loaders, preprocessors, augmentation for CubiCasa5K & RPLAN.
models/     Generator, Discriminator, Encoder, and constraint networks.
training/   Training loop, loss functions, schedulers, checkpointing.
evaluation/ Metrics — IoU, adjacency accuracy, diversity, constraint violations.
export/     Predicted layout → DXF/CAD conversion with wall thickness, doors,
            windows, dimension lines.
"""

__version__ = "1.0.0"
