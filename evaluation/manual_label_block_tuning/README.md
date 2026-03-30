## Manual-Label Tuning Summary

This folder records the `Nuage_HP` block-backend tuning against the manual `block` and `non-block` polygons in `pe3d_labels/Nuage_HP/`.

Point-wise agreement was evaluated inside the labeled zones only.

Baseline backend:

- true positives: `61034`
- false positives: `40467`
- false negatives: `111385`
- true negatives: `179667`
- precision: `0.6013`
- recall: `0.3539`
- specificity: `0.8162`
- F1: `0.4456`
- balanced accuracy: `0.5850`

Tuned backend:

- true positives: `119983`
- false positives: `80754`
- false negatives: `52436`
- true negatives: `139380`
- precision: `0.5977`
- recall: `0.6959`
- specificity: `0.6332`
- F1: `0.6431`
- balanced accuracy: `0.6645`

Interpretation:

- the tuned backend sacrifices some specificity,
- but it recovers far more labeled block points,
- and it improves both F1 and balanced accuracy substantially for the avalanche-deposit use case.
