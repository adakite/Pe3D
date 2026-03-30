# Preprocess Mode Benchmark

This bundle benchmarks the two `pe3d` preprocessing modes (`matlab` and `fast`) on the Otira reference dataset and on `Nuage_HP.ply`.

## Runtime and counts

- `otira_matlab`: runtime `9.84 s`, labels `539`, accepted ellipsoids `427`
- `otira_fast`: runtime `3.10 s`, labels `505`, accepted ellipsoids `401`
- `nuage_hp_matlab`: runtime `122.70 s`, labels `5220`, accepted ellipsoids `5210`
- `nuage_hp_fast`: runtime `126.44 s`, labels `4911`, accepted ellipsoids `4907`

## Otira against MATLAB

- normalized comparison: `evaluation/preprocess_mode_benchmark/otira_vs_matlab_normalized/summary.json`
- raw comparison: `evaluation/preprocess_mode_benchmark/otira_vs_matlab/summary.json`

## Nuage_HP mode-to-mode comparison

- summary: `evaluation/preprocess_mode_benchmark/nuage_hp_mode_compare/summary.json`

