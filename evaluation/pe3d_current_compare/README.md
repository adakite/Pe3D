# Pe3d Current Comparison

This comparison measures the current `pe3d` Otira run against the bundled MATLAB reference.

Reference:
- MATLAB CSV: `G3point/Excel/Otira_1cm_grains_n1/Otira_1cm_grains.ply_granulo.csv`

Candidate:
- `pe3d_current`: `pe3d_outputs_eval_current/Excel/Otira_1cm_grains_n1/Otira_1cm_grains.ply_granulo.csv`

Key matched results:
- MATLAB rows: `396`
- `pe3d` rows: `401`
- Center-distance mean: `0.1763 m`
- `Dmax` MAE: `0.1289 m`
- `Dmed` MAE: `0.0894 m`
- `Dmin` MAE: `0.0781 m`
- `angle_Mview` MAE: `37.62 deg`
- `angle_Xview` MAE: `31.82 deg`

Artifacts:
- `summary.json`
- `diameter_cdfs.png`
- `angle_cdfs.png`
- `diameter_mae.png`
- `angle_mae.png`
