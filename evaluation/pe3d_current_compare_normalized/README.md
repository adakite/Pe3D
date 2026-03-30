# Normalized Otira Comparison

Both MATLAB and current `pe3d` angle columns were normalized with `angle % 180` before generating the comparison plots and metrics.

Reference rows: 396
Candidate rows: 401
Matched rows: 396

Normalized map-angle MAE: 37.624 deg
Normalized X-view-angle MAE: 31.822 deg

Map-angle transform test after normalization:
- identity: 37.624 deg MAE
- mirror: 37.751 deg MAE
- mirror_plus90: 52.249 deg MAE
- mirror_minus90: 52.249 deg MAE
- plus90: 52.376 deg MAE
- minus90: 52.376 deg MAE

Best transform: `identity`

Interpretation: once both datasets are wrapped to the same `[0, 180)` convention, the current `pe3d` map-angle distribution is best aligned without an added `+90 deg` or `-90 deg` correction. Any earlier visible 90 deg shift came from historical convention/range differences in older evaluation snapshots, not from the current `pe3d` CSVs.
