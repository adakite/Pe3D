# Orientation Focus Comparison

This comparison measures the archived Python baseline against the current
orientation-focused hybrid strategy on the bundled Otira dataset.

Reference:
- MATLAB CSV: `G3point/Excel/Otira_1cm_grains_n1/Otira_1cm_grains.ply_granulo.csv`

Candidates:
- `baseline`: archived `python_reference_v1`
- `orientation_focus`: current `hybrid_direct` strategy

Key matched results:
- Center-distance mean improved from `0.1900 m` to `0.1624 m`
- `Dmax` MAE improved from `0.1608 m` to `0.1183 m`
- `Dmed` MAE improved from `0.1097 m` to `0.0805 m`
- `Dmin` MAE improved from `0.0826 m` to `0.0711 m`
- `angle_Mview` MAE improved from `38.27 deg` to `37.83 deg`
- `angle_Xview` MAE improved from `33.37 deg` to `31.43 deg`

Implementation note:
- The current strategy keeps the stable bounded-PCA ellipsoid gate, upgrades
  accepted grains to a direct algebraic ellipsoid when that fit is reliable,
  and uses a MATLAB-like row convention for the map-view axis override.
