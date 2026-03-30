# Otira Comparison

This folder compares the archived Python baseline against the bundled MATLAB
reference and the new bounded-fit Python candidates.

Files:
- `summary.json`: matched metric summary
- `diameter_cdfs.png`: CDF comparison for `Dmax`, `Dmed`, `Dmin`
- `angle_cdfs.png`: CDF comparison for `angle_Mview`, `angle_Xview`
- `diameter_mae.png`: matched diameter MAE by candidate
- `angle_mae.png`: matched angle MAE by candidate

Key outcome:
- The bounded ellipsoid fit improves size parity versus the archived baseline.
- The orientation formula swap alone does not materially improve matched angle
  parity.
- A bounded rotation-refinement prototype was also tested and performed worse,
  so it was not adopted.

Matched Otira metrics:
- Baseline `Dmax` MAE: `0.1608 m`
- Bounded fit `Dmax` MAE: `0.1326 m`
- Baseline `Dmed` MAE: `0.1097 m`
- Bounded fit `Dmed` MAE: `0.0944 m`
- Baseline center-distance mean: `0.1900 m`
- Bounded fit center-distance mean: `0.1732 m`

Main caveat:
- Orientation remains the weakest part of the Python port and still needs a
  more faithful ellipsoid-axis or post-fit orientation strategy.
