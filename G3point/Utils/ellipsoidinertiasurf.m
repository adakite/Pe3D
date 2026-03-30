function [R, center, radii, angles, ell] = ellipsoidinertiasurf(xyz)
% Modified from the function equivalentEllipsoid written by David Legland
% Original author: David Legland / slightly modified by Philippe Steer to
% adapt the function to inertia "surfacic" ellipsoids sampled only by surface points
% See Geom3D toolbox

% number of points
n = size(xyz, 1);
% compute centroid
center = mean(xyz);
% compute the covariance matrix
covPts = cov(xyz)/n;
% perform a principal component analysis with 2 variables, 
% to extract equivalent axes
[U, S] = svd(covPts);
% extract length of each semi axis
%radii = sqrt(5) * sqrt(diag(S)*n)'; % Correct for solid ellipsoids
radii = sqrt(3) * sqrt(diag(S)*n)'; % Correct for surfacic ellipsoids - modification made by P. Steer (correct factor for hollow ellipsoids based on surface points https://en.wikipedia.org/wiki/List_of_moments_of_inertia)

% sort axes from greater to lower
[radii, ind] = sort(radii, 'descend');
% format U to ensure first axis xyz to positive x direction
U = U(ind, :);
if U(1,1) < 0
    U = -U;
    % keep matrix determinant positive
    U(:,3) = -U(:,3);
end
% convert axes rotation matrix to Euler angles
angles = rotation3dToEulerAngles(U);
% concatenate result to form an ellipsoid object
ell = [center, radii, angles];
R=eulerAnglesToRotation3d(-angles(1),angles(2),angles(3));R=R(1:3,1:3);