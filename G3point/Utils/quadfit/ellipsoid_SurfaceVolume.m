function [V,A]=ellipsoid_SurfaceVolume(radii)
% Compute the surface area and volume of an ellipsoid based on the radii 

V = 4/3*pi*radii(1)*radii(2)*radii(3);
A = ellipsoidSurfaceArea(radii');