function [x,y,z]=randsamplingellipsoid(a,b,c,cx,cy,cz,R,np)
u = rand(np,1);
v = rand(np,1);
theta = u*2*pi;
phi = acos(2*v-1);
sinTheta = sin(theta);
cosTheta = cos(theta);
sinPhi = sin(phi);
cosPhi = cos(phi);
x = a .* sinPhi .* cosTheta;
y = b .* sinPhi .* sinTheta;
z = c .* cosPhi;
% rotate surface mesh
X = [x(:),y(:),z(:)]*R;
% reconstruct mesh from points
x = X(:,1) + cx;  % add center offset
y = X(:,2) + cy;
z = X(:,3) + cz;
