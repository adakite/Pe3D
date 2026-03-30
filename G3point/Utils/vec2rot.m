function R=vec2rot(v1, v2, method)
% R*v1=v2
% v1 and v2 should be column vectors and 3x1
if size(v1,1)==1; v1=v1';end
if size(v2,1)==1; v2=v2';end

switch method
    case 'Rik'         
        v = cross(v1,v2);
        ssc = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
        R = eye(3) + ssc + ssc^2*(1-dot(v1,v2))/(norm(v))^2;
    case 'Kjetil'        
        G = [ dot(v1,v2)         -norm(cross(v1,v2)) 0;
              norm(cross(v1,v2)) dot(v1,v2)          0;
              0                  0                   1];
        Fi = [ v1  (v2-dot(v1,v2)*v1)/norm(v2-dot(v1,v2)*v1) cross(v2,v1) ];
        R = Fi*G*inv(Fi);
end

function x_skew=fcn_GetSkew(x)
x_skew=[0 -x(3) x(2);  x(3) 0 -x(1);  -x(2) x(1) 0];