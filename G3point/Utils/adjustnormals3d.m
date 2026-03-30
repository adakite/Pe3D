function [normals]=adjustnormals3d(x,y,z,normals,sensorCenter)
% Adjust normals flip the normals to be orientated towards the sensor center
% x,y,z: point coordinates
% u,v,w: normals
% ox,oy,oz: sensor center

u=normals(:,1);v=normals(:,2);w=normals(:,3);
for k = 1 : numel(x)
   p1 = sensorCenter - [x(k),y(k),z(k)];
   p2 = [u(k),v(k),w(k)];
   % Flip the normal vector if it is not pointing towards the sensor.
   angle = atan2(norm(cross(p1,p2)),p1*p2');
   if angle > pi/2 || angle < -pi/2
       u(k) = -u(k);
       v(k) = -v(k);
       w(k) = -w(k);
   end
end
normals(:,1)=u;normals(:,2)=v;normals(:,3)=w;

end