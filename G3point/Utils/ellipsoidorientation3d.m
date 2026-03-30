function [granulo]=ellipsoidorientation3d(ptCloud,Ellipsoidm,granulo)    

delta=1e32;
nEllipsoidm=numel(Ellipsoidm);

k=0; sensorCenter = [mean(ptCloud.Location(:,1)),mean(ptCloud.Location(:,2))+delta,mean(ptCloud.Location(:,3))]; %x-y plot - mapview (angle with y axis)
for j=1:nEllipsoidm
    try
        if Ellipsoidm(j).fitok==1 && Ellipsoidm(j).Aqualityok==1
            k=k+1;
            u(k)=Ellipsoidm(j).axis1(1);v(k)=Ellipsoidm(j).axis1(2);w(k)=Ellipsoidm(j).axis1(3);granulo.radius(k)=Ellipsoidm(j).r(1);granulo.norm3D(k)=sqrt(u(k)^2+v(k)^2+w(k)^2);granulo.norm2Dxy(k)=sqrt(u(k)^2+v(k)^2);granulo.norm2Dxz(k)=sqrt(u(k)^2+w(k)^2);
            p1 = sensorCenter ;  p2 = [u(k),v(k),w(k)];
            angle = atan2(norm(cross(p1,p2)),p1*p2');
            if angle > pi/2 || angle < -pi/2
                u(k) = -u(k); v(k) = -v(k); w(k) = -w(k);
            end
            alpha(k)=atan(v(k)/u(k))+pi/2;
        end
    end
end
granulo.angle_Mview=alpha;
granulo.u_Mview=u;granulo.v_Mview=v;granulo.w_Mview=w;

% Normalize arrows and orient them in the upper part of a rose plot - here large-axis
k=0; sensorCenter = [mean(ptCloud.Location(:,1)),mean(ptCloud.Location(:,2)),mean(ptCloud.Location(:,3))+delta]; %x-z plot
for j=1:nEllipsoidm
    try
        if Ellipsoidm(j).fitok==1 && Ellipsoidm(j).Aqualityok==1
            k=k+1;
            u(k)=Ellipsoidm(j).axis1(1);v(k)=Ellipsoidm(j).axis1(2);w(k)=Ellipsoidm(j).axis1(3);granulo.radius(k)=Ellipsoidm(j).r(1);granulo.norm3D(k)=sqrt(u(k)^2+v(k)^2+w(k)^2);granulo.norm2Dxy(k)=sqrt(u(k)^2+v(k)^2);granulo.norm2Dxz(k)=sqrt(u(k)^2+w(k)^2);
            p1 = sensorCenter ;  p2 = [u(k),v(k),w(k)];
            angle = atan2(norm(cross(p1,p2)),p1*p2');
            if angle > pi/2 || angle < -pi/2
                u(k) = -u(k); v(k) = -v(k); w(k) = -w(k);
            end
            alpha(k)=atan(v(k)/w(k))+pi/2;          
        end
    end
end
granulo.angle_Xview=alpha;
granulo.u_Xview=u;granulo.v_Xview=v;granulo.w_Xview=w;

% Save the location of the ellipsoid center
k=0;
for j=1:nEllipsoidm
    if Ellipsoidm(j).fitok==1 && Ellipsoidm(j).Aqualityok==1
        k=k+1;granulo.Location(:,k)=Ellipsoidm(j).c;
    end
end

end