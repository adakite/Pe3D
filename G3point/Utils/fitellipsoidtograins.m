function [Ellipsoidm]=fitellipsoidtograins(Pebble,param,nlabels)

tic;
display(['--- FITTING ELLIPSOIDS TO GRAINS']);
for j=1:nlabels
    P=Pebble(j).Location;Ellipsoidm(j).fitok=1;Ellipsoidm(j).Aqualityok=0;
    % Shift point cloud to have only positive coordinates (problem with
    % quadfit if the point cloud is far from the coordinates of the origin (0,0,0))
    meanx=nanmean(P(:,1)); meany=nanmean(P(:,2)); meanz=nanmean(P(:,3));
    maxx=nanmax(P(:,1));   maxy=nanmax(P(:,2));   maxz=nanmax(P(:,3));
    minx=nanmin(P(:,1));   miny=nanmin(P(:,2));   minz=nanmin(P(:,3));
    % find the scaling factor
    dxmax=maxx-minx;dymax=maxy-miny;dzmax=maxz-minz;fact=1./nanmax(nanmax(dxmax,dymax),dzmax);
    switch param.fitmethod
        case 'direct'
            % Direct least squares fitting of ellipsoids under the constraint 4J - I^2 > 0. The constraint confines the class of ellipsoids to fit to those whose smallest radius is at least half of the largest radius.
            try
                % Ellispoid fit
                Ellipsoidm(j).p = ellipsoidfit_direct(fact.*(P(:,1)-meanx),fact.*(P(:,2)-meany),fact.*(P(:,3)-meanz));
                % Define the explicit parameters
                [center,radii,quat,R] = ellipsoid_im2ex(Ellipsoidm(j).p);
            catch
                Ellipsoidm(j).fitok=0;
            end
        case 'inertia'
            % Inertia ellipsoid of a set of 3D points
            try
                % Ellispoid fit
                [R,center,radii,angles,ell] = ellipsoidinertiasurf([fact.*(P(:,1)-meanx) fact.*(P(:,2)-meany) fact.*(P(:,3)-meanz)]); % find ellipsoid by svd method
            catch
                Ellipsoidm(j).fitok=0;
            end
        case 'simple'
            % Simple iterative least squares fitting of ellipsoids under the constraint k*J - I^2 > 0.
            try
                % Ellispoid fit
                Ellipsoidm(j).p = ellipsoidfit_simple(fact.*(P(:,1)-meanx),fact.*(P(:,2)-meany),fact.*(P(:,3)-meanz));
                % Define the explicit parameters
                [center,radii,quat,R] = ellipsoid_im2ex(Ellipsoidm(j).p);
            catch
                Ellipsoidm(j).fitok=0;
            end
        case 'koopmans'
            % Fit an ellipsoid to data using the nonlinear Koopmans method.
            try
                % Ellispoid fit
                Ellipsoidm(j).p = ellipsoidfit_koopmans(fact.*(P(:,1)-meanx),fact.*(P(:,2)-meany),fact.*(P(:,3)-meanz));
                % Define the explicit parameters
                [center,radii,quat,R] = ellipsoid_im2ex(Ellipsoidm(j).p);
            catch
                Ellipsoidm(j).fitok=0;
            end
        case 'direct_iterative'
            % Direct least squares fitting of ellipsoids under the constraint 4J - I^2 > 0. The constraint confines the class of ellipsoids to fit to those whose smallest radius is at least half of the largest radius.
            try
                % Ellispoid fit
                Ellipsoidm(j).p = ellipsoidfit(fact.*(P(:,1)-meanx),fact.*(P(:,2)-meany),fact.*(P(:,3)-meanz));
                % Define the explicit parameters
                [center,radii,quat,R] = ellipsoid_im2ex(Ellipsoidm(j).p);
            catch
                Ellipsoidm(j).fitok=0;
            end
    end
    if Ellipsoidm(j).fitok==1
        % Rescale the explicit parameters (the quaternions and R are unchanged by the scaling)
        center(1)=center(1)/fact+meanx;center(2)=center(2)/fact+meany;center(3)=center(3)/fact+meanz;radii=radii/fact;
        % Recompute the implicit form of the ellipsoid
        Ellipsoidm(j).p = ellipsoid_ex2im(center, radii, R);
        % Compute various metrics
        % Recompute the explicit form (Not mandatory and redondant)
        [Ellipsoidm(j).c,Ellipsoidm(j).r,Ellipsoidm(j).q,Ellipsoidm(j).R] = ellipsoid_im2ex(Ellipsoidm(j).p); 
        % Extract the vectors of the 3 axis
        Ellipsoidm(j).axis1=Ellipsoidm(j).R(:,1);Ellipsoidm(j).axis2=Ellipsoidm(j).R(:,2);Ellipsoidm(j).axis3=Ellipsoidm(j).R(:,3);
        % distance and r2 between points and the ellipsoid
        [Ellipsoidm(j).d,Ellipsoidm(j).r2] = ellipsoid_distance(P(:,1),P(:,2),P(:,3),Ellipsoidm(j).p); 
        % volume and surface area of the ellipsoid
        [Ellipsoidm(j).V,Ellipsoidm(j).A]=ellipsoid_SurfaceVolume(Ellipsoidm(j).r);
        % euler angles
        [Ellipsoidm(j).phi, Ellipsoidm(j).theta, Ellipsoidm(j).psi] = rotation3dToEulerAngles(Ellipsoidm(j).R);
        % surface ratio of the pebble/ellipsoid (in %)
        Ellipsoidm(j).Aratio=sum(Pebble(j).surface)./Ellipsoidm(j).A;        
        % Surface cover of the ellipsoid (using a discretized ellipsoid and closest neighbours) (in %)
        np=200;[xE,yE,zE]=randsamplingellipsoid(Ellipsoidm(j).r(1),Ellipsoidm(j).r(2),Ellipsoidm(j).r(3),Ellipsoidm(j).c(1),Ellipsoidm(j).c(2),Ellipsoidm(j).c(3),Ellipsoidm(j).R,np);Idx = knnsearch([xE yE zE],Pebble(j).Location);
        Ellipsoidm(j).Acover=100.*numel(unique(Idx))./np;   
        if Ellipsoidm(j).Acover>param.Aquality_thresh; Ellipsoidm(j).Aqualityok=1; end
    end
end
% for j=1:nlabels
%     if Ellipsoidm(j).fitok==1
%         P1=Pebble(j).Location;Ellipsoidm(j).Aqualityok=0;
%         % Translate the barycenter of the point cloud at the origin
%         P1(:,1)=P1(:,1)-nanmean(P1(:,1)); P1(:,2)=P1(:,2)-nanmean(P1(:,2)); P1(:,3)=P1(:,3)-nanmean(P1(:,3));
%         % Compute the best fitting plan and its unity normal vector
%         [A,B,C,distsigned,distabs]=fitplan(P1); normal=[A B 1]./sqrt(A^2+B^2+1^2);
%         % Compute the transformation matrix to rotate the plan normal towards
%         % the vertical
%         [T,R]=ptCloudTransformationFromTwoVectors(normal, [0 0 1]);
%         TF = isRigid(T);
%         if TF==1
%             % Transform the point cloud accordingly
%             ptCloudOut = pctransform(pointCloud(P1),T);P2=ptCloudOut.Location;
%         else
%             % Transform the point cloud accordingly to the imposed rotation (there is no reason
%             % here for T not to be a rigid transformation)
%             P2=(T.T(1:3,1:3)'*P1')';%plot3(P1(:,1),P1(:,2),P1(:,3),'k.');hold on;plot3(P2(:,1),P2(:,2),P2(:,3),'r.');
%         end
%         % Compute metrics - 2D area
%         % [~,Ellipsoidm(j).A2Dpts] = boundary(P2(:,1),P2(:,2),1); % 2D surface area of the point cloud
%         Ellipsoidm(j).A2Dpts=area(alphaShape(double(P2(:,1:2)),'HoleThreshold',100));
%         % surface cover of the pebble
%         Ellipsoidm(j).Acover=sum(Pebble(j).surface)./Ellipsoidm(j).A;
%         % Determine grain quality by computy the area covered by the grains
%         Ellipsoidm(j).Aquality=100.*Ellipsoidm(j).A2Dpts/(Ellipsoidm(j).A./2);
%         % Determine if grain quality is acceptable
%         if Ellipsoidm(j).Aquality>param.Aquality_thresh;   Ellipsoidm(j).Aqualityok=1;  end       
%     end
% end

toc;
   

end