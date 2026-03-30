function [ptCloud,param]=loadptCloud(param)

% Read point cloud
ptCloud = pcread([param.ptCloudpathname param.ptCloudname]);
% Test if ptCloud has colors
param.iscolor=abs(1-isempty(ptCloud.Color));
% Remove outliers and make the origin (0,0,0)
x=ptCloud.Location(:,1);x=x-min(x);y=ptCloud.Location(:,2);y=y-min(y);z=ptCloud.Location(:,3);z=z-min(z);
if param.iscolor==1; ptCloud = pointCloud([x y z],'Color',ptCloud.Color); else ptCloud = pointCloud([x y z]);  end
% Remove Invalid points (Inf or Nan)
[ptCloud] = removeInvalidPoints(ptCloud);