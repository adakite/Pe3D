function [dist]=disttoplanemultiscale(ptCloud,minscale,maxrad,nscale)
% This function computes the Euclidian average distance between the points of
% ptCloud to the local plans that best fit the local surrounding cloud
% defined by a radius
%
% --- Inputs:
% ptCloud: a matlab point cloud
% minscale : minimum radius for the local neighbour scale
% maxrad : maximum potential radius for the local neighbour scale
% nscale : number of scale to use
%
% --- Outputs:
% dist: the avergared distance to the local plans
%
% --- Philippe Steer - philippe.steer@gmail.com - Universite Rennes 1 - France

tic;
display(['--- COMPUTING DISTANCE TO LOCAL PLANS']);

% Compute a rough area for the ptcloud
xl=max(ptCloud.Location(:,1))-min(ptCloud.Location(:,1));yl=max(ptCloud.Location(:,2))-min(ptCloud.Location(:,2));area=xl.*yl;
nscale=max(1,round(area)); % Number of sliding window for 1 m radius (2 times the number of circles to fill up the param.area)
maxrad=min(maxrad,sqrt(area)); % Maximum radius
radvec=logspace(log10(minscale),log10(maxrad),nscale); % Range of radius used
ncirclesvec=round(nscale./radvec.^2);% Number of circles used: this should be large

% Initiate outputs
dist=zeros(ptCloud.Count,1);
nplane=zeros(ptCloud.Count,1);

for j=numel(radvec):-1:1
    % Identify some random points around which to subsample the initial point cloud
    ind_rand=round(rand([1,ncirclesvec(j)]).*(ptCloud.Count-1)+1);
    radius=radvec(j);   
    % This loop can be parallelized
    dist_temp=cell(numel(ind_rand),1);
    indices_sub = rangesearch(ptCloud.Location,ptCloud.Location(ind_rand,1:3),radius)';
    parfor i=1:numel(ind_rand)
        % Fit a plan to this local cloud
        [~,~,~,dist_temp{i},~]=fitplan(ptCloud.Location(indices_sub{i},1:3));      
    end
    for i=1:numel(ind_rand)
        % Compute outputs 
        dist(indices_sub{i})=dist(indices_sub{i})+dist_temp{i}./radius;         
        nplane(indices_sub{i})=nplane(indices_sub{i})+1;
    end
end
% End computations
dist=dist./nplane;

toc;

