function [labels,nlabels,labelsnpoint,stack,nstack,ndon,isink]=segment_labels(ptCloud,param,indNeighbors)

tic;
display(['--- SEGMENTING POINT CLOUD INTO GRAINS']);

% Compute recevers and slope
x=ptCloud.Location(:,1);y=ptCloud.Location(:,2);z=ptCloud.Location(:,3);nn=ptCloud.Count;
dx=x-x(indNeighbors);    dy=y-y(indNeighbors);    dz=z-z(indNeighbors);
s=dz./(dx.^2+dy.^2+dz.^2).^0.5;
% s=dz./(dx.^2+dy.^2).^0.5;
[slope,rec]=min(s,[],2);
rec=indNeighbors(sub2ind(size(indNeighbors),[1:numel(x)]',rec));  

% Find sinks nodes (local maxima)
% isink=find(slope>=0);rec(isink)=isink;
isink=find(slope>0);rec(isink)=isink;

% Donors
ndon = zeros(nn,1); % number of donors
donor = zeros(nn,param.nnptCloud); % donor list
for ij = 1:nn
    if rec(ij) ~= ij
        ijk = rec(ij);
        ndon(ijk) = ndon(ijk) + 1;
        donor(rec(ij),ndon(rec(ij))) = ij;
    end
end

% Build the stack
labels=zeros(nn,1);labelsk=zeros(nn,1);labelsnpoint=zeros(nn,1);
for k = 1:numel(isink)
    nstack(k) = 0;
    stack{k} = [];
    ij=isink(k);
    [stack{k},nstack(k)]=addtoStack(ij,ndon,donor,stack{k},nstack(k)); % recursive function
    labels(stack{k})=k;
    labelsnpoint(stack{k})=nstack(k);
end 
nlabels=numel(isink);

toc