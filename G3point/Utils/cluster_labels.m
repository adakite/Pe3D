function [labels,nlabels,stack,isink]=cluster_labels(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals)

tic;
display(['--- CORRECTING SEGMENTATION']);

% ---- Inter-distance between isinks associated to each label
D1=pdist2(ptCloud.Location(isink,:),ptCloud.Location(isink,:));
% Radius of each label (assuming the surface correspond to a disk)
A=zeros(1,nlabels);for k=1:nlabels;A(k) = sum(surface(stack{k}));end;radius=sqrt(A./pi);

% Inter-distance by summing radius
D2=zeros(nlabels,nlabels);D2=D2+radius+radius';
Dist=zeros(nlabels,nlabels);ind=find(param.radfactor.*D2>D1);Dist(ind)=1;Dist=Dist-eye(size(Dist));

% ---- Determine if labels are neighbours
Nneigh=zeros(nlabels,nlabels);for k=1:nlabels;ind=unique(labels(indNeighbors(stack{k},:)));Nneigh(k,ind)=1;end

% ---- Determine if the normals at the border of labels are similars
% Find the indexborder nodes (No bonor and many other labels in the Nieghbourhood)
temp=param.nnptCloud-sum((labels(indNeighbors)==labels),2);indborder=find(temp>=param.nnptCloud/4 & ndon==0);
% Compute the angle of the normal vector between the neighbours of each
% grain/label
A=zeros(nlabels,nlabels);N=zeros(nlabels,nlabels);
for k=1:numel(indborder)
    % i = index of the point / j = index of the neighbourhood of i
    i=indborder(k); j=indNeighbors(i,:);
    % Take the normals vector for i and j (repmat on the normal vector for i to have the
    % same size as for j)
    P1=repmat(normals(i,:),param.nnptCloud,1);P2=normals(j,:);
    % Compute the angle between the normal of i and the normals of j
    % Add this angle to the angle matrix between each label
    A(labels(i),labels(j))=A(labels(i),labels(j))+anglerot2vecmat(P1,P2);
    % Number of occurence
    N(labels(i),labels(j))=N(labels(i),labels(j))+1;
end
% Take the mean value
Aangle=A./N;

% ---- Merge grains
% Matrix of labels to be merged
Mmerge=zeros(nlabels,nlabels);Mmerge(Dist<1 | Nneigh<1 | Aangle>param.maxangle1)=Inf;
% Mmerge=zeros(nlabels,nlabels);Mmerge(Dist<1 | Nneigh<1)=Inf;
% Mmerge=zeros(nlabels,nlabels);Mmerge(Dist<1)=Inf;
% Merge labels using a dbscan on isink using the relative distance matrix
[idx, ~] = dbscan(Mmerge,1,1,'Distance','precomputed');
newlabels=zeros(size(labels));newstack=cell(numel(unique(idx)),1);
for i=1:numel(unique(idx))
    ind=find(idx==i);
    for j=1:numel(ind)
        newlabels(stack{ind(j)})=i; 
        newstack{i}=[newstack{i} stack{ind(j)}];
    end
end
labels=newlabels;nlabels=max(labels);stack=newstack;nstack=cellfun(@numel, stack);
clear isink;for i=1:nlabels;[~,temp]=max(ptCloud.Location(stack{i},3));isink(i)=stack{i}(temp);end

toc