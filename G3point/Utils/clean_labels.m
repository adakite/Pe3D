function [labels,nlabels,stack,isink]=clean_labels(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals)

tic;
display(['--- CLEANING SEGMENTATION']);

% ---- Determine if the normals at the border of labels are similars
% Find the indexborder nodes (No bonor and many other labels in the Nieghbourhood)
temp=param.nnptCloud-sum((labels(indNeighbors)==labels),2);
indborder=find(temp>=param.nnptCloud/4 & ndon==0);
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
Aangle=A./N;Aangle(N==0)=0;

% ---- Merge grains
% Matrix of labels to be merged
Mmerge=zeros(nlabels,nlabels);Mmerge(Aangle>param.maxangle2 | Aangle==0)=Inf;
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

% ---- Remove small labels
clear newstack newisink newlabels
indtokeep=find(nstack>=param.minnpoint);newlabels=0*labels;
for k=1:numel(indtokeep)
    newstack{k}=stack{indtokeep(k)};
    newisink(k)=isink(indtokeep(k));
    newlabels(newstack{k})=k;
end
stack=newstack;isink=newisink;labels=newlabels;nlabels=max(labels);nstack=cellfun(@numel, stack);

% ---- Remove flattish labels (probably not grains)
clear newstack newisink newlabels
r=zeros(nlabels,3);for k=1:nlabels;r(k,1:3)=svd(ptCloud.Location(stack{k},1:3)-mean(ptCloud.Location(stack{k},1:3)));end
indtokeep=find(r(:,3)./r(:,1)>param.minflatness | r(:,2)./r(:,1)>2.*param.minflatness);newlabels=0*labels;
for k=1:numel(indtokeep)
    newstack{k}=stack{indtokeep(k)};
    newisink(k)=isink(indtokeep(k));
    newlabels(newstack{k})=k;
end
stack=newstack;isink=newisink;labels=newlabels;nlabels=max(labels);nstack=cellfun(@numel, stack);

labels(labels==0)=NaN;

toc