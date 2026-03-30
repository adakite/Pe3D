function exportgrains(ptCloud,labels,param,nlabels)

tic;
display(['--- EXPORTING GRAINS'])

if param.iscolor==1
    for j=1:nlabels
        ind=find(labels==j);
        ptCloud_tosave=pointCloud([ptCloud.Location(ind,1),ptCloud.Location(ind,2),ptCloud.Location(ind,3)],'Color',ptCloudgrains.Color(ind,1:3));
        pcwrite(ptCloud_tosave,[param.grainfolder param.ptCloudname '_grain_' num2str(j) '.ply']);
    end
else
    for j=1:nlabels
        ind=find(labels==j);
        ptCloud_tosave=pointCloud([ptCloud.Location(ind,1),ptCloud.Location(ind,2),ptCloud.Location(ind,3)]);
        pcwrite(ptCloud_tosave,[param.grainfolder param.ptCloudname '_grain_' num2str(j) '.ply']);
    end
end

toc;