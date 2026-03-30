function [distri]=gridbynumbersampling(granulo,ptCloud,Ellipsoidm,labels,cmaplabels,diam,mindiam)

% Make a grid-by-number sampling (Wolman)
% diam= 1 = large diameter (a-axis) / 2 = intermediate diameter (b-axis) /3 = small diameter (c-axis)

% Extract grain diameter
r_all=2.*horzcat(Ellipsoidm.r);r_all=r_all(diam,:);r_all=r_all(r_all>mindiam); % select grains with diameter larger than 4 mm
% Determine egdges and bin center
distri.binedges(1)=0.0001;for i=2:100;distri.binedges(i)=distri.binedges(i-1)*1.5;end;ind=find(distri.binedges>=min(r_all)/2 & distri.binedges<=max(r_all)*2);distri.binedges=distri.binedges(ind); % distri.binedges using a log2 base
distri.bincenter=sqrt(distri.binedges(1:end-1).*distri.binedges(2:end)); % center of classes
% Granulo - all grains (assumed to be representative of a surface measurement)
distri.Nall=histcounts(r_all,distri.binedges,'Normalization','cdf');
% Granulo - grid by number (by conversion from area by number)
temp = histcounts(r_all,distri.binedges);distri.Ngbn=(cumsum(100*((100*temp./sum(temp)).*(distri.bincenter.^2))/sum((100*temp./sum(temp)).*(distri.bincenter.^2))))/100;

% Compare with Wolman performed on the labeled grains
distri.niter=50;distri.Nwol=zeros(distri.niter,numel(distri.bincenter));
for gg=1:distri.niter
    clear iwol dist
    r=rand(2,1);dx=max(granulo.diameter(diam,:));
    xgrid=min(ptCloud.Location(:,1))-r(1)*dx:dx:max(ptCloud.Location(:,1));ygrid=min(ptCloud.Location(:,2))-r(2)*dx:dx:max(ptCloud.Location(:,2));
    [Xgrid,Ygrid]=meshgrid(xgrid,ygrid);Xgrid=reshape(Xgrid,numel(Xgrid),1);Ygrid=reshape(Ygrid,numel(Ygrid),1);
    for i=1:numel(Xgrid)
        [dist(i),iwol(i)]=min(sqrt((ptCloud.Location(:,1)-Xgrid(i)).^2+(ptCloud.Location(:,2)-Ygrid(i)).^2));
    end
    iwol=iwol(dist<dx/10);
    iEllipsoidm=labels(iwol);iEllipsoidm(isnan(iEllipsoidm))=[];
    r_wol=horzcat(Ellipsoidm(iEllipsoidm).r);r_wol=2*r_wol(diam,:);
    r_wol=r_wol(r_wol>mindiam); % select grains with diameter larger than 4 mm
    distri.Nwol(gg,:)=histcounts(r_wol,distri.binedges,'Normalization','cdf');
    if gg~=distri.niter;iwol=[];dist=[];end % save to plot an example of the grid at the end
end

% Plot resulting distributions
figure;
semilogx(distri.bincenter,distri.Nall,'r','linewidth',2);hold on;
semilogx(distri.bincenter,distri.Ngbn,'g','linewidth',2);
semilogx(distri.bincenter,mean(distri.Nwol),'k','linewidth',2)
for gg=1:distri.niter;semilogx(distri.bincenter,distri.Nwol(gg,:),'b');hold on;end
%semilogx(distri.binedges(2:end),cumsum(mean(distri.Nwol),2)/sum(mean(distri.Nwol)),'k','linewidth',2)
xlabel('Diameter (m)');ylabel('CDF');legend('G3Point','G3Point>grid-by-number','Mean grean-by-number on labeled grains','grid-by-number on labeled grains', 'location','northwest');axis tight

% Plot sampling locations
figure
pcshow(ptCloud.Location,labels);colormap(cmaplabels);hold on;set(gcf,'color','w');set(gca,'color','w');axis equal tight;hold on; 
plot3(ptCloud.Location(iwol,1),ptCloud.Location(iwol,2),ptCloud.Location(iwol,3)+0.025,'.r','MarkerSize',50);axis off;

