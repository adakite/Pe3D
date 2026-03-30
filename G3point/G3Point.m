%clearvars;
addpath('Utils','-end');addpath('Utils/quadfit','-end');  addpath('Utils/geom3d/geom3d/','-end'); 

%% Inputs
param.ptCloudname='';% 'Mangaweka.ply' 'Otira_1cm_grains.ply' 'Test1_clean_registered.ply'
if isempty(param.ptCloudname)==1;[param.ptCloudname,param.ptCloudpathname] = uigetfile('*.ply','Select the *.ply point cloud file');end

%% Loading data
[ptCloud,param]=loadptCloud(param);


%% Algorithm parameters - Compute point cloud size and scaling of the algorithm
% Load parameters
param=defineparameters(ptCloud,param);
% Display algorithm header
displayheader(ptCloud,param);

%% Denoise and decimate point cloud
% Denoise
if param.denoise==1;  ptCloud = pcdenoise(ptCloud); end
% Decimate
if param.decimate==1; ptCloud = pcdownsample(ptCloud, 'gridAverage', param.res);end

%% Remove the point that are localized in local minima (multiscale) to ease segmentation and delimitation of grains
if param.minima==1
    % Compute distance to local plan using a multiscale approach 
    [dist]=disttoplanemultiscale(ptCloud,param.minscale,param.maxscale,param.nscale);
    % Identify and remove local minimas
    ind=find(dist<prctile(dist,95));
    x=ptCloud.Location(:,1); y=ptCloud.Location(:,2); z=ptCloud.Location(:,3); ptCloud = pointCloud([x(ind) y(ind) z(ind)]);
end

%% Rotate and detrend the point cloud 
if param.rotdetrend==1 
    % Detrending assume that the plan is already orientated with the grains
    % upward in the z-direction (but the point cloud can be tilted compared to
    % the horizontal plan)
    [A,B,C,distsigned,distabs]=fitplan(ptCloud.Location);Normal=[-A -B 1]; [Normal]=adjustnormals3d(0,0,0,Normal,[0 0 1e32]); R=vec2rot(Normal, [0 0 1], 'Rik'); meanptCloud=mean(ptCloud.Location); xyzPoints=(R*(ptCloud.Location-meanptCloud)')'+meanptCloud; ptCloudRot = pointCloud(xyzPoints);
    % Then, if needed, remove a polynomial trend from the ptCloud
    x=ptCloudRot.Location(:,1);y=ptCloudRot.Location(:,2);z=ptCloudRot.Location(:,3); A = [ones(size(x)) x y x.^2 x.*y y.^2] \ z; z= z -( A(1) + A(2).*x + A(3).*y + A(4).*x.^2 + A(5).*x.*y + A(6).*y.^2); ptCloudRot = pointCloud([x y z]);
end

%% Show the clean point cloud
    % Plot
    if param.iplot==1;pcshow(ptCloudRot.Location,ptCloudRot.Location(:,3));set(gcf,'color','w');set(gca,'color','w');axis equal tight;cb = colorbar('north');set(cb,'position',[.5 .75 .1 .02]);ylabel(cb,'Elevation');axis off;end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'elevation'];print('-djpeg','-r500',nom);savefig(nom);close;end; %print('-dpdf','-painters',nom);

%% Segment and cluster the point cloud into a point cloud of potential grains
% Find neighbors of point cloud 
[indNeighbors,D]=knnsearch(ptCloud.Location,ptCloud.Location,'K',param.nnptCloud+1);indNeighbors=indNeighbors(:,2:end);D=D(:,2:end);
% determine node surface
surface=pi.*min(D,[],2).^2; 
% Compute normals and force them to point towards positive Z
normals = pcnormals(ptCloud,param.nnptCloud);[normals]=adjustnormals3d(ptCloud.Location(:, 1),ptCloud.Location(:, 2),ptCloud.Location(:, 3),normals,[mean(ptCloud.Location(:,1)), mean(ptCloud.Location(:,2)),10000]);
% Initial segmentation with Fastscape
[labels,nlabels,labelsnpoint,stack,nstack,ndon,isink]=segment_labels(ptCloudRot,param,indNeighbors);cmaplabels=rand(nlabels,3);
    % Plot
    if param.iplot==1;pcshow(ptCloud.Location,labels);colormap(cmaplabels);hold on;set(gcf,'color','w');set(gca,'color','w');axis equal tight;hold on;plot3(ptCloud.Location(isink,1),ptCloud.Location(isink,2),ptCloud.Location(isink,3),'.r');axis off;end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'labels_ini'];print('-djpeg','-r500',nom);savefig(nom);close;end; %print('-dpdf','-painters',nom);
% Cluster Labels to prevent over-segmentation
[labels,nlabels,stack,isink]=cluster_labels(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals);cmaplabels=rand(nlabels,3);
    % Plot
    if param.iplot==1;pcshow(ptCloud.Location,labels);colormap(cmaplabels);hold on;set(gcf,'color','w');set(gca,'color','w');axis equal tight;hold on;plot3(ptCloud.Location(isink,1),ptCloud.Location(isink,2),ptCloud.Location(isink,3),'.r');axis off;end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'labels_cluster'];print('-djpeg','-r500',nom);savefig(nom);close;end; %print('-dpdf','-painters',nom);    

%% Clean the segmentation
if param.clean==1
    % Clean the segmentation
    [labels,nlabels,stack,isink]=clean_labels(ptCloud,param,indNeighbors,labels,nlabels,stack,ndon,isink,surface,normals);
    % Plot
    if param.iplot==1;pcshow(ptCloud.Location,labels);colormap(cmaplabels);hold on;set(gcf,'color','w');set(gca,'color','w');axis equal tight;hold on;plot3(ptCloud.Location(isink,1),ptCloud.Location(isink,2),ptCloud.Location(isink,3),'.r');axis off;end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'labels_clean'];print('-djpeg','-r500',nom);savefig(nom);close;end; %print('-dpdf','-painters',nom);         
end

%% Generate a Pebble structure
for i=1:nlabels;ind=find(labels==i);Pebble(i).Location=ptCloud.Location(ind,:);Pebble(i).ind=ind;Pebble(i).surface=surface(ind);end 

%% Fitting cuboids (no dip allowed but azimuth is optimized)
% Fit
display(['--- FITTING CUBOIDS TO GRAINS']);
tic;for k=1:nlabels;Cuboidm(k)=pcfitcuboid(pointCloud(Pebble(k).Location));end;toc
    % Plot
    if param.iplot==1
    h=figure;plot3(ptCloud.Location(:,1),ptCloud.Location(:,2),ptCloud.Location(:,3),'.k','MarkerSize',1);axis equal tight;hold on;axis off
    for j=1:nlabels;  plot(Cuboidm(j));  end; cb = colorbar('north');set(cb,'position',[.5 .75 .1 .02]);ylabel(cb,'Labels');end; % How to plot colored cuboids?
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'fitted_cuboids'];print('-djpeg','-r500',nom);savefig(nom);close;end

%% Fitting ellipsoids
% fit
[Ellipsoidm]=fitellipsoidtograins(Pebble,param,nlabels);
    % Plot
    if param.iplot==1
    h=figure;plot3(ptCloud.Location(:,1),ptCloud.Location(:,2),ptCloud.Location(:,3),'.k','MarkerSize',1);axis equal tight;hold on;axis off
    for j=1:nlabels;  try if Ellipsoidm(j).fitok==1; plot_ellipsoid_im(Ellipsoidm(j).p,'EdgeColor',cmaplabels(j,:)); end; end; end;cb = colorbar('north');set(cb,'position',[.5 .75 .1 .02]);ylabel(cb,'Labels');end; 
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'fitted_ellipsoids'];print('-djpeg','-r500',nom);savefig(nom);close;end

%% Grain-size distribution
% Estimated based on the correctly fitted ellipsoids
[granulo]=grainsizedistribution(Ellipsoidm); 
    % Plot    
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.diameter(1,:),granulo.diameter_edges_log,'FaceColor','r');ylim([0 granulo.diameter_Nmax_log]);xlabel('diameter (m)');ylabel('N');title('a axis');axis square;set(gca,'xscale','log');set(gca,'fontsize',7);
    subplot(1,3,2);histogram(granulo.diameter(2,:),granulo.diameter_edges_log,'FaceColor','b');ylim([0 granulo.diameter_Nmax_log]);xlabel('diameter (m)');ylabel('N');title('b axis');axis square;set(gca,'xscale','log');set(gca,'fontsize',7);
    subplot(1,3,3);histogram(granulo.diameter(3,:),granulo.diameter_edges_log,'FaceColor','g');ylim([0 granulo.diameter_Nmax_log]);xlabel('diameter (m)');ylabel('N');title('c axis');axis square;set(gca,'xscale','log');set(gca,'fontsize',7);end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'grain-size_distribution_log'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.diameter(1,:),granulo.diameter_edges_lin,'FaceColor','r');ylim([0 granulo.diameter_Nmax_lin]);xlabel('diameter (m)');ylabel('N');title('a axis');axis square;set(gca,'fontsize',7);
    subplot(1,3,2);histogram(granulo.diameter(2,:),granulo.diameter_edges_lin,'FaceColor','b');ylim([0 granulo.diameter_Nmax_lin]);xlabel('diameter (m)');ylabel('N');title('b axis');axis square;set(gca,'fontsize',7);
    subplot(1,3,3);histogram(granulo.diameter(3,:),granulo.diameter_edges_lin,'FaceColor','g');ylim([0 granulo.diameter_Nmax_lin]);xlabel('diameter (m)');ylabel('N');title('c axis');axis square;set(gca,'fontsize',7);end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'grain-size_distribution_lin'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end       
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.vol, granulo.vol_edges_log, 'FaceColor','k');ylim([0 granulo.vol_Nmax_log]); xlabel('volume (m^3)');ylabel('N');title('volume');axis square;set(gca,'xscale','log');set(gca,'fontsize',7);
    subplot(1,3,2);histogram(granulo.area,granulo.area_edges_log,'FaceColor','k');ylim([0 granulo.area_Nmax_log]);xlabel('area (m^2)');  ylabel('N');title('area');axis square;set(gca,'xscale','log');set(gca,'fontsize',7);end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'volume-area_distribution_log'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.diameter(3,:)./granulo.diameter(1,:),granulo.nbin,'FaceColor','k');xlim([0 1]); xlabel('3D axis ratio');ylabel('N');title('c/a');axis square;set(gca,'fontsize',7);
    subplot(1,3,2);histogram(granulo.diameter(2,:)./granulo.diameter(1,:),granulo.nbin,'FaceColor','k');xlim([0 1]); xlabel('2D axis ratio');ylabel('N');title('b/a');axis square;set(gca,'fontsize',7);end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'axis-ratio_distribution'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.Acover,granulo.nbin,'FaceColor','k');xlabel('area cover (%)');ylabel('N');title('surface cover');axis square;xlim([0 100]);set(gca,'fontsize',7);end 
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'Acover_distribution'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end

%% Grain-size orientation
[granulo]=ellipsoidorientation3d(ptCloudRot,Ellipsoidm,granulo);   
    % Plot
    if param.iplot==1;figure;
    subplot(1,3,1);histogram(granulo.angle_Mview.*180/pi,20,'FaceColor','c');xlim([0 180]);xlabel('angle (^o)');ylabel('N');title('azimuth angle');axis square;set(gca,'fontsize',7);
    subplot(1,3,2);histogram(granulo.angle_Xview.*180/pi,20,'FaceColor','m');xlim([0 180]);xlabel('angle (^o)');ylabel('N');title('dip angle');axis square;set(gca,'fontsize',7);end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'grain-orientation_distribution'];print('-djpeg','-r500',nom);savefig(nom);print('-dpdf','-painters',nom);  close;end
    if param.iplot==1;figure;
    nhs=256;cmap=colormap(jet(nhs));close;rmax=max(granulo.radius);colorscale=linspace(0,rmax,nhs);rmax=max(max([Ellipsoidm.r]));
    subplot(2,2,1);
    h=compass(granulo.radius.*granulo.u_Mview./granulo.norm2Dxy,granulo.radius.*granulo.v_Mview./granulo.norm2Dxy);ylim([0 1]);xlabel('Map view');
    for i=1:length(h);[~,j]=min(abs(colorscale-granulo.radius(i)));set(h(i),'color',cmap(j,:));end
    set(findall(gcf, 'String', '210', '-or','String','240', '-or','String','270', '-or','String','300', '-or','String','330') ,'String', '  ');
    subplot(2,2,2);
    h=compass(granulo.radius.*granulo.u_Xview./granulo.norm2Dxz,granulo.radius.*granulo.w_Xview./granulo.norm2Dxz);ylim([0 1]);xlabel('X-view');
    for i=1:length(h);[~,j]=min(abs(colorscale-granulo.radius(i)));set(h(i),'color',cmap(j,:));end
    set(findall(gcf, 'String', '210', '-or','String','240', '-or','String','270', '-or','String','300', '-or','String','330') ,'String', '  ');
    subplot(2,2,3);
    rose(granulo.angle_Mview);h=gca;ylim([0 h.XLim(2)]);
    set(findall(gcf, 'String', '210', '-or','String','240', '-or','String','270', '-or','String','300', '-or','String','330') ,'String', '  ');
    subplot(2,2,4);
    rose(granulo.angle_Xview);h=gca;ylim([0 h.XLim(2)]);
    set(findall(gcf, 'String', '210', '-or','String','240', '-or','String','270', '-or','String','300', '-or','String','330') ,'String', '  ');
    end
    if param.saveplot==1 && param.iplot==1;nom=[param.figurefolder 'grain_orientation'];print('-djpeg','-r500',nom);savefig(nom);close;end

    h = polarhistogram(granulo.angle_Mview.*180/pi,20,'BinLimits',[-pi/2 pi/2])
    h.DisplayStyle = 'stairs';
    
%% Compare results from G3Points with a Wolman grid-by-number sampling
if param.gridbynumber==1; distri=gridbynumbersampling(granulo,ptCloud,Ellipsoidm,labels,cmaplabels,param.naxis,param.mindiam); end

%% Export results 
% Export grain point cloud to .ply files
if param.savegrain==1; exportgrains(ptCloud,labels,param,nlabels); end
% Export results to an Excel worksheet
if param.savegranulo==1;exportgranulotoxls(granulo,param); end