function [param]=defineparameters(ptCloud,param)
% Define parameters for G3Point
% This function will try to read parameters associated to the chosen point
% cloud in the param.csv file that is locatedat the same path than the
% point clouds. If it does not find an associated set of parameters, the
% function will use the parameters defined manually below.

%% Try to read a parameter file (in csv format) and find a line corresponding to the used point cloud
if isfile([param.ptCloudpathname 'param.csv'])
    [N] = readvars([param.ptCloudpathname 'param.csv']);
    iname=strcmp(N,param.ptCloudname);inamenum=find(iname==1);
else
    inamenum=[];
end

%% If the file exist and has a line corresponding to the used point cloud then read the parameters from the file
if isempty(inamenum)==0
    display(['--- READING PARAMETERS FROM THE CSV FILE']);
    [b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]=readvars([param.ptCloudpathname 'param.csv'],'Range',[inamenum+1 2 inamenum+1 26]);
    % Yes (=1) or No (=0) parameters
    param.iplot           = b;               % Plot results
    param.saveplot        = c;               % Save plot
    param.denoise         = d;               % Denoise the point cloud
    param.decimate        = e;               % Decimate the point cloud
    param.minima          = f;               % Remove local minima to ease segmentation
    param.rotdetrend      = g;               % Rotate and detrend the point cloud
    param.clean           = h;               % Cleaning segmentation
    param.gridbynumber    = i;               % Perform a virtual grid-by-number grain size distribution
    param.savegranulo     = j;               % Export granulometry to a xls file
    param.savegrain       = k;               % Export grain point cloud in .ply
    % To decimate the point cloud (used if para.decimate==1)   
    param.res             = l;               % New resolution of the point cloud
    % To remove local minima before segmentation (used if param.minima==1) !! TIME-CONSUMING !!
    param.nscale          = m;               % Number of scale to use
    param.minscale        = n;               % Minimum scale(radius)
    param.maxscale        = o;               % Maximum potential scale (radius)
    % Segmentation parameters
    param.nnptCloud       = p;               % K nearest neighbours to route water with Fastscape
    param.radfactor       = q;               % Prefactor that is used to determine if two grains should be merged
    param.maxangle1       = r;               % Maximum angle above which the normals between two labels are considered different
    % Cleaning segmentation parameters  (used if param.clean==1)
    param.maxangle2       = s;               % Maximum angle above which the normals between two labels are considered different
    param.minflatness     = t;               % Minimum flatness that should have a grain
    param.minnpoint=max(param.nnptCloud,y);  % Minimal number of points that should contain a grain
    % Fitting method    
    param.fitmethod       = char(u);         % (['direct'] / 'simple' / 'koopmans' / 'inertia' / 'direct_iteratve'/ but 'direct' works much better - and yet use a strong constraint)
    param.Aquality_thresh = v;               % Minimum surface cover threshold for the ellipsoid (in %)
    % To perform a grid-by-number sampling (if param.gridbynumber==1)    
    param.mindiam         = w;               % Minimum grain diameter considered
    param.naxis           = x;               % which axis to use ? (1=a-axis, 2=b-axis, 3=c-axis)
    param.dx_gbn          = z;               % grid spacing used for the grid-by-number sampling of grain for gsd
else
    %% Otherwise define the parameters manually  
    display(['--- READING PARAMETERS MANUALLY FROM defineparameters.m']);
    % Yes (=1) or No (=0) parameters
    param.iplot=1;                           % Plot results
    param.saveplot=1;                        % Save plot
    param.denoise=1;                         % Denoise the point cloud
    param.decimate=0;                        % Decimate the point cloud
    param.minima=0;                          % Remove local minima to ease segmentation
    param.rotdetrend=1;                      % Rotate and detrend the point cloud
    param.clean=1;                           % Cleaning segmentation
    param.gridbynumber=1;                    % Perform a virtual grid-by-number grain size distribution
    param.savegranulo=1;                     % Export granulometry to a xls file
    param.savegrain=1;                       % Export grain point cloud in .ply
    % To decimate the point cloud (used if para.decimate==1)
    param.res=0.002;                         % New resolution of the point cloud
    % To remove local minima before segmentation (used if param.rotdetrend==1) !! TIME-CONSUMING !!
    param.nscale=4;                          % Number of scale to use
    param.minscale=0.04;                     % Minimum scale(radius)
    param.maxscale=2;                        % Maximum potential scale (radius)
    % Segmentation parameters
    param.nnptCloud=20;                      % K nearest neighbours to route water with Fastscape
    param.radfactor=0.6;                     % Prefactor that is used to determine if two grains should be merged
    param.maxangle1=60;                      % Maximum angle above which the normals between two labels are considered different
    % Cleaning segmentation parameters  (used if param.clean==1)
    param.maxangle2=10;                      % Maximum angle above which the normals between two labels are considered different
    param.minflatness=0.1;                   % Minimum flatness that should have a grain
    param.minnpoint=max(param.nnptCloud,50); % Minimal number of points that should contain a grain
    % Fitting method
    param.fitmethod='direct';                % (['direct'] / 'simple' / 'koopmans' / 'inertia' / 'direct_iteratve'/ but 'direct' works much better - and yet use a strong constraint)
    param.Aquality_thresh=10;                % Minimum surface cover threshold for the ellipsoid (in %)
    % To perform a grid-by-number sampling (if param.gridbynumber==1)
    param.mindiam=0.04;                      % Minimum grain diameter considered
    param.naxis=2;                           % which axis to use ? (1=a-axis, 2=b-axis, 3=c-axis)
end

%% Create folders and define folder names
% Creat Figure and Grain folders
if param.savegrain==1;    temp = exist('Grain/','dir');  if temp==0; mkdir('Grain/');  end; end
if param.saveplot==1;     temp = exist('Figure/','dir'); if temp==0; mkdir('Figure/'); end; end
if param.savegranulo==1;  temp = exist('Excel/','dir');  if temp==0; mkdir('Excel/');  end; end
% Find an appropriate folder name
grainfolderexist=1;figurefolderexist=1;xlsfolderexist=1;nfolder=0;
while grainfolderexist>0 || figurefolderexist>0 || xlsfolderexist>0
    nfolder=nfolder+1;
    param.grainfolder  = ['Grain/'  param.ptCloudname(1:end-4) '_n' num2str(nfolder) '/'];
    param.figurefolder = ['Figure/' param.ptCloudname(1:end-4) '_n' num2str(nfolder) '/'];
    param.xlsfolder    = ['Excel/'  param.ptCloudname(1:end-4) '_n' num2str(nfolder) '/'];    
    grainfolderexist   = exist(param.grainfolder,'dir');
    figurefolderexist  = exist(param.figurefolder,'dir'); 
    xlsfolderexist     = exist(param.xlsfolder,'dir');    
end
% Create folders
if param.savegrain==1;   mkdir(param.grainfolder);  end
if param.saveplot==1;    mkdir(param.figurefolder); end
if param.savegranulo==1; mkdir(param.xlsfolder);    end

end
