function [granulo]=grainsizedistribution(Ellipsoidm)

% Find correctly fitted ellipsoids
ind=find([Ellipsoidm.fitok]==1 & [Ellipsoidm.Aqualityok]==1);

% --- Radius distribution
granulo.diameter=2.*horzcat(Ellipsoidm(ind).r);
% Number of bins 
granulo.nbin=ceil(sqrt(size(granulo.diameter,2)));
% Create a edge vector in linear and log-scale
granulo.diameter_edges_lin=linspace(min(granulo.diameter(3,:)),max(granulo.diameter(1,:)),granulo.nbin);   granulo.diameter_edges_log=logspace(log10(min(granulo.diameter(3,:))),log10(max(granulo.diameter(1,:))),granulo.nbin);
% Binning data
granulo.diameter_Nrmax_lin=histcounts(granulo.diameter(1,:),granulo.diameter_edges_lin);                                    granulo.diameter_Nrmax_log=histcounts(granulo.diameter(1,:),granulo.diameter_edges_log);
granulo.diameter_Nrmed_lin=histcounts(granulo.diameter(2,:),granulo.diameter_edges_lin);                                    granulo.diameter_Nrmed_log=histcounts(granulo.diameter(2,:),granulo.diameter_edges_log);
granulo.diameter_Nrmin_lin=histcounts(granulo.diameter(3,:),granulo.diameter_edges_lin);                                    granulo.diameter_Nrmin_log=histcounts(granulo.diameter(3,:),granulo.diameter_edges_log);
granulo.diameter_Nmax_lin=max(max(max(granulo.diameter_Nrmax_lin,granulo.diameter_Nrmed_lin),granulo.diameter_Nrmin_lin));  granulo.diameter_Nmax_log=max(max(max(granulo.diameter_Nrmax_log,granulo.diameter_Nrmed_log),granulo.diameter_Nrmin_log));

% --- Volume distribution
granulo.vol=horzcat(Ellipsoidm(ind).V);
% Number of bins 
% Create a edge vector in linear and log-scale
granulo.vol_edges_lin=linspace(min(granulo.vol),max(granulo.vol),granulo.nbin);     granulo.vol_edges_log=logspace(log10(min(granulo.vol)),log10(max(granulo.vol)),granulo.nbin);
% Binning data
granulo.vol_N_lin=histcounts(granulo.vol,granulo.vol_edges_lin);                    granulo.vol_N_log=histcounts(granulo.vol,granulo.vol_edges_log);
granulo.vol_Nmax_lin=max(granulo.vol_N_lin);                                        granulo.vol_Nmax_log=max(granulo.vol_N_log);

% --- Area distribution
granulo.area=horzcat(Ellipsoidm(ind).A);
% Number of bins 
% Create a edge vector in linear and log-scale
granulo.area_edges_lin=linspace(min(granulo.area),max(granulo.area),granulo.nbin);  granulo.area_edges_log=logspace(log10(min(granulo.area)),log10(max(granulo.area)),granulo.nbin);
% Binning data
granulo.area_N_lin=histcounts(granulo.area,granulo.area_edges_lin);                 granulo.area_N_log=histcounts(granulo.area,granulo.area_edges_log);
granulo.area_Nmax_lin=max(granulo.area_N_lin);                                      granulo.area_Nmax_log=max(granulo.area_N_log);

% --- Other metrics
for j=1:numel(ind)
granulo.r2(j)=Ellipsoidm(ind(j)).r2;
granulo.d(j) =sum(Ellipsoidm(ind(j)).d) ./numel(Ellipsoidm(ind(j)).d);
granulo.Acover(j)=Ellipsoidm(ind(j)).Acover;
end


end