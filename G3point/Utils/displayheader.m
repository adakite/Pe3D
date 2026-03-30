function displayheader(ptCloud,param)
display([' ']);
display([' ']);
display(['************************* 3D GRAIN SIZE DETECTION ON POINT CLOUD *************************']);
display([' ']);
display(['                                                                 Written by Philippe STEER']);
display(['                                                                 philippe.steer@univ-rennes1.fr']);
display(['                                                                 Geosciences Rennes']);
display(['                                                                 Université Rennes 1 / CNRS']);   

display(['--- POINT CLOUD:     ' param.ptCloudname ]);
display(['--- POINT CLOUD:     number of points '  num2str(ptCloud.Count)]);

display([' ']);
if param.iplot==1;        display(['--- PLOT RESULTS:                yes']);         else display(['--- PLOT RESULTS:                no']);end;
if param.saveplot==1;     display(['--- SAVE PLOTS   => in folder:' param.figurefolder ]);end
if param.denoise==1;      display(['--- DENOISE:                     yes']);         else display(['--- DENOISE:                     no']);end;
if param.decimate==1;     display(['--- DECIMATE:                    yes']);         else display(['--- DECIMATE:                    no']);end;
if param.minima==1;       display(['--- REMOVE LOCAL MINIMA:         yes']);         else display(['--- REMOVE LOCAL MINIMA:         no']);end;
if param.rotdetrend==1;   display(['--- ROTATE & DETREND:            yes']);         else display(['--- ROTATE & DETREND:            no']);end;
if param.clean==1;        display(['--- CLEAN SEGMENTATION:          yes']);         else display(['--- CLEAN SEGMENTATION:          no']);end;
if param.gridbynumber==1; display(['--- GRID-BY-NUMBER DISTRI:       yes']);         else display(['--- GRID-BY-NUMBER DISTRI:       no']);end;
if param.savegrain==1;    display(['--- EXPORT GRAINS:               yes']);   else       display(['--- EXPORT GRAINS:   no']);end;
if param.savegrain==1;    display(['--- SAVE GRAINS  => in folder:' param.grainfolder ]); end
end