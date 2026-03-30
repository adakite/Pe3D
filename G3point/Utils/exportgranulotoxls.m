function exportgranulotoxls(granulo,param)

tic;
display(['--- EXPORTING GRANULOMETRY']);

if param.savegranulo==1
    filename=[param.xlsfolder param.ptCloudname '_granulo.xlsx'];sheet = 1;
    % Header
    A = {'Ngrain','Xc','Yc','Zc','Dmax','Dmed','Dmin','angle_Mview','angle_Xview'};
    xlswrite(filename,A,sheet,'A1');
    % Data
    for i=1:numel(granulo.angle_Mview)
        % row of data
        A(i,:)={i,granulo.Location(1,i),granulo.Location(2,i),granulo.Location(3,i),granulo.diameter(1,i),granulo.diameter(2,i),granulo.diameter(3,i),granulo.angle_Mview(i).*180/pi,granulo.angle_Xview(i).*180/pi}; 
    end
    % insert row in Excel inside the 1st sheet, starting at cell specifed
    xlswrite(filename, A, sheet, 'A2');       
end

toc;