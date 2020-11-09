%m1 = ft_read_atlas('HMAT_Right_M1.nii')  % goes deeper in the sulcus
afni = ft_read_atlas('~/soft/fieldtrip/template/atlas/afni/TTatlas+tlrc.HEAD');  
% pts_converted = mni2icbm_spm( pts )
% atlas.  pts_converted = mni2icbm_spm( pts )
atlas = ft_convert_units(afni,'cm'); % ftrop and our sourcemodels have cm units
srsstd = load('~/soft/fieldtrip/template/sourcemodel/standard_sourcemodel3d5mm'); % 5mm spacing
srsstd_1cm = load('~/soft/fieldtrip/template/sourcemodel/standard_sourcemodel3d10mm'); % 5mm spacing


data_dir = getenv('DATA_DUSS');
%subjstr = 'S10';

if ~exist("subjstr")
  subjstr = 'S02';
end
fprintf('Preparing source coordinates for subject %s',subjstr)

basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

figVis = 'off';
%figVis = 'on';
do_create_surround_pts = 1;
%surround_radius = 0.8;
surround_radius = 1.2;
%surround_radius = 1e-10;  
radius_inc = 0.1;
min_number_of_pts = 90;

if ~do_create_surround_pts
  surround_radius = 1e-10;
  min_number_of_pts = 8;
end

scalemat = containers.Map;
% x -- between ears (positive means right hemisphere), z -- vert
sx = 1; sy=1; sz=1.09;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S02') = S;

sx = 1; sy=1; sz=1.09;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S01') = S;

sx = 1; sy=1; sz=1;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S03') = S;

sx = 1.04; sy=1; sz=1.06;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S04') = S;

sx = 1; sy=1; sz=1.03;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S05') = S;

sx = 1; sy=1; sz=1.0;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S06') = S;

sx = 1; sy=1; sz=1.0;
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S07') = S;

sx = 1; sy=1; sz=1.01; 
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S08') = S;

sx = 1; sy=1; sz=1.0; 
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S09') = S;

sx = 1; sy=1; sz=1.0; 
S = [ [sx 0 0]; [0 sy 0]; [0 0 sz] ]  ;
scalemat('S10') = S;


S = [ [1 0 0]; [0 1 0]; [0 0 1] ]  ;
scalemat('S01') = S;
scalemat('S02') = S;
scalemat('S03') = S;
scalemat('S04') = S;
scalemat('S05') = S;
scalemat('S06') = S;
scalemat('S07') = S;
scalemat('S08') = S;
scalemat('S09') = S;
scalemat('S10') = S;
save("head_scalemat",'scalemat');

S = scalemat(subjstr);

% I first find map between head-alagned grids and standards grids, 
% then transform Jan coords from the papers to actual (head-related) coords and then add surround

% Marius takes the cortical grid, runs ft_sourceanalysis with data being the cov matrix only 
%    (output of of ft_timelockanalysis)
% takes spatial filter (trial indep), mutliplies by hand the cleaned data
% puts to source_time the output of it (entire 3D grid) and parcel.pos

% I dont want to use their ft_parcel, so I can just take MNI coords of the parcel points that I need and then convert them using my favorite matrix probably
% but I also want to make sure conversion works and plot the outcome

%TODO: what does 'projectmom' do?

% get matrix mapping MNI to my coords
%vecinds = [1 2 3];
%vecinds = [2 3 4];
%vecinds = [1 200 3000 10000];
%vecinds = randi( [1,  length( hdmf.mni_aligned_grid.pos ) ], 1, 4 );
vecinds = [8157       16509          99       22893];          % don't touch these indices!
preX = srsstd.sourcemodel.pos;   % in each ROW -- x,y,z
preY = hdmf.mni_aligned_grid.pos;
% M * X = Y 
X0 = preX( vecinds, : );  %3x3
Y0 = preY( vecinds, : );  %3x3

X1 = transpose(X0);
Y1 = transpose(Y0);

%X = [X1; [0 0 0 1] ];
%Y = [Y1; [0 0 0 1] ];
X = [X1; [1 1 1 1] ];
Y = [Y1; [1 1 1 1] ];

d = det(Y);
if abs(d) < 1e-10
  printf("Bad selection of vectors")
  return
end

M = Y * inv(X);
%M = Y \ X
tryinds = [7 13 567];
ii = 2;
dev = M * transpose( [ preX(tryinds(ii), :) 1] ) - transpose( [preY(tryinds(ii),:) 1] );
devv = dev(1:3);
fprintf('dev = %d\n',devv)

%dev = M * transpose( preX(tryinds(ii), :) ) - transpose( preY(tryinds(ii),:) )

%return

load('coordsJan.mat')    % contains labels besides other things
%coords_Jan_MNI =  transpose( [ [33 -22 57] ; [-33 -22 57] ] ) / 10. ;
coords_Jan_MNI = transpose( coords );
coords_Jan_MNI_t = [ coords_Jan_MNI; ones( 1, size(coords_Jan_MNI,2) ) ];
yy = M * coords_Jan_MNI_t ;
coords_Jan_actual = transpose( yy(1:3,:)  );

%coords_Jan_actual
%coords_Jan_actual_upd = coords_Jan_actual;
coords_Jan_actual_upd = projectPtOnBrainSurf(hdmf.hdm, coords_Jan_actual, 1);

coords_Jan_actual = coords_Jan_actual_upd;

%%%%%%%%%  take all 3D grid pts and select those that has to do with some Brodmann areas
%%%%%%%%%  Marius's script for every cortical grid point gets which elements of the atlas it corresponds to 


cfg = [];
cfg.atlas = atlas;
%cfg.roi = {'Brodmann area 4'};
cfg.roi = {'Brodmann area 6'};      %6-M1, SMA, preSMA;  4-PMC
cfg.roi = {'Brodmann area 6' 'Brodmann area 4'};
cfg.roi = {'Brodmann area 6' 'Brodmann area 4' 'Brodmann area 5' 'Brodmann area 7'}; %5,7 -- PPC
%cfg.roi = atlas.tissuelabel;
cfg.inputcoord = 'mni';  % coord of the source
%cfg.inputcoord = 'tal';
mask_Brodmann_roi = ft_volumelookup(cfg,srsstd.sourcemodel);  % selecting only a subset
mask_Brodmann_roi_1cm = ft_volumelookup(cfg,srsstd_1cm.sourcemodel);  % selecting only a subset

% Creating string (just for visuatlization)
roistr = '';
for i=1:length(cfg.roi) 
  roistr = [roistr ',' cfg.roi{i}(end-6:end)];
end

% aux
mask_roi_1cm                  = repmat(srsstd_1cm.sourcemodel.inside,1,1);
mask_roi_1cm(mask_roi_1cm==1)     = 0;
mask_roi_1cm(mask_Brodmann_roi_1cm)            = 1;
roipts0_1cm = srsstd_1cm.sourcemodel.pos(mask_roi_1cm,:);     
%roipts0_1cm = srsstd_1cm.sourcemodel.pos(srsstd_1cm.sourcemodel.inside,:);     


mask_roi                  = repmat(srsstd.sourcemodel.inside,1,1);
mask_roi(mask_roi==1)         = 0;
mask_roi(mask_Brodmann_roi)   = 1;

tmpstd = srsstd.sourcemodel.inside;
%hdmf.mni_aligned_grid.inside -- mask of points with 0.5 spacing that are inside the head. 
%It is a volume mask, not surface mask 


roipts0 = hdmf.mni_aligned_grid.pos(mask_roi,:);     
% I could take all grid points but than I will need to make sure 
% that I don't have too many points after I project to the 
% surface. I.e. projection of ball from 3D grid onto 2D surface will make non-uniform grid 
% I could do it assining to each triangle minimum ditance of the intersection and then only taking 
% the point with minimum distance
%roipts0 = hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:);  

do_project_all_roipts = 0;  %just a little bit more accurate but a lot slower
if do_project_all_roipts
  roipts =  projectPtOnBrainSurf(hdmf.hdm, roipts0, 1); 
  if length(roipts) ~= sum(mask_roi)
    fprintf('Non-unique projection happend (some rays intersect more than one triangle)\n')
    exit(0)
  end
else
  roipts = roipts0;
end

%if do_create_surround_pts
nsurrPts = 0;
while nsurrPts < min_number_of_pts
  surroundPts = [];  % repopulate on each cycles run
  point_ind_corresp  = [];  % in final array which points corresp to which Jan point
  nrms = [];
  for i=1:length(coords_Jan_actual)
    ptJan_cur = coords_Jan_actual(i,:);
    nclose = 0;
    % cycle over grid points
    for j=1:length(roipts)
      roi_pt = roipts(j,:);

      % if grid point is close to the given center
      nrm = norm(roi_pt-ptJan_cur);
      nrms = [nrms nrm];
      if nrm < surround_radius
        surroundPts = [surroundPts; roi_pt];
        point_ind_corresp = [point_ind_corresp i];
        nclose = nclose + 1;
      end
    end
    % if we did not find any from the area, add the point itself
    if nclose == 0 && do_create_surround_pts
      
      fprintf('Warning! Failed to find roi pts close to point number %d = ',i)
      fprintf('%.2f',ptJan_cur)
      fprintf('\n')
      surroundPts = [ptJan_cur; surroundPts ];
      point_ind_corresp = [i point_ind_corresp];
    end

  end
  % put original points in the beginning
  surroundPts = [coords_Jan_actual;  surroundPts];
  point_ind_corresp = [1:length(coords_Jan_actual) point_ind_corresp];

  nsurrPts = length(surroundPts);
  surround_radius = surround_radius + radius_inc;
  fprintf('Radius increased to %d\n',surround_radius);
end
fprintf('Num of surroundPts = %d, radius was %d\n',length(point_ind_corresp), surround_radius )
surroundPts =  projectPtOnBrainSurf(hdmf.hdm, surroundPts, 0); 
%else
%  surroundPts =  coords_Jan_actual
%  point_ind_corresp = 1:length(coords_Jan_actual)
%  surround_radius = 0
%end

save( strcat(subjstr,'_modcoord'),  'coords_Jan_actual', 'labels', 'surroundPts', 'point_ind_corresp', 'surround_radius' );

%TODO: so far it is shifted :( I guess I need to apply transform matrix
% I'd need to find a fancier way to convert between resolutions because head 'inside' mesh is provided only for 5mm
%save( strcat(subjstr,'_manycoord'),  'roipts0_1cm' );
%%%%%%%%---------------

%[sorted_pts,sort_inds] = sortrows(surroundPts,1)
 
if exist("skipPlot") & skipPlot
  return;
end

plotSrcCoords

%%%ft_plot_sens(dataica.grad,'style','*r'); % plot the sensor array

%%%%%

% 1st coord >= 0 =>  viewed from the top the right hemisphere get highlited
% 1nd coord <  0 =>  viewed from the top the right hemisphere get highlited

