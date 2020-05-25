%m1 = ft_read_atlas('HMAT_Right_M1.nii')  % goes deeper in the sulcus
afni = ft_read_atlas('~/soft/fieldtrip-20190716/template/atlas/afni/TTatlas+tlrc.HEAD');  
srs = load('~/soft/fieldtrip-20190716/template/sourcemodel/standard_sourcemodel3d5mm');

% pts_converted = mni2icbm_spm( pts )
% atlas.  pts_converted = mni2icbm_spm( pts )
atlas = ft_convert_units(afni,'cm'); % ftrop and our sourcemodels have cm units
srsstd = load('~/soft/fieldtrip-20190716/template/sourcemodel/standard_sourcemodel3d5mm');


data_dir = getenv('DATA_DUSS');
%subjstr = 'S10';

if ~exist("subjstr")
  subjstr = 'S02';
end
fprintf('Preparing source coordinates for subject %s',subjstr)

basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

figVis = 'off'
%figVis = 'on'

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

save("head_scalemat",'scalemat')

S = scalemat(subjstr);


% get matrix mapping MNI to my coords
vecinds = [1 2 3];
vecinds = [2 3 4];
vecinds = [1 200 3000 10000];
vecinds = randi( [1,  length( hdmf.mni_aligned_grid.pos ) ], 1, 4 );
vecinds = [8157       16509          99       22893];
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


cfg = [];
cfg.atlas = atlas;
%cfg.roi = {'Brodmann area 4'};
cfg.roi = {'Brodmann area 6'};
cfg.roi = {'Brodmann area 6' 'Brodmann area 4'};
%cfg.roi = atlas.tissuelabel;
cfg.inputcoord = 'mni';  % coord of the source
%cfg.inputcoord = 'tal';
mask = ft_volumelookup(cfg,srsstd.sourcemodel);  % selecting only a subset

roistr = cfg.roi{1};

mask_roi                  = repmat(srsstd.sourcemodel.inside,1,1);
mask_roi(mask_roi==1)          = 0;
mask_roi(mask)            = 1;

tmpstd = srsstd.sourcemodel.inside;
tmpsubj = hdmf.mni_aligned_grid.inside;  % this one will be used for source reconstruction


surround_radius = 0.8;
radius_inc = 0.1;
min_number_of_pts = 80

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

nsurrPts = 0;
while nsurrPts < min_number_of_pts
  surroundPts = [];
  point_ind_corresp  = [];  % in final array which points corresp to which Jan point
  nrms = [];
  for i=1:length(coords_Jan_actual)
    ptJan_cur = coords_Jan_actual(i,:);
    nclose = 0;
    for j=1:length(roipts)
      roi_pt = roipts(j,:);

      nrm = norm(roi_pt-ptJan_cur);
      nrms = [nrms nrm];
      if nrm < surround_radius
        surroundPts = [surroundPts; roi_pt];
        point_ind_corresp = [point_ind_corresp i];
        nclose = nclose + 1;
      end
    end
    % if we did not find any from the area, add the point itself
    if nclose == 0
      
      fprintf('Warning! Failed to find roi pts close to point number %d = ',i)
      fprintf('%d',ptJan_cur)
      fprintf('\n')
      surroundPts = [surroundPts; ptJan_cur];
      point_ind_corresp = [point_ind_corresp i];
    end

  end
  nsurrPts = length(surroundPts);
  surround_radius = surround_radius + radius_inc;
  fprintf('Radius increased to %d\n',surround_radius);
end
surroundPts =  projectPtOnBrainSurf(hdmf.hdm, surroundPts, 0); 
fprintf('Num of surroundPts = %d, radius was %d\n',length(point_ind_corresp), surround_radius )

save( strcat(subjstr,'_modcoord'),  'coords_Jan_actual', 'labels', 'surroundPts', 'point_ind_corresp', 'surround_radius' );
%%%%%%%%---------------

%[sorted_pts,sort_inds] = sortrows(surroundPts,1)
 
if exist("skipPlot") & skipPlot
  return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot
close all
viewarrs = [ [0 -90 0]; [0 0 90] ];
nr = 4;

figure('visible',figVis);
for viewarrind = 1:2 
  %figure(i,'visible','off');
  viewarr = viewarrs(viewarrind,:);
%viewarr = [0 0 90];

  srsstd.sourcemodel.inside    = tmpstd; % define inside locations according to the atlas based mask 
  hdmf.mni_aligned_grid.inside = tmpstd;  % this one will be used for source reconstruction

  subplot(nr,2,1)
  hold on     % plot all objects in one figure
  %%% Plot standard stuff
  hdmstdf = load('~/soft/fieldtrip-20190716/template/headmodel/standard_singleshell.mat');  % just for plotting
  ft_plot_headmodel(hdmstdf.vol,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(srsstd.sourcemodel.pos(srsstd.sourcemodel.inside,:)); % plot only locations inside the volume
  hold off
  title('standard head inside')
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,2)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:)); % plot only locations inside the volume
  hold off
  title(sprintf('%s head inside',subjstr))
  view (viewarr)


  %%%%%%%%%%% set masks
  %srsstd.sourcemodel.inside    = mask_roi; % define inside locations according to the atlas based mask 
  %hdmf.mni_aligned_grid.inside = mask_roi;  % this one will be used for source reconstruction

  %%% Plot standard stuff
  subplot(nr,2,3)
  hold on     % plot all objects in one figure
  hdmstdf = load('~/soft/fieldtrip-20190716/template/headmodel/standard_singleshell.mat');  % just for plotting
  ft_plot_headmodel(hdmstdf.vol,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = srsstd.sourcemodel.pos(mask_roi,:);
  ft_plot_mesh(Q); % plot only locations inside the volume
  title(sprintf('standard head %s',roistr))
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,4)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(hdmf.mni_aligned_grid.pos(mask_roi,:)); % plot only locations inside the volume
  title(sprintf('%s %s, mask of mni_aligned_grid',subjstr,roistr))
  hold off

  view (viewarr)



  %% Plot Jan's stuff
  subplot(nr,2,5)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:);

  Q = surroundPts;
  Q2 = zeros(size(Q));
  for i=1:length(Q)
    Q2(i,:) = S * transpose( Q(i,:) );
  end
  ft_plot_mesh(Q2); % plot only locations inside the volume
  title( sprintf('%s head inside, sclaed %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,6)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = hdmf.mni_aligned_grid.pos(mask_roi,:);
  Q2 = zeros(size(Q))  ;
  for i=1:length(Q);
    Q2(i,:) = S * transpose( Q(i,:) );
  end
  ft_plot_mesh(Q2); % plot only locations inside the volume
  title(sprintf('%s %s, mask of mni_aligned_grid, scaled %3f,%3f,%3f',subjstr,roistr,sx,sy,sz))
  hold off

  view (viewarr)


  %% Plot Jan's pts
  subplot(nr,2,7)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = coords_Jan_MNI;
  Q2 = zeros(size(Q));
  for i=1:size(Q,2)
    Q2(:,i) = Q(:,i) ;
  end
  ft_plot_mesh( transpose(Q2) ); % plot only locations inside the volume
  title( sprintf('%s standard head inside, sclaed %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,8)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = coords_Jan_actual;
  Q2 = zeros(size(Q))  ;
  for i=1:size(Q,1);
    Q2(i,:) = S *  transpose( Q(i,:)  );
  end

  Q2(1,1) = 2 * Q2(1,1); % just to visually aid distinguishing hemispheres
  Q2(1,1)

  ft_plot_mesh( Q2 ); % plot only locations inside the volume
  title(sprintf('%s %s, actual coords, scaled %3f,%3f,%3f',subjstr,roistr,sx,sy,sz))
  hold off

  view (viewarr)

  %figname = sprintf('source_pics_%s_%i.fig',subjstr, viewarrind);
  %savefig(figname)

  figname = sprintf('source_pics_%s_%i.png',subjstr,viewarrind);
  saveas(gcf,figname)
  fprintf('Plot saved\n')
end


%%%ft_plot_sens(dataica.grad,'style','*r'); % plot the sensor array

%%%%%

% 1st coord >= 0 =>  viewed from the top the right hemisphere get highlited
% 1nd coord <  0 =>  viewed from the top the right hemisphere get highlited

