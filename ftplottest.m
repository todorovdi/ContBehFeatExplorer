%m1 = ft_read_atlas('HMAT_Right_M1.nii')  % goes deeper in the sulcus
afni = ft_read_atlas('~/soft/fieldtrip-20190716/template/atlas/afni/TTatlas+tlrc.HEAD');  
srs = load('~/soft/fieldtrip-20190716/template/sourcemodel/standard_sourcemodel3d5mm');

% pts_converted = mni2icbm_spm( pts )
% atlas.  pts_converted = mni2icbm_spm( pts )
atlas = ft_convert_units(afni,'cm'); % ftrop and our sourcemodels have cm units
srsstd = load('~/soft/fieldtrip-20190716/template/sourcemodel/standard_sourcemodel3d5mm');


data_dir = getenv('DATA_DUSS');
subjstr = 'S10';
basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid


scalemat = containers.Map;
% x -- between ears, z -- vert
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


cfg = [];
cfg.atlas = atlas;
cfg.roi = {'Brodmann area 4'};
cfg.roi = {'Brodmann area 6'};
%cfg.roi = atlas.tissuelabel;
cfg.inputcoord = 'mni';  % coord of the source
%cfg.inputcoord = 'tal';
mask = ft_volumelookup(cfg,srsstd.sourcemodel);  % selecting only a subset

roistr = cfg.roi{1};

tmp                  = repmat(srsstd.sourcemodel.inside,1,1);
tmp(tmp==1)          = 0;
tmp(mask)            = 1;

tmpstd = srsstd.sourcemodel.inside;
tmpsubj = hdmf.mni_aligned_grid.inside;  % this one will be used for source reconstruction

%%%%%%%%---------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot
close all
viewarrs = [ [0 -90 0]; [0 0 90] ];
for i = 1:2 
  figure(i);
  viewarr = viewarrs(i,:);
%viewarr = [0 0 90];

  srsstd.sourcemodel.inside    = tmpstd; % define inside locations according to the atlas based mask 
  hdmf.mni_aligned_grid.inside = tmpstd;  % this one will be used for source reconstruction

  subplot(3,2,1)
  hold on     % plot all objects in one figure
  %%% Plot standard stuff
  hdmstdf = load('~/soft/fieldtrip-20190716/template/headmodel/standard_singleshell.mat');  % just for plotting
  ft_plot_headmodel(hdmstdf.vol,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(srsstd.sourcemodel.pos(srsstd.sourcemodel.inside,:)); % plot only locations inside the volume
  hold off
  title('standard head inside')
  view (viewarr)


  %% Plot Jan's stuff
  subplot(3,2,2)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:)); % plot only locations inside the volume
  hold off
  title(sprintf('%s head inside',subjstr))
  view (viewarr)


  %%%%%%%%%%% set masks
  %srsstd.sourcemodel.inside    = tmp; % define inside locations according to the atlas based mask 
  %hdmf.mni_aligned_grid.inside = tmp;  % this one will be used for source reconstruction

  %%% Plot standard stuff
  subplot(3,2,3)
  hold on     % plot all objects in one figure
  hdmstdf = load('~/soft/fieldtrip-20190716/template/headmodel/standard_singleshell.mat');  % just for plotting
  ft_plot_headmodel(hdmstdf.vol,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = srsstd.sourcemodel.pos(tmp,:);
  ft_plot_mesh(Q); % plot only locations inside the volume
  title(sprintf('standard head %s',roistr))
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(3,2,4)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(hdmf.mni_aligned_grid.pos(tmp,:)); % plot only locations inside the volume
  title(sprintf('%s %s, mask of mni_aligned_grid',subjstr,roistr))
  hold off

  view (viewarr)



  %% Plot Jan's stuff
  subplot(3,2,5)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:);
  Q2 = zeros(size(Q));
  for i=1:length(Q)
    Q2(i,:) = S * transpose( Q(i,:) );
  end
  ft_plot_mesh(Q2); % plot only locations inside the volume
  title( sprintf('%s head inside, sclaed %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(3,2,6)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  Q = hdmf.mni_aligned_grid.pos(tmp,:);
  Q2 = zeros(size(Q))  ;
  for i=1:length(Q);
    Q2(i,:) = S * transpose( Q(i,:) );
  end
  ft_plot_mesh(Q2); % plot only locations inside the volume
  title(sprintf('%s %s, mask of mni_aligned_grid, scaled %3f,%3f,%3f',subjstr,roistr,sx,sy,sz))
  hold off

  view (viewarr)
end
%%%ft_plot_sens(dataica.grad,'style','*r'); % plot the sensor array

%%%%%

% 1st coord >= 0 =>  viewed from the top the right hemisphere get highlited
% 1nd coord <  0 =>  viewed from the top the right hemisphere get highlited

