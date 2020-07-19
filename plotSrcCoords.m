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
  title('standard head inside, 0.5cm spacing')
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,2)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  ft_plot_mesh(hdmf.mni_aligned_grid.pos(hdmf.mni_aligned_grid.inside,:)); % plot only locations inside the volume
  hold off
  title(sprintf('%s head inside, 0.5cm spacing',subjstr))
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
  %title( sprintf('%s head inside, sclaed %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
  title( sprintf('%s head, surround pts %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
  hold off
  view (viewarr)


  %% Plot Jan's stuff
  subplot(nr,2,6)
  hold on     % plot all objects in one figure
  ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
  %Q = hdmf.mni_aligned_grid.pos(mask_roi,:);
  %Q2 = zeros(size(Q))  ;
  %for i=1:length(Q);
  %  Q2(i,:) = S * transpose( Q(i,:) );
  %end
  Q2 = roipts0_1cm
  ft_plot_mesh(Q2); % plot only locations inside the volume
  %title(sprintf('%s %s, mask of mni_aligned_grid, scaled %3f,%3f,%3f',subjstr,roistr,sx,sy,sz))
  title(sprintf('%s %s, 1cm spacing',subjstr,roistr,sx,sy,sz))
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
  title( sprintf('%s head, Jan pts MNI coords, sclaed %.3f,%.3f,%.3f',subjstr,sx,sy,sz)) 
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
  title(sprintf('%s %s, Jan pts actual coords, scaled %3f,%3f,%3f',subjstr,roistr,sx,sy,sz))
  hold off

  view (viewarr)

  %figname = sprintf('source_pics_%s_%i.fig',subjstr, viewarrind);
  %savefig(figname)

  figname = sprintf('source_pics_%s_%i.png',subjstr,viewarrind);
  saveas(gcf,figname)
  fprintf('Plot saved\n')
end
