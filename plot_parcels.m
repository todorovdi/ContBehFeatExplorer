%All Areas
check_areas = group_labels;% Or select group labels e.g.: {'Cerebellum_R','Postcentral_L'};
%check_areas = group_labels(1:10);% Or select group labels e.g.: {'Cerebellum_R','Postcentral_L'};
%check_areas = group_labels(20:30);% Or select group labels e.g.: {'Cerebellum_R','Postcentral_L'};
%check_areas = group_labels(40:60);% Or select group labels e.g.: {'Cerebellum_R','Postcentral_L'};
%check_areas = group_labels(60:80);% Or select group labels e.g.: {'Cerebellum_R','Postcentral_L'};
%precentral, suppm, cerebellum , parietal, supramarginal, paracentral, temporal

%seltype = 'inv'         % select undesired from check_areas 
seltype = 'direct'       % select desired from check_areas
%seltype = 'simple'      % just show check_areas
%presel_labels = group_labels(1:20);
%presel_labels = group_labels(20:40);
%presel_labels = group_labels(40:60);
presel_labels = group_labels;
desired = { 'Precentral', 'Supp', 'Cerebellum', 'Parietal', 'Supramarginal', 'Paracentral', 'Temporal_lob', 'Temporal_med', 'Rolandic', 'Postcentral', 'Precuneus', 'Supp_Motor_Area'};
% desired = ["Cerebellum"];
% maybe also "angular"

sel_areas = {};
ctr = 1;
for j = 1:length(presel_labels)
  for jj = 1:length(desired)
    if contains(presel_labels(j), desired(jj) , 'IgnoreCase', 1   )
      sel_areas{ctr} = presel_labels{j};
      ctr = ctr + 1;
      break
    end
  end
end

if strcmp(seltype,'direct')
  check_areas = sel_areas;
elseif strcmp(seltype,'inv')
  check_areas = setdiff(presel_labels,sel_areas);
else
  check_areas = presel_labels;
end


figure('visible',figVis);

Num_elem = numel(check_areas)
%prepare colors
Col = colormap('hsv'); close;
Col = Col(round(linspace(1,size(Col,1),Num_elem)), :);

%Load a surface to make a nicer picture
%load('~/soft/fieldtrip/template/anatomy/surface_white_both.mat')
%mesh = ft_convert_units(mesh,'cm');
%ft_plot_mesh(mesh,'facealpha',0.1);
ft_plot_headmodel(hdmf.hdm,  'facecolor', 'cortex', 'edgecolor', 'none'); alpha 0.15; camlight;
view([90,0]);
hold on

%if ~exist('parcel')
%end

h = [];
for j = 1:Num_elem
    %indices of areas corresponding to the label
    relevant_areas = find(strcmpi(parcel.masklabel,check_areas{j}));
    %make a mask for a grand region specified in check_areas [sum of related area masks]
    paint = ismember(parcel.mask,relevant_areas);
    
    %plot area
    region = parcel.pos(paint == 1,:); % 10 .*  to transfer from cm to mm ... but we already used ft_convert_units()
    hi = plot3(region(:,1),region(:,2),region(:,3),'o','MarkerFaceColor',Col(j,:),'MarkerEdgeColor',Col(j,:),'MarkerSize',4);
    h(j) = hi(1);

    hold on
end
%legend(h, check_areas{j}, 'TextColor', Col(j,:) )
legend(h, string(check_areas)  );
