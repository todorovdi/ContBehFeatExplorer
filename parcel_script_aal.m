%% Script is divided in two parts. 1. Parcellation -> 2. Time Domain Source Reconstruction -> 3. Parcel Time Domain Reconstruction

%1. Parcellation: Aims on combining several grid-points / sources to predefined
%anatomical areas given their position in mni space. The assignment of several
%sources (the mask) to a single unit is called a 'parcel'.

%2. Source Reconstruction: Compute spatial filters (a linear mapping between the
%activity in the sensor space and the and the activity in the source space) with the
%information of the sensor data and the headmodel (subject-specific).

%3. Parcel Reconstruction: Several sources time courses are then combined into a
%parcel time course according to their contribution to a parcel as given by 1.

%% 1. Parcellation of template grid according to anatomical atlas


%%% ___ Prepare Parcellation (define areas/groups of interest) ___ %%%
%clear all
%clc

%define path to fieldtrip & functions & raw data
%addpath /data/apps/fieldtrip/latest;
%addpath '/data/project/modo/cbs/analysis/functions' '/data/project/modo/cbs/analysis/scripts';
ft_defaults;

%use Jans cortical surface grid
%load '/data/project/megdbs/Jan/templates/cortical_grid.mat'
load 'cortical_grid.mat'
template_grid.pos = cortical_grid' .* 100; %units are initially in m, now in cm
template_grid.inside = ones(length(template_grid.pos),1) == 1;
template_grid.mask = ones(length(template_grid.pos),1) == 1;
template_grid.coordsys = 'mni';
template_grid.unit = 'cm';
clear cortical_grid

%load atlas
atlas = ft_read_atlas('~/soft/fieldtrip/template/atlas/aal/ROI_MNI_V4.nii');
%atlas = ft_read_atlas('/data/apps/fieldtrip/latest/template/atlas/aal/ROI_MNI_V4.nii');
atlas = ft_convert_units(atlas,'cm');

%remove parcels related to insula, cingulate gyrus, fusiform gyrus, parahippocampas gyrus, amygdala, ... should be deleted, as the structures more medial under the superficial cortex often are linearly dependent on each other.
delete_parcels = {'Rectus','Olfactory','Insula','Cingulum','Hippocampus','ParaHippocampal','Amygdala','Thalamus'...
                  'Pallidum','Putamen','Caudate','Vermis','Fusiform'};
area_labels = atlas.tissuelabel( ~contains(atlas.tissuelabel,delete_parcels) );

%sum parcels having the 'characteristics' mentioned in a cell, together
keywords = {{'Heschl_L','Temporal_Sup_L','Temporal_Pole_Sup_L'},...
            {'Heschl_R','Temporal_Sup_R','Temporal_Pole_Sup_R'},...
            {'Cerebellum_Crus1_R','Cerebellum_Crus2_R','Cerebellum_3_R','Cerebellum_4_5_R','Cerebellum_6_R','Cerebellum_7b_R','Cerebellum_8_R','Cerebellum_9_R','Cerebellum_10_R'},...
            {'Cerebellum_Crus1_L','Cerebellum_Crus2_L','Cerebellum_3_L','Cerebellum_4_5_L','Cerebellum_6_L','Cerebellum_7b_L','Cerebellum_8_L','Cerebellum_9_L','Cerebellum_10_L'}};
groupname = {'Temporal_Sup_L','Temporal_Sup_R',...
             'Cerebellum_R','Cerebellum_L'};
%The updated new area 'groups'
area_labels = for_dimitrii_sumareas(area_labels,keywords);

%group_labels is later used to connect the mask indices with an area 'name' in the parcel structure
group_labels = area_labels;
nested_cells = cellfun(@iscell,area_labels);
group_labels(nested_cells) = groupname;

%cfg in template grid
cfg = [];
cfg.atlas = atlas;
cfg.maskparameter = 'mask';
cfg.minqueryrange = 1;
cfg.maxqueryrange = 7;
cfg.inputcoord = 'mni';

idx_out_of_range = [];
idx_no_areas = [];
idx_multi_areas = [];
pointlabel = cell(length(template_grid.pos),1);

%connect template source with areas
if exist('idx_multi_areas_aal.mat')
  load('idx_multi_areas_aal.mat')
else
  %iterate over surface grid points
  for k = 1:length(template_grid.pos)
      point.inside = template_grid.inside(k);
      point.pos = template_grid.pos(k,:); %The position units must match with the point.unit field
      point.mask = template_grid.mask(k);
      point.coordsys = template_grid.coordsys;
      point.unit = 'cm';
      
      %save labels
      % which parcel is the point related to (can be multiple)
      mask = ft_volumelookup(cfg,point);
      
      if ~isempty( mask.name(mask.count == 1) )
          
          pointlabel{k} = mask.name(mask.count == 1);
          
          %save multiple assignments
          if length( mask.name(mask.count == 1) ) > 1
             idx_multi_areas = [idx_multi_areas,k];
          end
          %save no label found
          if strcmpi( mask.name(mask.count == 1), 'no_label_found')
             idx_no_areas = [idx_no_areas,k];
          end
          
      else
          idx_out_of_range = [idx_out_of_range,k];
      end
  end
  save('idx_multi_areas_aal.mat','idx_multi_areas','pointlabel');
end
clear mask

%from multiple areas simple choose the first assignment
for k = 1:length(idx_multi_areas)
    pointlabel{idx_multi_areas(k)} = pointlabel{idx_multi_areas(k)}(1);
end
%logical indexing empty cells
pointlabel( cellfun(@isempty,pointlabel) ) = {'-'};
%peal off one 'layer' of cell
pointlabel = [pointlabel{:}];

% TODO: make volumelookup for thalamus (Thalamus_L and Thalamus_R)
% attach coordinates of the found points to the cortext grid
% attached repeated thalamus labels to the pointlabel


%% connect pointlabel with indices of area labels (depending on in which cell the pointlabel is located)

%make a mask.  Each point gets index of its label
mask = zeros(length(pointlabel),1);
for k = 1:length(area_labels)
       mask( contains(pointlabel,area_labels{k}) ) = k;
end

%save results in a structure
parcel = [];
parcel.pos = template_grid.pos;
parcel.unit = 'cm';
parcel.coordsys = 'mni';
parcel.mask = mask;                %mask with mask label indices
parcel.masklabel = group_labels';  %area labels


%% Visual Inspectation  (in generic coords)

%figVis = 'on';
figVis = 'off';
if strcmp(figVis, 'on')
  run plot_parcels
end


data_dir = getenv('DATA_DUSS');
%%%%%%%%%% Coord sys conversion
if ~exist("subjstr")
  subjstr = 'S01';
end

sgf = load([data_dir, sprintf('/headmodel_grid_%s_surf.mat',subjstr)] );
subject_surf_grid = sgf.mni_aligned_grid;  % in cm


use_template_grid = 0;
if use_template_grid
  grid_to_use = template_grid;
else
  grid_to_use = subject_surf_grid;
end
%%%%%%%%%%%%%%%%%%% some old ver, make makes sense if I use template_grid instead of subject_surf_grid
M = get_transform_mat(subjstr);
coords_Jan_MNI = transpose( grid_to_use.pos );
coords_Jan_MNI_t = [ coords_Jan_MNI; ones( 1, size(coords_Jan_MNI,2) ) ];
yy = M * coords_Jan_MNI_t ;
coords_Jan_actual = transpose( yy(1:3,:)  );
%%%%%%%%%%%%%%%% 

coords_Jan_actual = grid_to_use.pos;

%pointlabel_check = {}
%pointlabel(mask == k)
% cycle over all points, if pointlabel{k} contains one of the pointlabel_check, then put new el in pointlable_check
% AND put index to the mask_check


check_areas = group_labels;

% zeros would mean that it belongs to areas we don't consider
submask = zeros(length(pointlabel),1);
for k = 1:length(check_areas)
       submask( contains(pointlabel,check_areas{k}) ) = k;
end
%TODO delete points that are from non-selected areas

labels_areas_to_use = check_areas;
labels = labels_areas_to_use;
point_ind_corresp = mask;


surroundPts = [];
surround_radius = 0;

basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid


if ~exist("desired_dist")
  desired_dist=0.5; 
end

if ~exist("project_only_outside_points")
  project_only_outside_points = 1;
end

%%
%project_on_surf = 'no' % 'ray', 'nearest'
project_on_surf = 'nearest';
%project_on_surf = 'no';

coords_Jan_actual_upd = projectPtOnBrainSurf(hdmf.hdm, coords_Jan_actual, project_on_surf, desired_dist, project_only_outside_points, 1);
coords_Jan_actual_old = coords_Jan_actual;
coords_Jan_actual = coords_Jan_actual_upd;

fprintf('Saving\n');
save( strcat(subjstr,'_modcoord_parcel_aal'),  'coords_Jan_actual', 'labels', 'surroundPts', 'point_ind_corresp', 'pointlabel', 'surround_radius', 'project_on_surf', 'template_grid'  );


%return



figure('visible','on');
%headmodel = sgf.hdm
headmodel = hdmf.hdm;
ft_plot_headmodel(headmodel,  'facecolor', 'cortex', 'edgecolor', 'none');alpha 0.5; camlight;
ft_plot_mesh(coords_Jan_actual);


return
%% 2. Time Domain Source Reconstruction [LCMV Beamformer]

clearvars -except parcel template_grid

%subject names
subjects =  {'cbs02','cbs02_002','cbs03'};

%Initiate subject loop to perform source reconstruction per subject
for i = 1:length(subjects)
    
    
    %%% ___ Prepare necessary variables for the source reconstruction computation ___ %%%
    
    %load headmodel & grid (subject specific)
    load(['/data/project/modo/cbs/analysis/intermediate_data/cbs/headmodels/',subjects{i},'/',subjects{i},'_forward_model.mat']);
    
    %load the cleaned and segmented subject data (one second dummy trials in time domain)
    load(['/data/project/modo/cbs/analysis/intermediate_data/cbs/clean_data/',subjects{i},'_rest_time.mat']);
    data = data_select_seg_clean;
    
    %%% ___ Compute Spatial Filters to translate between Sensor Space and Source Space ___ %%%
    
    %change time-field to be the same for all trials to calculate the covariance matrix
    data.time(:) = data.time(1);
    
    %we need the covariance matrix for the calculation of spatial filters
    cfg = [];
    cfg.covariance = 'yes';
    avg = ft_timelockanalysis(cfg,data);
    
    %compute eigendecomposition to get eigenvalues for determining lambda as 1/1000 * largest eigenvalue (see https://mailman.science.ru.nl/pipermail/fieldtrip/2012-March/030749.html)
    eigval = eig(avg.cov);
    lambda = max(eigval) * 0.001;
        
    cfg = [];
    cfg.method = 'lcmv';                    %use lcmv beamforming
    cfg.lcmv.lamba = [num2str(lambda),'%']; %regularization value. Can be changed by inspection
    cfg.headmodel = hdm;                    %subject specific headmodel
    cfg.sourcemodel = grid;                 %subject specific grid
    cfg.normalize = 'no';                   %can also be set to 'no'. Whatever works best. 'yes' normalizes the leadfield (scales the spatial filter at a grid point)
    cfg.lcmv.projectmom = 'yes';            %projects time-courses of xyz directions onto a single time-course
    cfg.lcmv.keepfilter = 'yes';            %Keep the spatial filters 'yes'. These are used to project each trial individually from sensor space to source space
    source_time_tmp = ft_sourceanalysis(cfg,avg);
    %computation of virtual channels is done like here (svd happens implicitely: old.fieldtriptoolbox.org/tutorial/connectivity)
    
    %The spatial filter for subject{i}, dimensionality: Sources X Channels
    spatial_filt = cell2mat(source_time_tmp.avg.filter);
    
    
    %%% ___ Apply spatial filter on trials to get trial activity in source space ___ %%%
    
    %preallocate variable that contains the source activity per trial
    trial_source = zeros( size(spatial_filt,1), data_select_seg_clean.fsample, length(data_select_seg_clean.trial) ); %Sources X Samples X Trials
    
    %get the data [with all trials] projected into source space with the individual spatial filter
    for j = 1:length(data_select_seg_clean.trial)
        trial_source(:,:,j) = spatial_filt * data_select_seg_clean.trial{j}; %Sources X Timepoint X Trails
    end
    
    %line trials up as a continuous (the parcel calculation is much faster then. Later I go back to trials)
    trial_source = reshape( trial_source, [size(trial_source,1),size(trial_source,2) * size(trial_source,3)] );
    
    %save results in a structure
    source_time.pos = parcel.pos;           %we need to go back to the template_grid and its .pos, so we use this here. source_time_tmp.pos WOULD BE the the subject-specific grid (so what is in leadfield)
    source_time.avg.mom = trial_source;     % #sources X [sampling_rate*trials]
    source_time.unit = 'cm';    
    
    %% 3. Compute Parcel Time Courses
    
    cfg = [];
    cfg.method = 'eig';                     %or median/mean/...
    cfg.parcellation = 'mask';              %fieldname with wished parcellation as determined by the mask indices
    cfg.parameter = 'mom';                  %fieldname with data that should be parcellated
    parcel_time = ft_sourceparcellate(cfg,source_time,parcel);
    %check for rank deficiency of parcels
    if rank(parcel_time.mom) ~= size(parcel_time.mom,1); error('Rank deficiency: Parcel time courses are linearly dependent.'); end
    
    %leakage reduction [https://ohba-analysis.github.io/osl-docs/matlab/osl_example_roinets_1_synthetic.html]
    parcel_time.mom = ROInets.remove_source_leakage(parcel_time.mom,'closest');
    parcel_time.mom = reshape( parcel_time.mom, [size(parcel_time.mom,1), size(data_select_seg_clean.trial{1},2), numel(data_select_seg_clean.trial)]);
    
    %go back to trial-format (I use the knowledge, that we have 1 second trial length)
    parcel_time.trial = squeeze( mat2cell( parcel_time.mom, size(parcel_time.mom,1), size(parcel_time.mom,2), ones(1,size(parcel_time.mom,3)) ) )';
    parcel_time.time = data.time; %also add back the time field of a trial
    
    %remove (now) redundant fields
    parcel_time = rmfield(parcel_time,'mom');
    parcel_time = rmfield(parcel_time,'momdimord');
    
    %save subject parcel time-courses    
    save(['/data/project/modo/cbs/analysis/intermediate_data/cbs/parcel/',subjects{i},'/',subjects{i},'_parcel_time.mat'],'parcel_time');
    
end
