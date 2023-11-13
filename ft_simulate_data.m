data_dir = getenv("DATA_DUSS");

subjstr = 'S01';
medstr= 'off';
taskstr='move';
newrawname = sprintf('S99_%s_%s',medstr,taskstr);


load_mat_file = 0;
load_fif_file = 0;
recalc_sim = 0;
recalc_srcrec = 0;
plot_true_dipole_loc = 0;
plot_sensor_timecourse = 1
plot_helmet_distr = 1
use_Heaviside_modulation = 0;
xlims = [6,11]

dipole_moment_type  = 'radial';
dipole_moment_shift = [5 0 0];

dipole_sim_relnoise = 3;   

use_rest_cov = 1
rest_thr = 0.1;
%rest_thr = 1e-5; does not leave much noise


if load_mat_file
  basename_resampled = sprintf('/%s_%s_%s_resampled.mat',subjstr,medstr,taskstr);
  fname_resampled = strcat(data_dir, basename_resampled);
  if ~isfile(fname_resampled)

    basename = sprintf('/%s_%s_%s.mat',subjstr,medstr,taskstr);
    fname_mat = strcat(data_dir, basename );

    datall_mat = load(fname_mat);
    datall_mat = datall_mat.data;

    cfg_resample = [];
    cfg.resamplefs      = 256;
    if ~isfile(fname)
      cfg.outputfile = fname_resampled;
    end
    data_resampled = ft_resampledata(cfg, datall_mat);
  else
    data_resampled = load(fname_resampled);
    data_resampled = data_resampled.data;
  end
end
times = data_resampled.time{1};

if load_fif_file
  basename = sprintf('/%s_%s_%s_resample_afterICA_raw.fif',subjstr,medstr,taskstr);
  fname = strcat(data_dir, basename );

  cfgload = [];
  cfgload.dataset = fname;          % fname should be char array (single quotes), not string (double quotes)
  cfgload.chantype = {'meg'};
  cfgload.coilaccuracy = 1;   % 0 1 or 2, idk what each of them means
  datall_ = ft_preprocessing(cfgload);   
end


sgf = load([data_dir, sprintf('/headmodel_grid_%s_surf.mat',subjstr)] );
subject_surf_grid = sgf.mni_aligned_grid;  % in cm


basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid


fn = strcat( subjstr, '_modcoord_parcel_aal.mat');
sprintf('Loading %s',fn)
ff = load(fn);
srcpos = ff.coords_Jan_actual;

all_coords = ff.coords_Jan_actual;


test_dataset_info = load(strcat(data_dir,sprintf("/%s_supp_info.mat",newrawname) ));
roi_labels_with_tremor = fieldnames(test_dataset_info.roi_sensors);
sensor_inds = cell(length(roi_labels_with_tremor));
for ii = 1:length(roi_labels_with_tremor)
    sensor_inds{ii} = test_dataset_info.roi_sensors.(roi_labels_with_tremor{ii});
end

src_coords = zeros(0,3);
srcinds = [];
for roi_lab_ii=1:length(roi_labels_with_tremor)
  lab_cur = roi_labels_with_tremor{roi_lab_ii};
  roii = find(strcmp(ff.labels, lab_cur ) );
  srcinds_cur = find(ff.point_ind_corresp == roii);
  srcinds = [srcinds;srcinds_cur];
  fprintf('%s %d\n',lab_cur,length(srcinds_cur) );
  %src_coords_cur = all_coords(srcinds_cur,:);
  %src_coords = [src_coords;src_coords_cur];
end
src_coords  = all_coords(srcinds,:);
%src_coords = src_coords(1,:);

[sz1,sz2] = size(src_coords);


if recalc_sim
  dip_dir = [0 1 1];
  sfreq = cast(test_dataset_info.sfreq, 'double');

  cfg_sim = [];
  cfg_sim.headmodel = hdmf.hdm;
  cfg_sim.grad = data_resampled.grad;
  cfg_sim.fsample = sfreq;
  cfg_sim.numtrl = 1;
  %time_duration = 150;

  time_duration = cast(test_dataset_info.total_duration, 'double');
  cfg_sim.trllen = time_duration;
  cfg_sim.dip.pos = src_coords;  % I am not sure it has any influence
  if strcmp(dipole_moment_type,'fixed') 
    cfg_sim.dip.mom = 10 * repmat( dip_dir, sz1, 1)' ;   % the dipole points along the x-axis
  elseif strcmp(dipole_moment_type,'radial')
    cfg_sim.dip.mom = src_coords + dipole_moment_shift;  % this way reconstruction works weirdly --
  end
  %I don't see corresponding frequencies in the result
  times_for_sim = ( 1:(sfreq*time_duration) ) / sfreq;
  lowfreq = 1;
  highfreq = cast(test_dataset_info.trem_burst_meg_freq, 'double');

  bl = 0;
  ampl = 0.3;
  start_time = 8

  modulated_signal = sin(times_for_sim * 2 * pi * highfreq);
  signal_common = modulated_signal;
  %modulation = sin(times_for_sim * 2 * pi * lowfreq) ;
  if use_Heaviside_modulation
    modulation = heaviside(times_for_sim - start_time) ;
  else
    modulation = test_dataset_info.tremor_intensity;
  end
  signal_common = bl + ampl * signal_common .* modulation
  signals = repmat(signal_common, sz1, 1);

  cfg_sim.dip.signal  = {signals};
  %ts = cell(length(src_coords),1);
  ts = {times_for_sim};
  cfg_sim.dip.time = ts;
  %cfg_sim.relnoise = 10;
  %cfg_sim.dip.frequency = 30;
  %cfg_sim.dip.amplitude = 10;
  %   cfg.fsample    = simulated sample frequency (default = 1000)
  %   cfg.trllen     = length of simulated trials in seconds (default = 1)
  %   cfg.numtrl     = number of simulated trials (default = 10)
  %   cfg.baseline   = number (default = 0.3)
  cfg_sim.relnoise = dipole_sim_relnoise;  
  cfg_sim.ntrials = 1;
  data_simulated = ft_dipolesimulation(cfg_sim);
 
  save( [data_dir,sprintf('/%s_MEGsim.mat',newrawname) ], "data_simulated","-v7.3" );

  cfg_tla = [];
  cfg_tla.covariance = 'yes';
  tlout = ft_timelockanalysis(cfg_tla,data_simulated);

  if plot_helmet_distr
    cfg = [];
    %cfg.xlim = [0.3 0.5];
    %cfg.zlim = [0 6e-14];
    cfg.layout = 'neuromag306all.lay';
    cfg.layout = 'neuromag306all_helmet.mat';
    %cfg.parameter = 'individual'; % the default 'avg' is not present in the data
    figure; ft_topoplotER(cfg,tlout); colorbar; 
    aa = sprintf("dip mom dir = %.2f %.2f %.2f",dip_dir(1), dip_dir(2), dip_dir(3));
    title(aa)
  end
   

  if plot_sensor_timecourse
    figure();
    N = length(sensor_inds);
    si = 1

    for i = 1:length(sensor_inds)
        subplot(N,1,si); 
        hold on
        for ii = 1:length(sensor_inds{i})
            plot(times_for_sim, data_simulated.trial{1}(sensor_inds{i}(ii),:));
        end
        xlim(xlims);  title(roi_labels_with_tremor{i});
        hold off
        si = si+1
    end

    subplot(N,1,si); 
    plot(times_for_sim, signal_common);
    xlim(xlims);  title('true data');
  end
end


if recalc_srcrec
  bins_bad = find(modulation > rest_thr );
  bins_artif = zeros(1,2);
  bins_artif(1,:) = [bins_bad(1) length(modulation)];
  cfg_ar = [];
  cfg_ar.artfctdef.imported.artifact = bins_artif;
  cfg_ar.artfctdef.reject          = 'partial';
  data_rest = ft_rejectartifact(cfg_ar,data_simulated);

  cfg_tla = [];
  cfg_tla.covariance = 'yes';
  tlout_rest = ft_timelockanalysis(cfg_tla,data_rest);


  if use_rest_cov
      r = srcrec(subjstr,data_simulated,data_rest,hdmf,"parcel_aal",[],[],[],[],0);
      source_time_tmp_pre = r{1}.source_data;
  else
      tlout_to_use = tlout_rest;
      r = srcrec(subjstr,data_simulated,data_simulated,hdmf,"parcel_aal",[],[],[],[],0);
      source_time_tmp_pre = r{1}.source_data;
      %eigval = eig(tlout_to_use.cov);
      %lambda = max(eigval) * 0.001;
      %
      %
      %cfg_srcrec=[];
      %cfg_srcrec.method='lcmv';
      %cfg_srcrec.headmodel=hdmf.hdm;
      %cfg_srcrec.sourcemodel = [];
      %cfg_srcrec.sourcemodel.pos = all_coords;
      %%cfg_srcrec.resolution = 2;
      %%cfg_srcrec.supchan = bads;
      %
      %%cfg_srcrec.lcmv.lambda='5%';
      %cfg_srcrec.lcmv.lamba = [num2str(lambda),'%']; %regularization value. Can be changed by inspection
      %
      %cfg_srcrec.lcmv.keepfilter = 'yes';
      %cfg_srcrec.lcmv.projectmom='yes';
      %%cfg_srcrec.lcmv.reducerank=2;  % always like that for MEG
      %cfg_srcrec.lcmv.projectnoise = 'yes'; % needed for neural activity index
      %%cfg.resolution = 1;
      %
      %
      %%cfg.resolution = 1;
      %%cfg.method = 'lcmv';
      %
      %source_time_tmp_pre = ft_sourceanalysis(cfg_srcrec,tlout_to_use);  
  end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  fband_cur = [-1 -1];
  resEntry = [];
  resEntry.source_data = source_time_tmp;
  resEntry.bpfreq = fband_cur;
  source_data = {resEntry};

  roicur = 'parcel_aal';
  for fbi = 1:length(resEntry) 
    source_data{fbi}.source_data.roi = {roicur};
  end

  if save_srcrec
    basename_srcd = sprintf("/srcd_%s_%s.mat",newrawname,roicur);
    fname_srcd = strcat(data_dir, basename_srcd );
    fprintf("Saving to %s\n",fname_srcd)
    save(fname_srcd,"source_data","-v7.3");
  end


  % this is needed to get standard FieldTrip format with cell arrays. 
  % But actually my code expects a different one
  [s1, s2] = size(source_time_tmp_pre.avg.mom);
  ss = cell(s1,1);
  for ind=1:s1
      ss{ind} = source_time_tmp_pre.avg.mom(ind,:);
  end
  source_time_tmp = source_time_tmp_pre;
  source_time_tmp.avg.mom = ss;


  cfg = [];
  cfg.powmethod = 'none'; % keep the power as estimated from the data covariance, i.e. the induced power
  %cfg.powmethod = 'pow'; 
  source_nai = ft_sourcedescriptives(cfg, source_time_tmp);
end



if plot_true_dipole_loc
  figVis=1;
  parcel = []; parcel.pos=ff.coords_Jan_actual;
  parcel.masklabel = ff.labels;
  parcel.mask = ff.point_ind_corresp;
  group_labels = roi_labels_with_tremor;
  plot_parcels;
  ft_plot_mesh(data_resampled.grad.chanpos);
  ft_plot_mesh(src_coords,'vertexcolor','blue','vertexsize',20);
end


plot_sourse3D = 0;
if plot_sourse3D
  cfg = [];
  %cfg.method = 'ortho','surface','glassbrain' seem to require MRI anatomy 
  cfg.method = 'vertex';  % or cloud
  %cfg.method = 'cloud';  % or cloud
  %cfg.funparameter = 'nai';
  cfg.funparameter = 'pow';
  %cfg.funcolorlim = [1.4 1.5];  % the voxel in the center of the volume conductor messes up the autoscaling
  ft_sourceplot(cfg, source_nai);
  view([45,45,160]);
end


%neural activity index (NAI), in order to remove the center of the head bias
%shown above. The NAI is the power normalized with an estimate of the spatially
%inhomogeneous noise. An estimate of the noise has been done by
%ft_sourceanalysis, by setting cfg.dics.projectnoise='yes' (default is 'no').
%This noise estimate was computed on the basis of the smallest eigenvalue of the
%cross-spectral density matrix. To calculate the NAI do the following:



%  The dipoles position and orientation have to be specified with
%   cfg.dip.pos     = [Rx Ry Rz] (size Nx3)
%   cfg.dip.mom     = [Qx Qy Qz] (size 3xN)
%
% The number of trials and the time axes of the trials can be specified by
%   cfg.fsample    = simulated sample frequency (default = 1000)
%   cfg.trllen     = length of simulated trials in seconds (default = 1)
%   cfg.numtrl     = number of simulated trials (default = 10)
%   cfg.baseline   = number (default = 0.3)
% or by
%   cfg.time       = cell-array with one time axis per trial, for example obtained from an existing dataset
%
% The timecourse of the dipole activity is given as a cell-array with one
% dipole signal per trial
%   cfg.dip.signal     = cell-array with one dipole signal per trial
% or by specifying the parameters of a sine-wave signal
%   cfg.dip.frequency  =   in Hz
%   cfg.dip.phase      =   in radians
%   cfg.dip.amplitude  =   per dipole
%
% Random white noise can be added to the data in each trial, either by
% specifying an absolute or a relative noise level
%   cfg.relnoise    = add noise with level relative to data signal
%   cfg.absnoise    = add noise with absolute level
%   cfg.randomseed  = 'yes' or a number or vector with the seed value (default = 'yes')
%
% Optional input arguments are
%   cfg.channel    = Nx1 cell-array with selection of channels (default = 'all'),
%                    see FT_CHANNELSELECTION for details
%   cfg.dipoleunit = units for dipole amplitude (default nA*m)
%   cfg.chanunit   = units for the channel data
%
% Optionally, you can modify the leadfields by reducing the rank, i.e. remove the weakest orientation
%   cfg.reducerank    = 'no', or number (default = 3 for EEG, 2 for MEG)
%   cfg.backproject   = 'yes' or 'no',  determines when reducerank is applied whether the 
%                       lower rank leadfield is projected back onto the original linear 
%                       subspace, or not (default = 'yes')




%cfgload = [];
%cfgload.dataset = fname_mat;          % fname should be char array (single quotes), not string (double quotes)
%cfgload.chantype = {'meg'};
%cfgload.coilaccuracy = 1;   % 0 1 or 2, idk what each of them means


% from fif -- grad.units  = 'm'  from mat = 'cm'
% different sizes of coilori and coilpos and tra



% The data should be organised in a structure as obtained from the FT_PREPROCESSING
% function. The configuration should contain
%   cfg.resamplefs      = frequency at which the data will be resampled (default = 256 Hz)
%   cfg.detrend         = 'no' or 'yes', detrend the data prior to resampling (no default specified, see below)
%   cfg.demean          = 'no' or 'yes', whether to apply baseline correction (default = 'no')
%   cfg.baselinewindow  = [begin end] in seconds, the default is the complete trial (default = 'all')
%   cfg.feedback        = 'no', 'text', 'textbar', 'gui' (default = 'text')
%   cfg.trials          = 'all' or a selection given as a 1xN vector (default = 'all')


%
%
%% create an array with some magnetometers at 12cm distance from the origin
%[X, Y, Z] = sphere(10);
%pos = unique([X(:) Y(:) Z(:)], 'rows');
%pos = pos(pos(:,3)>=0,:);
%grad = [];
%grad.coilpos = 12*pos;
%grad.coilori = pos; % in the outward direction
%% grad.tra = eye(length(pos)); % each coils contributes exactly to one channel
%for i=1:length(pos)
%  grad.label{i} = sprintf('chan%03d', i);
%end
%
%% create a spherical volume conductor with 10cm radius
%vol.r = 10;
%vol.o = [0 0 0];
%
%% note that beamformer scanning will be done with a 1cm grid, so you should
%% not put the dipole on a position that will not be covered by a grid
%% location later
%cfg = [];
%cfg.headmodel = vol;
%cfg.grad = grad;
%cfg.dip.pos = [0 0 4];    % you can vary the location, here the dipole is along the z-axis
%cfg.dip.mom = [1 0 0]';   % the dipole points along the x-axis
%cfg.relnoise = 10;
%cfg.ntrials = 20;
%data = ft_dipolesimulation(cfg);
%
%% compute the data covariance matrix, which will capture the activity of
%% the simulated dipole
%cfg = [];
%cfg.covariance = 'yes';
%timelock = ft_timelockanalysis(cfg, data);
%
%% do the beamformer source reconstuction on a 1 cm grid
%cfg = [];
%cfg.headmodel = vol;
%cfg.grad = grad;
%cfg.resolution = 1;
%cfg.method = 'lcmv';
%cfg.lcmv.projectnoise = 'yes'; % needed for neural activity index
%source = ft_sourceanalysis(cfg, timelock);
%
%% compute the neural activity index, i.e. projected power divided by
%% projected noise
%cfg = [];
%cfg.powmethod = 'none'; % keep the power as estimated from the data covariance, i.e. the induced power
%source_nai = ft_sourcedescriptives(cfg, source);
%
%cfg = [];
%cfg.method = 'ortho';
%cfg.funparameter = 'nai';
%cfg.funcolorlim = [1.4 1.5];  % the voxel in the center of the volume conductor messes up the autoscaling
%ft_sourceplot(cfg, source_nai);
%
