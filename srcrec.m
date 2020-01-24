function output = srcrec(subjstr,datall,hdmf,roi,bads,S,srs,mask)
  %cd 

  do_load_only_ifnew   = 1;
  do_lookup_only_ifnew = 1;
  do_srcrec            = 1;

  %tstart = 300;
  %tend = 400;



  %save(fname_resampled,'datall')

  %cfg_sel = [];
  %cfg_sel.lpfilter = 'yes'
  %cfg_sel.lpfreq = 256
  %cfg_sel.channel = meg_channels_toinclude;
  %cfg_sel.latency = [tstart tend];
  %seldat = ft_preprocessing(cfg_sel,data)

  if 1~= exist("srcpos") || (1==exist("srcpos") && ~do_lookup_only_ifnew)
    %m1 = ft_read_atlas('HMAT_Right_M1.nii')  % goes deeper in the sulcus


    if strcmp(roi{1} , "HirschPt2011,2013direct" ) == 1 || strcmp(roi{1} , "HirschPt2011" ) == 1 

      % places where cortico-muscular coherence (at trem freq) changed during tremor
      %
      %They were located in the primary motor cortex (MNI coordinates:
      % 60, 15, 50), premotor cortex (MNI coordinates:  30, 10, 70)
      %and posterior parietal cortex (MNI coordinates:  20, 75, 50).
      fn = strcat( subjstr, '_modcoord.mat');
      if exist(fn,'file') > 0
        sprintf('Loading %s',fn)
        load(fn);
      else
        sprintf('CoordFile %s does not exist',fn)
        load('coordsJan.mat')

        % compute transformation matrix
        vecinds = [1 200 3000 10000];
        vecinds = [8157       16509          99       22893];
        preX = srs.sourcemodel.pos;   % in each ROW -- x,y,z
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

        %coords_tremCohTremFreq_M1 = [ [ 60, 15, 50]; [-60,16,50] ];
        %coords_tremCohTremFreq_PMC = [ [ 30 10 70]; [ -30 10 70]  ];
        %coords_betaCohMax = [ [33 -22 57] ; [-33 -22 57] ] ;  % spatial maxima where where beta coherence between PreCG (putative M1) and LFPs are largest 

        %coords_Jan_ = [coords_betaCohMax; coords_tremCohTremFreq_M1; coords_tremCohTremFreq_PMC]
        coords_Jan_ = coords;
        coords_Jan_MNI =  transpose( coords_Jan_ ) ;
        %size(coords_Jan_MNI)
        coords_Jan_MNI_t = [ coords_Jan_MNI; ones(1, size(coords_Jan_MNI, 2) )  ];
        yy = M * coords_Jan_MNI_t;
        coords_Jan_actual = transpose( yy(1:3,:)  );
      end


      srcpos = coords_Jan_actual;
    else
      srcpos_ = hdmf.mni_aligned_grid.pos(mask,:);
      srcpos = zeros(size(srcpos_));
      for i=1:length(srcpos)
        srcpos(i,:) = S * transpose( srcpos_(i,:) );
      end
    end
    %hdmf.mni_aligned_grid.inside = tmp;  % this one will be used for source reconstruction
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%  Plotting end
  %cfg                 = [];
  %cfg.channel         = data.label; % ensure that rejected sensors are not present
  %cfg.grad            = data.grad;
  %cfg.headmodel       = hdmf.hdm;
  %cfg.lcmv.reducerank = 2; % default for MEG is 2, for EEG is 3
  %cfg.grid = sourcemodel;
  %[grid] = ft_prepare_leadfield(cfg);


  cfg_srcrec=[];
  cfg_srcrec.method='lcmv';
  cfg_srcrec.lcmv.lambda='5%';
  cfg_srcrec.headmodel=hdmf.hdm;
  %cfg_srcrec.grid=mni_aligned_grid;
  cfg_srcrec.sourcemodel = [];
  cfg_srcrec.sourcemodel.pos = srcpos;
  %cfg_srcrec.grid = [];
  %cfg_srcrec.grid.pos = srcpos;

  cfg_srcrec.supchan = bads;

  cfg_srcrec.lcmv.projectmom='yes';
  cfg_srcrec.lcmv.reducerank=2;

  if do_srcrec
    source_data = ft_sourceanalysis(cfg_srcrec,datall);  % 100 sec, 6 channels takes 37 sec on desktop
    %%source_data = ft_sourceanalysis(cfg_srcrec,output_of_ft_timelockanalysis);
    %
  end


  output = source_data;
  %source_data.inside -- 3D array of  0 or 1
  %source_data.avg.pow -- 3D array of  floats (or NaN s for outside)
  %source_data.avg.mom -- 3D array of  floats (or NaN s for outside)

  % even sabasmpled to 256 Hz with only 100 second duration, source reconstructed weights 4 Gb single subject. Too dense grid
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% do_resample          = 0;
%  if do_resample
%    basename = sprintf('/%s_%s_%s.mat',subjstr,medstr,typestr);
%    fname = strcat(data_dir, basename );
%
%    basename_resampled = sprintf('/%s_%s_%s_resampled.mat',subjstr,medstr,typestr);
%    fname_resampled = strcat(data_dir, basename_resampled );
%    doload = 1;
%    if 1 ~= exist("data") && doload
%      load(fname );
%    end
%
%    MEGonly = 0
%
%
%    cfg1                     = [];
%    %cfg1.dataset             = [ fname];
%    cfg1.lpfreq = 256
%    cfg1.lpfilter = 'yes'
%    cfg1.latency = [tstart tend];
%
%    cfg2 = [];
%    cfg2.toilim = [tstart tend];
%
%    %cfgr.dataset = [ fname ];
%    cfgr            = [];
%    cfgr.resamplefs = 256;
%    %cfgr.baselinewindow = [tstart tend]
%
%    desiredChNum = 6
%
%    if exist('data') %&& ! exist( )
%      labels = data.label;
%      labels = data.label(1:desiredChNum);
%
%      meg_channels_toinclude = [];
%      nch = length(labels);
%      nchbad = length(bads);
%      for indch = 1:nch 
%        isbad = 0;
%        for indbad = 1:nchbad
%          if strcmp( bads(indbad) , labels{indch} )  == 1
%            isbad = 1;
%            break;
%          end
%          if MEGonly && length(strfind(labels{indch}, 'MEG' ) ) == 0
%            isbad = 1;
%            break
%          end
%        end
%        if isbad == 0
%          %meg_channels_toinclude = [ meg_channels_toinclude data.label(indch) ];
%          meg_channels_toinclude = [ meg_channels_toinclude indch ];
%        end
%      end
%    else
%      nch = desiredChNum
%    end
%
%    singlech = cell(nch,1);
%    %parfor indch = 1:nch
%    for indch = 1:nch
%      fprintf('Starting analysis of channel %i of %i',indch, nch)
%      cfg_ = cfg1;
%      cfg_.channel = indch;
%      tmpdat1 = ft_preprocessing(cfg_,data);
%      tmpdat2 = ft_redefinetrial(cfg2,tmpdat1);
%      clear tmpdat1;
%      cfgr.detrend = 'yes'
%      singlech{indch} = ft_resampledata(cfgr,tmpdat2);
%      clear tmpdat2;
%    end
%
%
%    cfgmerge=[];
%    datall = ft_appenddata(cfgmerge,singlech{:});
%    clear singlech
%  else
