cfg = []                               ;
cfg.artfctdef.eog.bpfilter   = 'yes'   ;
cfg.artfctdef.eog.bpfilttype = 'but'   ;
cfg.artfctdef.eog.bpfreq     = [1 15]  ;
%cfg.artfctdef.eog.bpfiltord  = 4      ;
cfg.artfctdef.eog.bpfiltor  = 3        ;
cfg.artfctdef.eog.hilbert    = 'yes'   ;

cfg.artfctdef.eog.channel      = 'eog'; 
cfg.artfctdef.eog.cutoff       = 4; %z-value at which to threshold (default = 4)
cfg.artfctdef.eog.trlpadding   = 0;
cfg.artfctdef.eog.fltpadding   = 0;
cfg.artfctdef.eog.artpadding   = 0.1;

cfg.artfctdef.eog.interactive   = 'yes';

[cfg, artifact]  = ft_artifact_eog(cfg,data);

%{'MEG2641','MEG2642'} Nx1 cell-array with selection of channels, see FT_CHANNELSELECTION for details






delimiterIn   = ',';
headerlinesIn = 3;
A = importdata(filename,delimiterIn,headerlinesIn);
