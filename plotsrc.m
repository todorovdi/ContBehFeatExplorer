subj='S01';
medcond = "off";
%task = "hold";
task = "move";
roi = "Brodmann area 6";

srcis = [10 24 50];

band_tremor = [3, 7];
band_beta = [15, 30];

Fs = 256;
seglen = 256;
noverlap = 192;
nfft = seglen;
%[s,freq,t] = spectrogram( dd, seglen, noverlap, nfft, Fs,  'yaxis', 'power'); 

nr = 3;
nc = length(srcis);

data_dir = getenv("DATA_DUSS");
basename_srcd = sprintf("/srcd_%s_%s_%s_%s.mat",subj,medcond,task,roi);
fname_srcd = strcat(data_dir, basename_srcd );
%srcf = load(fname_srcd);
times = srcf.source_data.time;

basename_raw = sprintf("/%s_%s_%s.mat",subj,medcond,task);
fname_raw = strcat(data_dir, basename_raw );
rawf = load(fname_raw);

figure(1)
cfg = [];
cfg.channel = 'MEG*';
cfg.viewmode = 'vertical';
cfg.blocksize = 0.9; % in minutes
ft_databrowser(cfg,rawf.data)


%flt = ft_preproc_bandpassfilter(dd,256,band,5,'but','twopass','no',[],'hamming',[],'no','no');
%apply



%%%%%%%%%%%%%%%%%%%%

figure(2)
%for curcol = 1:length(srcis)
%  srci = srcis(curcol);
%  dd_ = srcf.source_data.avg.mom(srci) ;
%  dd = dd_{ 1, :};
%  trempass = bandpass(dd, band_tremor, Fs);
%  betapass = bandpass(dd, band_beta, Fs);
%
%  rowind = 1;
%  ind = nc*(rowind-1) + curcol;
%  subplot(nr,nc,ind)
%  spectype = 'power';
%  spectype = 'psd';
%  spectrogram( dd, seglen, noverlap, nfft, Fs,  'yaxis', spectype); 
%  ylim([2,30])
%  xlim([0,0.9])
%  title(srci)
%  %plot3(t,freq,s)
%
%  rowind = 2;
%  ind = nc*(rowind-1) + curcol;
%  subplot(nr,nc,ind)
%  plot(times,trempass)
%  xlim([0,50])
%
%  rowind = 3;
%  ind = nc*(rowind-1) + curcol;
%  subplot(nr,nc,ind)
%  plot(times,betapass)
%  xlim([0,50])
%
%end

