do_bandpass = 1;
do_hilb = 0;
if do_hilb && ~do_bandpass
  printf('WRONG')
  exit(1)
end
show_all_src = 1;
show_all_sens=0;
%srcinds_to_show =  srcinds(end-35:end);
%srcinds_to_show =  srcinds(end-10:end);
%srcinds_to_show =  srcinds(1:5);
%srcinds_to_show =  1:length(all_coords);
srcinds_to_show =  srcinds;

if ~show_all_src
    nplots_src = 1;
else
    nplots_src = length(srcinds);
end
if ~show_all_sens
    nplots_sens = 1
else
    nplots_sens = 0
    for roii=1:length(sens_inds)
        nplots_sens = nplots_sens + length(sensor_inds{roii} ) ;
    end
end
N = 4 + nplots_src + nplots_sens;
fig = figure('visible',~show_all_src,'Position', [-200 0 900 100 * N]);
si = 1;

band = [highfreq-3,highfreq+3];
%band = [highfreq-3,highfreq+3];
xlims = [6 11];
%xlims = [6 34];

subplot(N,1,si); si = si + 1;
dat = signal_common;
plot(times_for_sim, dat);
xlim(xlims);  title('true data unfilt');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(N,1,si); si = si + 1;
dat1 = signal_common;
if do_bandpass
  dat1 = ft_preproc_bandpassfilter(dat1, sfreq, band);
end
if do_hilb
    dat1 = ft_preproc_hilbert(dat1);
end
plot(times_for_sim, dat1);
xlim(xlims);  title('true data');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(N,1,si);  si = si + 1;

%band = [0.5 2];

dat1 = source_time_tmp.avg.mom{2 };
if do_bandpass
  dat1 = ft_preproc_bandpassfilter(dat1, sfreq, band);
end
if do_hilb
    dat1 = ft_preproc_hilbert(dat1);
end
plot(times_for_sim, dat1);
xlim(xlims);  title('unrelated source');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(sensor_inds)
    if ~show_all_sens
        subplot(N,1,si); si = si+1
        hold on
    end
    for ii = 1:length(sensor_inds{i})
        if show_all_sens
            subplot(N,1,si); si = si+1; 
        end
        sensi = sensor_inds{i}(ii);
        dat1 = data_simulated.trial{1}(sensi,:);
        if do_bandpass
          dat1 = ft_preproc_bandpassfilter(dat1, sfreq, band);
        end
        if do_hilb
            dat1 = ft_preproc_hilbert(dat1);
        end
        plot(times_for_sim, dat1);
    end
    xlim(xlims);  title(sprintf('%s Sensor %d',roi_labels_with_tremor{i}, sensi ) );
    if ~show_all_sens
        hold off
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~show_all_src
    subplot(N,1,si); si = si + 1;
end
for ind = 1:length(srcinds_to_show)
    if show_all_src
        subplot(N,1,si); si = si + 1;
    end
    srcind = srcinds_to_show(ind);
    dat1 = source_time_tmp.avg.mom{srcind };
    if do_bandpass
      dat1 = ft_preproc_bandpassfilter(dat1, sfreq, band);
    end
    if do_hilb
        dat1 = ft_preproc_hilbert(dat1);
    end
    plot(times_for_sim, dat1); 
    if ~show_all_src
        hold on
    end
    roi_label = ff.labels{ ff.point_ind_corresp(srcind) };
    xlim(xlims);  title(sprintf('%s source %d',roi_label,srcind) );
    %ylim([-2 2]);
end
if ~show_all_src
    hold off
end

if show_all_src
    saveas(fig,sprintf('mlab_out2_mlab%d.png',load_mat_file) );
end
