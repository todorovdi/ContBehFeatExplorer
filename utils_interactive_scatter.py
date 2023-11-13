import os,sys
import mne
import utils  #my code
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import matplotlib as mpl
import utils_tSNE as utsne

from matplotlib.widgets import Button
from matplotlib.text import Annotation

class plots:
    def __init__(self):
        self.colors = []
        self.markers = []
        self.ivalis_tb_indarrays = {}
        self.X_embedded = None
        self.X = None
        self.Xtimes = None
        self.skip_tSNE = None
        self.skip_feat = None
        self.feature_names_all = None
        self.feature_names_all_mod = None

    def emph_interval(self,ax, int_name,int_ind, ls='--',msz=5,msz_emph=22,
                    lines_only=True):
        # prepare highlight of some interval
        emph_inds = self.ivalis_tb_indarrays[int_name][int_ind]

        alpha_noemph = 0.5

        sizes = np.array( [msz] * len(self.colors) )
        sizes[emph_inds] = msz_emph

        colors_rgba = utsne.colNames2Rgba(self.colors)
        colors_rgba = np.vstack(colors_rgba)
        noemph_inds = np.setdiff1d(np.arange(len(self.colors), ), emph_inds)
        colors_rgba[noemph_inds,-1] = alpha_noemph

    #%matplotlib qt
    #%matplotlib inline
    #ax = plt.gca()

        if not lines_only:
            utsne.plotMultiMarker(ax,self.X_embedded[:,0], self.X_embedded[:,1], c = colors_rgba, s=sizes,
                                m=self.markers);
        # connect by line
        ax.plot(self.X_embedded[emph_inds,0], self.X_embedded[emph_inds,1], lw=1, ls=ls)

    #plt.legend(handles = legend_elements)
    def plotEMG(a,b):
        return

    def plotPtInfo(self,axEMG,axFeat,trueInd, show_labels='strong', qstrong=0.95):
        if axEMG is not None:
            self.plotEMG(axEMG,trueInd)

        if axFeat is not None:
            dat = self.X[trueInd * (self.skip_tSNE//self.skip_feat)]

            axFeat.cla()
            axFeat.set_title('Feat time={:.2f}'.format(self.Xtimes[trueInd] ) )
            axFeat.plot(dat)

            dd = np.abs(dat )
            q = np.quantile(dd,qstrong)
            strongInds = np.where( dd  > q ) [0]
            strongestInd = np.argmax(dd)
            axFeat.axhline(y=q, c='purple', ls=':')
            axFeat.axhline(y=-q, c='purple', ls=':')

            feature_names_all = self.feature_names_all
            feature_names_all_mod = self.feature_names_all_mod
            #global gxlabels_were_set
            #if not gxlabels_were_set:
            if show_labels != 'none':
                axFeat.set_xticks(range(len(feature_names_all)))
                if show_labels == 'all':
                    axFeat.set_xticklabels(feature_names_all_mod,rotation=90)
                if show_labels == 'strong':
                    tl = np.array( feature_names_all_mod[:] )
                    inds_ = set(np.arange(len(feature_names_all))) - set(strongInds)
                    inds_ = list(inds_)
                    tl[ inds_ ] = ''
                    axFeat.set_xticklabels(tl,rotation=90)


                tls = axFeat.get_xticklabels()
                if len(tls):
                    for i in strongInds:
                        tls[i].set_color("red")
                    tls[strongestInd].set_color("blue")

            axFeat.set_xlim(0, len(feature_names_all)-1 )
            axFeat.set_ylim(-1.5,1.5)
            axFeat.axhline(y=0, c='r', ls=':')
