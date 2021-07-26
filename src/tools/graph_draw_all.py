# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
sys.path.append(os.getcwd())
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import skimage.io as skimage
from skimage import transform as tr
import skimage.morphology as mor
from argparse import ArgumentParser
from datetime import datetime
import pytz
plt.style.use('ggplot')

class GraphDraw():

    def __init__(self, opbase, file_list_born, file_list_abort):
        self.save_dir = save_dir
        # self.scale = 0.8 * 0.8 * 2.0
        self.scale = 1.0 * 1.0 * 1.0
        self.fn_born = file_list_born
        self.fn_abort = file_list_abort
        self.density = 0
        self.roi_pixel_num = 0
        self.label = ['born', 'abort']
        self.color = ['royalblue', 'tomato'] # born, abort
        self.time_max = 360
        self.label_size = 20
        self.figsize = (8, 6)

    def graph_draw_number(self, Time, Count_born, Count_abort):
        # Count
        label = []
        plt.figure(figsize=self.figsize)
        for num in range(len(Count_born)):
            plt.plot(Time[:len(Count_born[num])], Count_born[num], color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(Count_abort)):
            plt.plot(Time[:len(Count_abort[num])], Count_abort[num], color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Number of Nuclei', size=self.label_size)
        # plt.legend(self.label)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'number.pdf'))

        plt.figure(figsize=(10,6))
        plt.plot(1, 1, color=self.color[0], label=self.label[0])
        plt.plot(1, 1, color=self.color[1], label=self.label[1])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=16)
        plt.savefig(os.path.join(self.save_dir, 'legend.pdf'), bbox_inches='tight')

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(Count_born)):
            plt.plot(Time[:len(Count_born[num])], Count_born[num], color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(Count_born[num][:tp_min])
        for num in range(len(Count_abort)):
            plt.plot(Time[:len(Count_abort[num])], Count_abort[num], color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(Count_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Number of Nuclei', size=self.label_size)
        # plt.legend(self.label)
        if Time[-1] != 0:
            #plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.ylim([0, 40])
        plt.savefig(os.path.join(self.save_dir, 'mean_number.pdf'))



    def graph_draw_volume(self, Time, MeanVol_born, StdVol_born, MeanVol_abort, StdVol_abort):
        # Volume Mean & SD
        plt.figure(figsize=self.figsize)
        for num in range(len(MeanVol_born)):
            plt.plot(Time[:len(MeanVol_born[num])], np.array(MeanVol_born[num]) * self.scale, color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanVol_abort)):
            plt.plot(Time[:len(MeanVol_abort[num])], np.array(MeanVol_abort[num]) * self.scale, color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Volume [$\mu m^{3}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'volume_mean.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanVol_born)):
            plt.plot(Time[:len(MeanVol_born[num])], np.array(MeanVol_born[num]) * self.scale, color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(MeanVol_born[num][:tp_min])
        for num in range(len(MeanVol_abort)):
            plt.plot(Time[:len(MeanVol_abort[num])], np.array(MeanVol_abort[num]) * self.scale, color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(MeanVol_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Volume [$\mu m^{3}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_volume_mean.pdf'))


        plt.figure(figsize=self.figsize)
        for num in range(len(StdVol_born)):
            plt.plot(Time[:len(StdVol_born[num])], np.array(StdVol_born[num]) * self.scale, color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(StdVol_abort)):
            plt.plot(Time[:len(StdVol_abort[num])], np.array(StdVol_abort[num]) * self.scale, color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Volume (standard deviation) [$\mu m^{3}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'volume_std.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(StdVol_born)):
            plt.plot(Time[:len(StdVol_born[num])], np.array(StdVol_born[num]) * self.scale, color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(StdVol_born[num][:tp_min])
        for num in range(len(StdVol_abort)):
            plt.plot(Time[:len(StdVol_abort[num])], np.array(StdVol_abort[num]) * self.scale, color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(StdVol_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Volume (standard deviation) [$\mu m^{3}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_volume_std.pdf'))


    def graph_draw_surface(self, Time, MeanArea_born, StdArea_born, MeanArea_abort, StdArea_abort):
        # Surface Mean & SD
        plt.figure(figsize=self.figsize)
        for num in range(len(MeanArea_born)):
            plt.plot(Time[:len(MeanArea_born[num])], np.array(MeanArea_born[num]) * self.scale, color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanArea_abort)):
            plt.plot(Time[:len(MeanArea_abort[num])], np.array(MeanArea_abort[num]) * self.scale, color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Surface Area [$\mu m^{2}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'surface_area_mean.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanArea_born)):
            plt.plot(Time[:len(MeanArea_born[num])], np.array(MeanArea_born[num]) * self.scale, color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(MeanArea_born[num][:tp_min])
        for num in range(len(MeanArea_abort)):
            plt.plot(Time[:len(MeanArea_abort[num])], np.array(MeanArea_abort[num]) * self.scale, color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(MeanArea_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Surface Area [$\mu m^{2}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_surface_area_mean.pdf'))


        plt.figure(figsize=self.figsize)
        for num in range(len(MeanArea_born)):
            plt.plot(Time[:len(StdArea_born[num])], np.array(StdArea_born[num]) * self.scale, color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanArea_abort)):
            plt.plot(Time[:len(StdArea_abort[num])], np.array(StdArea_abort[num]) * self.scale, color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Surface Area (standard deviation) [$\mu m^{2}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'surface_area_std.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanArea_born)):
            plt.plot(Time[:len(StdArea_born[num])], np.array(StdArea_born[num]) * self.scale, color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(StdArea_born[num][:tp_min])
        for num in range(len(MeanArea_abort)):
            plt.plot(Time[:len(StdArea_abort[num])], np.array(StdArea_abort[num]) * self.scale, color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(StdArea_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Surface Area (standard deviation) [$\mu m^{2}$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_surface_area_std.pdf'))


    def graph_draw_aspect_ratio(self, Time, MeanAsp_born, StdAsp_born, MeanAsp_abort, StdAsp_abort):
        # Aspect Ratio Mean & SD
        plt.figure(figsize=self.figsize)
        for num in range(len(MeanAsp_born)):
            plt.plot(Time[:len(MeanAsp_born[num])], np.array(MeanAsp_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanAsp_abort)):
            plt.plot(Time[:len(MeanAsp_abort[num])], np.array(MeanAsp_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Aspect Ratio', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'aspect_ratio_mean.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanAsp_born)):
            plt.plot(Time[:len(MeanAsp_born[num])], np.array(MeanAsp_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(MeanAsp_born[num][:tp_min])
        for num in range(len(MeanAsp_abort)):
            plt.plot(Time[:len(MeanAsp_abort[num])], np.array(MeanAsp_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(MeanAsp_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Aspect Ratio', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_aspect_ratio_mean.pdf'))


        plt.figure(figsize=self.figsize)
        for num in range(len(StdAsp_born)):
            plt.plot(Time[:len(StdAsp_born[num])], np.array(StdAsp_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(StdAsp_abort)):
            plt.plot(Time[:len(StdAsp_abort[num])], np.array(StdAsp_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Aspect Ratio (standard deviation)', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'aspect_ratio_std.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(StdAsp_born)):
            plt.plot(Time[:len(StdAsp_born[num])], np.array(StdAsp_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(StdAsp_born[num][:tp_min])
        for num in range(len(StdAsp_abort)):
            plt.plot(Time[:len(StdAsp_abort[num])], np.array(StdAsp_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(StdAsp_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Aspect Ratio (standard deviation)', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_aspect_ratio_std.pdf'))



    def graph_draw_solidity(self, Time, MeanSol_born, StdSol_born, MeanSol_abort, StdSol_abort):
        # Solidity Mean & SD
        plt.figure(figsize=self.figsize)
        for num in range(len(MeanSol_born)):
            plt.plot(Time[:len(MeanSol_born[num])], np.array(MeanSol_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanSol_abort)):
            plt.plot(Time[:len(MeanSol_abort[num])], np.array(MeanSol_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Solidity', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'solidity_mean.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanSol_born)):
            plt.plot(Time[:len(MeanSol_born[num])], np.array(MeanSol_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(MeanSol_born[num][:tp_min])
        for num in range(len(MeanSol_abort)):
            plt.plot(Time[:len(MeanSol_abort[num])], np.array(MeanSol_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(MeanSol_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Solidity', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_solidity_mean.pdf'))


        plt.figure(figsize=self.figsize)
        for num in range(len(StdSol_born)):
            plt.plot(Time[:len(StdSol_born[num])], np.array(StdSol_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(StdSol_abort)):
            plt.plot(Time[:len(StdSol_abort[num])], np.array(StdSol_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Solidity (standard deviation)', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'solidity_std.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(StdSol_born)):
            plt.plot(Time[:len(StdSol_born[num])], np.array(StdSol_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(StdSol_born[num][:tp_min])
        for num in range(len(StdSol_abort)):
            plt.plot(Time[:len(StdSol_abort[num])], np.array(StdSol_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(StdSol_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Solidity (standard deviation)', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_solidity_std.pdf'))


    def graph_draw_centroid(self, Time, MeanCen_born, StdCen_born, MeanCen_abort, StdCen_abort):
        # Centroid Mean & SD
        plt.figure(figsize=self.figsize)
        for num in range(len(MeanCen_born)):
            plt.plot(Time[:len(MeanCen_born[num])], np.array(MeanCen_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(MeanCen_abort)):
            plt.plot(Time[:len(MeanCen_abort[num])], np.array(MeanCen_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Centroid [$\mu m$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'centroid_mean.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(MeanCen_born)):
            plt.plot(Time[:len(MeanCen_born[num])], np.array(MeanCen_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(MeanCen_born[num][:tp_min])
        for num in range(len(MeanCen_abort)):
            plt.plot(Time[:len(MeanCen_abort[num])], np.array(MeanCen_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(MeanCen_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Centroid [$\mu m$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_centroid_mean.pdf'))


        plt.figure(figsize=self.figsize)
        for num in range(len(StdCen_born)):
            plt.plot(Time[:len(StdCen_born[num])], np.array(StdCen_born[num]), color=self.color[0], alpha=0.8, linewidth=1.0, label=self.label[0])
        for num in range(len(StdCen_abort)):
            plt.plot(Time[:len(StdCen_abort[num])], np.array(StdCen_abort[num]), color=self.color[1], alpha=0.8, linewidth=1.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Centroid (standard deviation) [$\mu m$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'centroid_std.pdf'))

        # mean plot
        tp_min = 488
        plt.figure(figsize=self.figsize)
        born_mean, abort_mean = [], []
        for num in range(len(StdCen_born)):
            plt.plot(Time[:len(StdCen_born[num])], np.array(StdCen_born[num]), color=self.color[0], alpha=0.2, linewidth=1.0, label=self.label[0])
            born_mean.append(StdCen_born[num][:tp_min])
        for num in range(len(StdCen_abort)):
            plt.plot(Time[:len(StdCen_abort[num])], np.array(StdCen_abort[num]), color=self.color[1], alpha=0.2, linewidth=1.0, label=self.label[1])
            abort_mean.append(StdCen_abort[num][:tp_min])
        plt.plot(Time[:tp_min], np.mean(born_mean, axis=0), color=self.color[0], alpha=1.0, linewidth=2.0, label=self.label[0])
        plt.plot(Time[:tp_min], np.mean(abort_mean, axis=0), color=self.color[1], alpha=1.0, linewidth=2.0, label=self.label[1])
        plt.xlabel('Time [day]', size=self.label_size)
        plt.ylabel('Centroid (standard deviation) [$\mu m$]', size=self.label_size)
        if Time[-1] != 0:
            # plt.xlim([0.0, round(Time[-1], 1)])
            plt.xlim([0.0, Time[self.time_max]])
        plt.savefig(os.path.join(self.save_dir, 'mean_centroid_std.pdf'))


if __name__ == '__main__':
    ap = ArgumentParser(description='python graph_draw.py')
    ap.add_argument('--root', '-r', nargs='?', default='/Users/tokkuman/git-tokkuman/embryo_classification/datasets', help='Specify root path')
    ap.add_argument('--save_dir', '-o', nargs='?', default='results/figures_criteria_all', help='Specify output files directory for create figures')
    # ap.add_argument('--label', '-l', nargs='?', default='born', help='Specify label class (born or abort)')
    args = ap.parse_args()
    argvs = sys.argv

    # Make Directory
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = '{0}_{1}'.format(args.save_dir, current_datetime)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(args.root, 'labels', '{}.txt'.format('born')), 'r') as f:
        file_list_born= np.sort([line.rstrip() for line in f])
    with open(os.path.join(args.root, 'labels', '{}.txt'.format('abort')), 'r') as f:
        file_list_abort = np.sort([line.rstrip() for line in f])


    # born
    number_born = []
    volume_mean_born, volume_sd_born = [], []
    surface_mean_born, surface_sd_born = [], []
    aspect_ratio_mean_born, aspect_ratio_sd_born = [], []
    solidity_mean_born, solidity_sd_born = [], []
    centroid_mean_born, centroid_sd_born = [], []
    for fl in file_list_born:
        file_name = os.path.join(args.root, 'input', fl, 'criteria.json')
        print('read: {}'.format(file_name))
        with open(file_name, 'r') as f:
            criteria_value = json.load(f)
        criteria_list = criteria_value.keys()

        if 'number' in criteria_list:
            number_born.append(criteria_value['number'])
        if 'volume_mean' in criteria_list:
            volume_mean_born.append(criteria_value['volume_mean'])
        if 'volume_sd' in criteria_list:
            volume_sd_born.append(criteria_value['volume_sd'])
        if 'surface_mean' in criteria_list:
            surface_mean_born.append(criteria_value['surface_mean'])
        if 'surface_sd' in criteria_list:
            surface_sd_born.append(criteria_value['surface_sd'])
        if 'aspect_ratio_mean' in criteria_list:
            aspect_ratio_mean_born.append(criteria_value['aspect_ratio_mean'])
        if 'aspect_ratio_sd' in criteria_list:
            aspect_ratio_sd_born.append(criteria_value['aspect_ratio_sd'])
        if 'solidity_mean' in criteria_list:
            solidity_mean_born.append(criteria_value['solidity_mean'])
        if 'solidity_sd' in criteria_list:
            solidity_sd_born.append(criteria_value['solidity_sd'])
        if 'centroid_mean' in criteria_list:
            centroid_mean_born.append(criteria_value['centroid_mean'])
        if 'centroid_sd' in criteria_list:
            centroid_sd_born.append(criteria_value['centroid_sd'])

    # abort
    number_abort = []
    volume_mean_abort, volume_sd_abort = [], []
    surface_mean_abort, surface_sd_abort = [], []
    aspect_ratio_mean_abort, aspect_ratio_sd_abort = [], []
    solidity_mean_abort, solidity_sd_abort = [], []
    centroid_mean_abort, centroid_sd_abort = [], []
    for fl in file_list_abort:
        file_name = os.path.join(args.root, 'input', fl, 'criteria.json')
        print('read: {}'.format(file_name))
        with open(file_name, 'r') as f:
            criteria_value = json.load(f)
        criteria_list = criteria_value.keys()

        if 'number' in criteria_list:
            number_abort.append(criteria_value['number'])
        if 'volume_mean' in criteria_list:
            volume_mean_abort.append(criteria_value['volume_mean'])
        if 'volume_sd' in criteria_list:
            volume_sd_abort.append(criteria_value['volume_sd'])
        if 'surface_mean' in criteria_list:
            surface_mean_abort.append(criteria_value['surface_mean'])
        if 'surface_sd' in criteria_list:
            surface_sd_abort.append(criteria_value['surface_sd'])
        if 'aspect_ratio_mean' in criteria_list:
            aspect_ratio_mean_abort.append(criteria_value['aspect_ratio_mean'])
        if 'aspect_ratio_sd' in criteria_list:
            aspect_ratio_sd_abort.append(criteria_value['aspect_ratio_sd'])
        if 'solidity_mean' in criteria_list:
            solidity_mean_abort.append(criteria_value['solidity_mean'])
        if 'solidity_sd' in criteria_list:
            solidity_sd_abort.append(criteria_value['solidity_sd'])
        if 'centroid_mean' in criteria_list:
            centroid_mean_abort.append(criteria_value['centroid_mean'])
        if 'centroid_sd' in criteria_list:
            centroid_sd_abort.append(criteria_value['centroid_sd'])


    # Time Scale
    dt = 10 / float(60 * 24)
    count_max = 0
    for i in range(len(number_born)):
        count_max = np.max([len(number_born[i]), count_max])
    time_point = [dt * x for x in range(count_max)]

    gd = GraphDraw(save_dir, file_list_born, file_list_abort)
    gd.graph_draw_number(time_point, number_born, number_abort)
    gd.graph_draw_volume(time_point, volume_mean_born, volume_sd_born, volume_mean_abort, volume_sd_abort)
    gd.graph_draw_surface(time_point, surface_mean_born, surface_sd_born, surface_mean_abort, surface_sd_abort)
    gd.graph_draw_aspect_ratio(time_point, aspect_ratio_mean_born, aspect_ratio_sd_born, aspect_ratio_mean_abort, aspect_ratio_sd_abort)
    gd.graph_draw_solidity(time_point, solidity_mean_born, solidity_sd_born, solidity_mean_abort, solidity_sd_abort)
    gd.graph_draw_centroid(time_point, centroid_mean_born, centroid_sd_born, centroid_mean_abort, centroid_sd_abort)
