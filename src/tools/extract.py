#!/usr/bin/env python
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
import pytz
from glob import glob
from datetime import datetime
from skimage import io
from skimage import morphology, measure, transform
from skimage.measure import regionprops
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import configparser

from src.lib.utils.cmd_args import create_dataset_parser, create_runtime_parser


def main(argv=None):
    starttime = time.time()

    ''' ConfigParser '''
    # parsing arguments from command-line or config-file
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    dataset_conf_dict = {}
    runtime_conf_dict = {}
    if args.conf_file is not None:
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        dataset_conf_dict = dict(config.items("Dataset"))
        runtime_conf_dict = dict(config.items("Runtime"))

    # Dataset options
    dataset_parser, dataset_args, remaining_argv = \
        create_dataset_parser(remaining_argv, **dataset_conf_dict)
    # Runtime options
    runtime_parser, runtime_args, remaining_argv = \
        create_runtime_parser(remaining_argv, **runtime_conf_dict)

    # merge options
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of embryo classification',
        parents=[conf_parser, dataset_parser, runtime_parser])
    args = parser.parse_args()

    # Seed
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Resolution
    spatial_resolution = np.array(eval(args.spatial_resolution))
    voxel_convert_scale = spatial_resolution[0] * spatial_resolution[1] * spatial_resolution[2]
    # spatial_resolution /= np.min(spatial_resolution)
    time_resolution = eval(args.time_resolution)

    # Make Directory
    os.makedirs(args.save_dir, exist_ok=True)
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    path_log = os.path.join(args.save_dir, 'log_{}'.format(current_datetime))

    # Selection Criteria
    criteria_list = []
    if eval(args.number):
        criteria_list.append('number')
    if eval(args.volume_sum):
        criteria_list.append('volume_sum')
    if eval(args.volume_mean):
        criteria_list.append('volume_mean')
    if eval(args.volume_sd):
        criteria_list.append('volume_sd')
    if eval(args.surface_sum):
        criteria_list.append('surface_sum')
    if eval(args.surface_mean):
        criteria_list.append('surface_mean')
    if eval(args.surface_sd):
        criteria_list.append('surface_sd')
    if eval(args.aspect_ratio_mean):
        criteria_list.append('aspect_ratio_mean')
    if eval(args.aspect_ratio_sd):
        criteria_list.append('aspect_ratio_sd')
    if eval(args.solidity_mean):
        criteria_list.append('solidity_mean')
    if eval(args.solidity_sd):
        criteria_list.append('solidity_sd')
    if eval(args.centroid_mean):
        criteria_list.append('centroid_mean')
    if eval(args.centroid_sd):
        criteria_list.append('centroid_sd')

    # Make log
    path_directory = np.sort(os.listdir(args.root_path))
    print(path_directory)
    print(criteria_list)
    with open(path_log, 'w') as f:
        f.write('python ' + ' '.join(argv) + '\n\n')
        f.write('target list: {}\n'.format(path_directory))
        f.write('criteria list: {}\n'.format(criteria_list))

    # Parameter of image processing
    kernel = np.ones((3,3,3),np.uint8)

    for pd in path_directory:
        with open(path_log, 'a') as f:
            f.write('\ntarget: {}\n'.format(pd))

        save_dir = os.path.join(args.save_dir, pd)
        os.makedirs(save_dir, exist_ok=True)
        path_images = np.sort(glob(os.path.join(args.root_path, pd, '*.tif')))

        # Each Criteria
        criteria_value = {}
        for c in criteria_list:
            criteria_value[c] = []

        with open(os.path.join(save_dir, 'criteria.csv'), 'w') as f:
            c = csv.writer(f)
            c.writerow(['time_point'] + criteria_list)

        tp = 0
        for pi in path_images:
            tp += 1
            criteria_current = [tp]
            img = io.imread(pi)
            center = np.array(img.shape * spatial_resolution) / 2
            if int(args.labeling) == 4:
                img = morphology.label(img, neighbors=4)
            elif int(args.labeling) == 8:
                img = morphology.label(img, neighbors=8)
            else:
                pass

            # Number
            if eval(args.number):
                criteria_value['number'].append(len(np.unique(img)) - 1)
                criteria_current.append(len(np.unique(img)) - 1)

            # Volume
            volume_list = np.unique(img, return_counts=True)[1][1:] * voxel_convert_scale
            if eval(args.volume_sum):
                criteria_value['volume_sum'].append(np.sum(volume_list))
                criteria_current.append(np.sum(volume_list))
            if eval(args.volume_mean):
                criteria_value['volume_mean'].append(np.mean(volume_list))
                criteria_current.append(np.mean(volume_list))
            if eval(args.volume_sd):
                criteria_value['volume_sd'].append(np.std(volume_list))
                criteria_current.append(np.std(volume_list))

            # Surface Area, Aspect Ratio, Solidity, Centroid Coodinates
            ip_size = np.array(img.shape * spatial_resolution).astype(np.uint8)
            img_area = np.zeros(img.shape)
            centroid, aspect_ratio, solidity = [], [], []
            for n in range(1, len(np.unique(img))):
                img_bin = np.array(img == n) * 1
                img_ero = img_bin - morphology.erosion(img_bin, selem=kernel)
                img_area += img_ero * n

                img_bin = np.array(transform.resize(img_bin * 255, ip_size, order=1, preserve_range=True) > 0) * 1
                p = measure.regionprops(img_bin)[0]
                try:
                    aspect_ratio.append(p.major_axis_length / p.minor_axis_length)
                except:
                    aspect_ratio.append(np.nan)
                try:
                    solidity.append(p.solidity)
                except:
                    solidity.append(np.nan)
                centroid.append(np.sqrt((center[2] - float(p.centroid[2])) ** 2 + (center[1] - float(p.centroid[1])) ** 2 + (center[0] - float(p.centroid[0])) ** 2))

            # Surface Area
            surface_list = np.unique(img_area, return_counts=True)[1][1:] * voxel_convert_scale
            if eval(args.surface_sum):
                criteria_value['surface_sum'].append(np.sum(surface_list))
                criteria_current.append(np.sum(surface_list))
            if eval(args.surface_mean):
                criteria_value['surface_mean'].append(np.mean(surface_list))
                criteria_current.append(np.mean(surface_list))
            if eval(args.surface_sd):
                criteria_value['surface_sd'].append(np.std(surface_list))
                criteria_current.append(np.std(surface_list))

            # Aspect Raio
            if eval(args.aspect_ratio_mean):
                criteria_value['aspect_ratio_mean'].append(np.nanmean(aspect_ratio))
                criteria_current.append(np.nanmean(aspect_ratio))
            if eval(args.aspect_ratio_sd):
                criteria_value['aspect_ratio_sd'].append(np.nanstd(aspect_ratio))
                criteria_current.append(np.nanstd(aspect_ratio))

            # Solidity
            if eval(args.solidity_mean):
                criteria_value['solidity_mean'].append(np.nanmean(solidity))
                criteria_current.append(np.nanmean(solidity))
            if eval(args.solidity_sd):
                criteria_value['solidity_sd'].append(np.nanstd(solidity))
                criteria_current.append(np.nanstd(solidity))

            # Centroid Coodinates
            if eval(args.centroid_mean):
                criteria_value['centroid_mean'].append(np.mean(centroid))
                criteria_current.append(np.mean(centroid))
            if eval(args.centroid_sd):
                criteria_value['centroid_sd'].append(np.std(centroid))
                criteria_current.append(np.std(centroid))


            with open(os.path.join(save_dir, 'criteria.csv'), 'a') as f:
                c = csv.writer(f)
                c.writerow(criteria_current)

            with open(path_log, 'a') as f:
                f.write('tp {0:03d}: {1}\n'.format(tp, criteria_current))

        with open(os.path.join(save_dir, 'criteria.json'), 'w') as f:
            json.dump(criteria_value, f, indent=4)

if __name__ == '__main__':
    main()
