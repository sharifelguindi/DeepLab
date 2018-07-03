# Written by: Sharif Elguindi, MS, DABR
# ==============================================================================
#
# This script returns PNG image files and associated masks for 2D training of images
# using FCN architectures in tensorflow.
#
# Usage:
#
#   python dicomrt_to_traindata.py \
#   --cerr=False
#   --rawdata_dir='path\to\data\'
#   --save_dir='path\to\save\'
#   --structure='structure_name_to_search_for'

from __future__ import print_function
import dicom
from shapely import geometry
import numpy as np
from PIL import Image, ImageDraw
import os, fnmatch
from scipy.misc import imsave, imrotate, toimage
import tensorflow as tf
import h5py
import sys
import glob
import math
import build_data
from functions import *
from image_augmentation import *
import cv2
from transforms import *
from numpy.random import RandomState


PRNG = RandomState()
flags = tf.app.flags
FLAGS = flags.FLAGS

# Default Inputs
# 'H:\\Treatment Planning\\Elguindi\\Segmentation\\CERR IO\\\mat files',
# 'H:\\Treatment Planning\\Elguindi\\Segmentation\\MRCAT_DATA'

flags.DEFINE_boolean('cerr', True,
                     'Set to true to collect data based on .mat CERR files.')

flags.DEFINE_integer('num_shards', 1,
                     'Split train/val data into chucks if large dateset >2-3000 (default, 1)')

flags.DEFINE_string('rawdata_dir', 'H:\\Treatment Planning\\Elguindi\\Segmentation\\CERR IO\\mat files',
                    'absolute path to where raw data is collected from.')

flags.DEFINE_string('save_dir', 'datasets',
                    'absolute path to where processed data is saved.')

flags.DEFINE_string('structure', 'parotids_inv',
                    'string name of structure to export')

flags.DEFINE_string('structure_match', 'Parotid',
                    'string name for structure match')

## This function converts a 3D numpy array image and mask set into .png files for machine learning 2D input
def data_export(data_vol_org, data_seg_org, save_path, p_num, cerrIO, struct_name):

    max_padding = 312
    scale_factor = 256
    strt = int((max_padding - scale_factor) / 2)
    stop = int(strt + scale_factor)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    transform = Compose([
        # [ColorJitter(), None],
        Merge(),
        Expand((0.5, 1.5)),
        RandomCompose([
            RandomResize(0.5, 1.5),
            RandomRotate(30),
            RandomShift(0.2)]),
        Scale(scale_factor),
        ElasticTransform(150),
        RandomCrop(scale_factor),
        HorizontalFlip(),
        Split([0, 3], [3, 6]),
    ],
        PRNG,
        border='constant',
        fillval=0,
        anchor_index=3)

    ## Create folders to store images/masks
    save_path = os.path.join(save_path, struct_name, 'processed')
    if not os.path.exists(os.path.join(save_path,'PNGImages')):
        os.makedirs(os.path.join(save_path,'PNGImages'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClass')):
        os.makedirs(os.path.join(save_path, 'SegmentationClass'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationVis')):
        os.makedirs(os.path.join(save_path, 'SegmentationVis'))

    ## If CERR flag, rotate image 90 degrees (not needed but easy)
    if cerrIO:
        data_vol_org = np.rot90(data_vol_org, axes=(2, 0))
        data_seg = np.rot90(data_seg_org, axes=(2, 0))
        ## For bilateral structure, convert to single class
        data_seg[ data_seg > 1] = 1
    else:
        data_seg = data_seg_org
    ## Verify size of scan data and mask data equivalent
    if data_vol_org.shape == data_seg.shape:

        data_vol = normalize_array(data_vol_org)
        size = data_seg.shape
        data_vol = data_vol.astype('uint8')
        data_vol_org = data_vol_org.astype('uint16')

        rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(data_seg, 0)

        offset_min = np.floor(max_padding / 2)
        offset_max = np.ceil(max_padding / 2)
        mid_r = int(((rmax - rmin) / 2) + rmin)
        mid_c = int(((cmax - cmin) / 2) + cmin)
        rrmin = int(mid_r - offset_min)
        rrmax = int(mid_r + offset_max)
        ccmin = int(mid_c - offset_min)
        ccmax = int(mid_c + offset_max)

        if rrmin < 0:
            rrmin = 0
        if rrmax > np.shape(data_vol_org)[0]:
            rrmax = np.shape(data_vol_org)[0]
        if ccmin < 0:
            ccmin = 0
        if ccmax > np.shape(data_vol_org)[1]:
            ccmax = np.shape(data_vol_org)[1]

        data_vol = data_vol[rrmin:rrmax, ccmin:ccmax, zmin:zmax]
        data_seg = data_seg[rrmin:rrmax, ccmin:ccmax, zmin:zmax]
        scan_shape = np.shape(data_vol)
        # Loop through axial slices, make 3-channel scan, single channel mask
        for i in range(0, scan_shape[2]):
            img = data_vol[:,:,i]
            contour = data_seg[:,:,i]
            if scan_shape[0] < max_padding or scan_shape[1] < max_padding:
                img_ax = np.pad(img, ((int(np.floor((max_padding - scan_shape[0])/2)), int(np.ceil((max_padding - scan_shape[0])/2))),
                                      (int(np.floor((max_padding - scan_shape[1])/2)), int(np.ceil((max_padding - scan_shape[1])/2)))), 'constant', constant_values=0)
                contour_ax = np.pad(contour, ((int(np.floor((max_padding - scan_shape[0])/2)), int(np.ceil((max_padding - scan_shape[0])/2))),
                                      (int(np.floor((max_padding - scan_shape[1])/2)), int(np.ceil((max_padding - scan_shape[1])/2)))), 'constant', constant_values=0)
            elif scan_shape[0] == max_padding and scan_shape[1] == max_padding:
                img_ax = img
                contour_ax = contour

            size_img = img_ax.shape
            stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int8)
            stacked_img_2 = np.zeros((size_img[0], size_img[1], 1), dtype=np.uint8)
            eq_img, stacked_img_1 = equalize(img_ax.astype('uint8'), stacked_img_1, clahe)
            stacked_img_2[:,:,0] = contour_ax
            unique, counts = np.unique(stacked_img_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if 1 in vals:
                img_name = os.path.join('PNGImages','ax' + str(p_num) + '_' + str(i))
                gt_name = os.path.join('SegmentationClass','ax' + str(p_num) + '_' + str(i))
                vis_name = os.path.join('SegmentationVis','ax' + str(p_num) + '_' + str(i))
                # stacked_img_1 = 255 - stacked_img_1
                stacked_img_1[:,:,0] = 255 - stacked_img_1[:,:,0]
                # stacked_img_1[:,:,1] = 255 - stacked_img_1[:,:,1]
                toimage(stacked_img_1[strt:stop,strt:stop,:], cmin=0, cmax=255).save(os.path.join(save_path,img_name + '.png'))
                toimage(stacked_img_2[strt:stop,strt:stop,0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '.png'))

                seg_vis = np.zeros([scale_factor, scale_factor, 1])
                seg_vis[:,:,0] = stacked_img_2[strt:stop,strt:stop,0]
                seg_vis[seg_vis == 1 ] = 200
                seg_vis = np.repeat(seg_vis, 3, axis=2)
                toimage(seg_vis, cmin=0, cmax=255).save(os.path.join(save_path, vis_name + '.png'))

                # zz = 0
                # while zz < 20:
                #     try:
                #         transformed_image, transformed_target = transform(stacked_img_1, stacked_img_2)
                #         transformed_image[np.where((transformed_image == [255, 0, 0]).all(axis=2))] = [255, 255, 255]
                #         transformed_target[transformed_target == 255] = 0
                #         # img_transformed = np.zeros((scale_factor, scale_factor, 3), dtype=np.int8)
                #         # eq_img, img_transformed = equalize(transformed_image[:,:,0], img_transformed, clahe)
                #         # img_transformed = 255 - img_transformed
                #         toimage(transformed_image, cmin=0, cmax=255).save(os.path.join(save_path, img_name + '_' + str(zz) + '.png'))
                #         toimage(transformed_target[:, :, 0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '_' + str(zz) + '.png'))
                #         seg_vis = np.zeros([scale_factor, scale_factor, 1])
                #         seg_vis[:, :, 0] = transformed_target[:, :, 0]
                #         seg_vis[seg_vis == 1] = 200
                #         seg_vis = np.repeat(seg_vis, 3, axis=2)
                #         toimage(seg_vis, cmin=0, cmax=255).save(os.path.join(save_path, vis_name + '_' + str(zz) + '.png'))
                #         zz = zz + 1
                #     except:
                #         "Assertion Error on transformation, trying again"

                # img_aug = [stacked_img_1]
                # seg_aug = [stacked_img_2]
                # save_augmentations(img_aug, seg_aug, save_path, img_name, gt_name)

        # # Loop through sagittal slices, make 3-channel scan, single channel mask, pad image to axial size
        # for i in range(0, scan_shape[1]):
        #     img_sag = data_vol[:,i,:]
        #     contour_sag = data_seg[:,i,:]
        #     stacked_sag_1 = np.zeros((max_padding, max_padding, 3), dtype=np.int16)
        #     stacked_sag_2 = np.zeros((max_padding, max_padding), dtype=np.uint8)
        #
        #     img_sag = cv2.resize(img_sag[:, :], (256, 256))
        #     stacked_sag_2[:, :] = cv2.resize(contour_sag[:, :], (256, 256))
        #
        #     eq_img, stacked_sag_1 = equalize(img_sag, stacked_sag_1, clahe)
        #
        #     unique, counts = np.unique(stacked_sag_2, return_counts=True)
        #     vals = dict(zip(unique, counts))
        #     if 1 in vals:
        #         img_name = os.path.join('PNGImages','sag' + str(p_num) + '_' + str(i) + '.png')
        #         gt_name = os.path.join('SegmentationClass','sag' + str(p_num) + '_' + str(i) + '.png')
        #         imsave(os.path.join(save_path,img_name), stacked_sag_1)
        #         imsave(os.path.join(save_path,gt_name), stacked_sag_2)
        #
        # # Loop through coronal slices, make 3-channel scan, single channel mask, pad image to axial size
        # for i in range(0,scan_shape[0]):
        #     img_cor = data_vol[i,:,:]
        #     contour_cor = data_seg[i,:,:]
        #     stacked_cor_1 = np.zeros((max_padding, max_padding, 3), dtype=np.int16)
        #     stacked_cor_2 = np.zeros((max_padding, max_padding), dtype=np.uint8)
        #
        #     img_cor = cv2.resize(img_cor[:, :], (256, 256))
        #     stacked_sag_2[:, :] = cv2.resize(contour_sag[:, :], (256, 256))
        #
        #     eq_img, stacked_cor_1 = equalize(img_cor, stacked_cor_1, clahe)
        #
        #     unique, counts = np.unique(stacked_cor_2, return_counts=True)
        #     vals = dict(zip(unique, counts))
        #     if 1 in vals:
        #         img_name = os.path.join('PNGImages','cor' + str(p_num) + '_' + str(i) + '.png')
        #         gt_name = os.path.join('SegmentationClass','cor' + str(p_num) + '_' + str(i) + '.png')
        #         imsave(os.path.join(save_path,img_name), stacked_cor_1)
        #         imsave(os.path.join(save_path,gt_name), stacked_cor_2)

    return

def create_tfrecord(structure_path):

    planeList = ['ax', 'cor', 'sag']
    planeDir = ['Axial', 'Coronal', 'Sag']
    filename_train = 'train_'
    filename_val = 'val_'

    i = 0
    for plane in planeList:

        file_base = os.path.join(structure_path, 'processed', 'ImageSets', planeDir[i])
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_train + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        path = os.path.join(structure_path, 'processed', 'PNGImages')
        pattern = plane + '*.png'
        files = find(pattern, path)
        for file in files:
            if file.find(plane) > 0 and (file.find(plane + '1_') < 1 and file.find(plane + '2_') < 1):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_train + plane, k)

        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_val + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        for file in files:
            if file.find(plane) > 0 and (file.find(plane + '1_') > 0 or file.find(plane + '2_') > 0):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_val + plane, k)
        i = i + 1

        dataset_splits = glob.glob(os.path.join(file_base, '*.txt'))
        for dataset_split in dataset_splits:
            _convert_dataset(dataset_split, FLAGS.num_shards, structure_path, plane)

    return

def _convert_dataset(dataset_split, _NUM_SHARDS, structure_path, plane):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  image_folder = os.path.join(structure_path, 'processed', 'PNGImages')
  semantic_segmentation_folder = os.path.join(structure_path, 'processed', 'SegmentationClass')
  image_format = label_format = 'png'

  if not os.path.exists(os.path.join(structure_path, 'tfrecord'+ '_' + plane)):
      os.makedirs(os.path.join(structure_path, 'tfrecord'+ '_' + plane))

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        structure_path, 'tfrecord'+ '_' + plane,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            image_folder, filenames[i] + '.' + image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            semantic_segmentation_folder,
            filenames[i] + '.' + label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, str.encode(filenames[i],'utf-8'), height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unused_argv):
    data_path = FLAGS.rawdata_dir

    if FLAGS.cerr:
        sys.stdout.write('Searching for .mat CERR files')
        p_num = 1
        matFiles = find('*.mat',data_path)
        matFiles.sort()
        if len(os.listdir(data_path)) > 0:
            for filename in matFiles:
                file = h5py.File(filename, 'r')
                scan = getScanArray(file)
                mask = getMaskArray(file)
                sys.stdout.write('\r>> Exporting patient %d of %d' % (
                    p_num, len(os.listdir(data_path))))
                sys.stdout.flush()
                data_export(scan, mask, FLAGS.save_dir, p_num, FLAGS.cerr, FLAGS.structure)
                p_num = p_num + 1
            print('\n')
            print("Update segemntation_dataset.py with new class and values")
            create_tfrecord(os.path.join(FLAGS.save_dir, FLAGS.structure))
        else:
            print("No CERR .mat files found")
            return

    else:

        # create_tfrecord(os.path.join(FLAGS.save_dir, FLAGS.structure))
        p_num = 1
        d_img, d_ss = collect_dicom('*.dcm', data_path)
        ## Start loop through each RS file, if found (p_num = patient number)
        sys.stdout.write('\n')
        if d_ss.keys():
            for rs_file in d_ss.keys():

                scan, mask = load_dicom_toarray(d_ss[rs_file][0], d_img[rs_file], FLAGS.structure_match)
                sys.stdout.write('\r>> Exporting patient {} of {}, file: {} '.format(str(p_num), str(len(d_ss.keys())), d_ss[rs_file][0]))
                sys.stdout.flush()
                if np.count_nonzero(mask):
                    data_export(scan, mask, FLAGS.save_dir, p_num, FLAGS.cerr, FLAGS.structure)
                else:
                    sys.stdout.write('\r>> Structure not found for patient {}'.format(d_ss[rs_file][0]))
                p_num = p_num + 1

            print('\n')
            print("Update segemntation_dataset.py with new class and values")
            create_tfrecord(os.path.join(FLAGS.save_dir, FLAGS.structure))

        else:
            print('Directory specified contains no RS files')

if __name__ == '__main__':
  # flags.mark_flag_as_required('rawdata_dir')
  # flags.mark_flag_as_required('save_dir')
  # flags.mark_flag_as_required('structure')
  tf.app.run()

