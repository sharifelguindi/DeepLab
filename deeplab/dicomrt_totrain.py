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
#   --strcuture='structure_name_to_search_for'

from __future__ import print_function
import dicom
from shapely import geometry
import numpy as np
from PIL import Image, ImageDraw
import os, fnmatch
from scipy.misc import imsave, imrotate
import tensorflow as tf
import h5py
import sys
import glob
import math
# import build_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# Default Inputs
# 'H:\\Treatment Planning\\Elguindi\\Segmentation\\CERR IO\\\mat files',
# 'H:\\Treatment Planning\\Elguindi\\Segmentation\\MRCAT_DATA'

flags.DEFINE_boolean('cerr', True,
                     'Set to true to collect data based on .mat CERR files.')

flags.DEFINE_integer('num_shards', 2,
                     'Split train/val data into chucks if large dateset >2-3000 (default, 1)')

flags.DEFINE_string('rawdata_dir', 'H:\\Treatment Planning\\Elguindi\\Segmentation\\CERR IO\\\mat files',
                    'absolute path to where raw data is collected from.')

flags.DEFINE_string('save_dir', 'datasets',
                    'absolute path to where processed data is saved.')

flags.DEFINE_string('structure', 'parotids',
                    'string name of structure to export')

flags.DEFINE_string('structure_match', 'parotids',
                    'string name for structure match')

def bit_conversion(img, stacked_img_1, LUT, structure):

    if structure == 'parotids':
        LUT_1 = np.clip(LUT, 500, 1000)
        LUT_2 = np.clip(LUT, 750, 1250)
        LUT_3 = np.clip(LUT, 1000, 1500)
        for i in range(0, len(LUT)):
            LUT_1[i] = np.int((255 / 500) * LUT_1[i] - 255)
            LUT_2[i] = np.int((255 / 500) * LUT_2[i] - 382)
            LUT_3[i] = np.int((255 / 500) * LUT_3[i] - 510)

    elif structure == 'bladder':
        LUT_1 = np.clip(LUT, 300, 800)
        LUT_2 = np.clip(LUT, 550, 1050)
        LUT_3 = np.clip(LUT, 800, 1300)
        for i in range(0, len(LUT)):
            LUT_1[i] = np.int((255 / 500) * LUT_1[i] - 153)
            LUT_2[i] = np.int((255 / 500) * LUT_2[i] - 280)
            LUT_3[i] = np.int((255 / 500) * LUT_3[i] - 408)

    img = img.astype(int)
    stacked_img_1[:, :, 0] = LUT_1[img]
    stacked_img_1[:, :, 1] = LUT_2[img]
    stacked_img_1[:, :, 2] = LUT_3[img]

    return stacked_img_1

def bbox2_3D(img, pad):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin - pad, rmax + pad, cmin - pad, cmax + pad, zmin - pad, zmax + pad

def getdataS(file):
    dataS = list(file['dataS'])
    return dataS

def getlabelNamesS(file):
    labelNameList = []
    labelNamesS = list(file['dataS/labelNameC'])
    for index in range(len(labelNamesS)):
        s = 'dataS/labelNameC/' + labelNamesS[index]
        currentName = list(file[s])
        currentName = "".join([chr(item) for item in currentName])
        labelNameList.append((currentName))
    return labelNameList

def getScanArray(file):
    scandata = file.get('dataS/scan3M')
    scandata_as_array = np.array(scandata)
    return scandata_as_array

def getMaskArray(file):
    maskdata = file.get('dataS/labelM')
    maskdata_as_array = np.array(maskdata)
    return maskdata_as_array

def getparamS(file):
    paramList = []
    p = list(file['dataS/paramS/'])
    for index in range(len(p)):
        s = 'dataS/paramS/' + p[index]
        currentParam = list(file[s])
        paramList = np.array(currentParam)
    return paramList

## This function converts a 3D numpy array image and mask set into .png files for machine learning 2D input
def data_export(data_vol, data_seg, save_path, p_num, cerrIO, struct_name):

    max_padding = 256
    ## Create folders to store images/masks
    save_path = os.path.join(save_path, struct_name, 'processed')
    if not os.path.exists(os.path.join(save_path,'PNGImages')):
        os.makedirs(os.path.join(save_path,'PNGImages'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClass')):
        os.makedirs(os.path.join(save_path, 'SegmentationClass'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClassVis')):
        os.makedirs(os.path.join(save_path, 'SegmentationClassVis'))

    ## If CERR flag, rotate image 90 degrees (not needed but easy)
    if cerrIO:
        data_vol = np.rot90(data_vol, axes=(2, 0))
        data_seg = np.rot90(data_seg, axes=(2, 0))
        ## For bilateral structure, convert to single class
        data_seg[ data_seg > 1] = 1

    ## Verify size of scan data and mask data equivalent
    if data_vol.shape == data_seg.shape:

        rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(data_seg, 10)
        data_vol = data_vol[rmin:rmax,cmin:cmax,zmin:zmax]
        data_seg = data_seg[rmin:rmax,cmin:cmax,zmin:zmax]
        size = data_seg.shape

        # Loop through axial slices, make 3-channel scan, single channel mask
        for i in range(0,size[2]):
            img = data_vol[:,:,i]
            contour = data_seg[:,:,i]
            img_ax = np.pad(img, ((int(np.floor((max_padding - size[0])/2)), int(np.ceil((max_padding - size[0])/2))),
                                  (int(np.floor((max_padding - size[1])/2)), int(np.ceil((max_padding - size[1])/2)))), 'constant', constant_values=0)
            contour_ax = np.pad(contour, ((int(np.floor((max_padding - size[0])/2)), int(np.ceil((max_padding - size[0])/2))),
                                  (int(np.floor((max_padding - size[1])/2)), int(np.ceil((max_padding - size[1])/2)))), 'constant', constant_values=255)
            size_img = img_ax.shape
            stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_img_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            if FLAGS.structure == 'parotids':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_img_1 = bit_conversion(img_ax, stacked_img_1, LUT, FLAGS.structure)
            elif FLAGS.structure == 'bladder':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_img_1 = bit_conversion(img_ax, stacked_img_1, LUT, FLAGS.structure)
            else:
                stacked_img_1[:,:,0] = img_ax
                stacked_img_1[:,:,1] = img_ax
                stacked_img_1[:,:,2] = img_ax

            stacked_img_2[:,:] = contour_ax
            unique, counts = np.unique(stacked_img_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if 1 in vals:
                img_name = os.path.join('PNGImages','ax' + str(p_num) + '_' + str(i) + '.png')
                gt_name = os.path.join('SegmentationClass','ax' + str(p_num) + '_' + str(i) + '.png')
                imsave(os.path.join(save_path,img_name), stacked_img_1)
                imsave(os.path.join(save_path,gt_name), stacked_img_2)

        # Loop through sagittal slices, make 3-channel scan, single channel mask, pad image to axial size
        for i in range(0,size[1]):
            img_sag = data_vol[:,i,:]
            contour_sag = data_seg[:,i,:]
            img_sag = np.pad(img_sag,((int(np.floor((max_padding - size[0])/2)), int(np.ceil((max_padding - size[0])/2))),
                                  (int(np.floor((max_padding - size[2])/2)), int(np.ceil((max_padding - size[2])/2)))), 'constant', constant_values=0)
            contour_sag = np.pad(contour_sag, ((int(np.floor((max_padding - size[0])/2)), int(np.ceil((max_padding - size[0])/2))),
                                  (int(np.floor((max_padding - size[2])/2)), int(np.ceil((max_padding - size[2])/2)))), 'constant', constant_values=255)
            size_img = img_sag.shape
            stacked_sag_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_sag_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            if FLAGS.structure == 'parotids':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_sag_1 = bit_conversion(img_sag, stacked_sag_1, LUT, FLAGS.structure)
            elif FLAGS.structure == 'bladder':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_sag_1 = bit_conversion(img_sag, stacked_sag_1, LUT, FLAGS.structure)
            else:
                stacked_sag_1[:,:,0] = img_sag
                stacked_sag_1[:,:,1] = img_sag
                stacked_sag_1[:,:,2] = img_sag

            stacked_sag_2[:,:] = contour_sag
            unique, counts = np.unique(stacked_sag_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if 1 in vals:
                img_name = os.path.join('PNGImages','sag' + str(p_num) + '_' + str(i) + '.png')
                gt_name = os.path.join('SegmentationClass','sag' + str(p_num) + '_' + str(i) + '.png')
                imsave(os.path.join(save_path,img_name), stacked_sag_1)
                imsave(os.path.join(save_path,gt_name), stacked_sag_2)

        # Loop through coronal slices, make 3-channel scan, single channel mask, pad image to axial size
        for i in range(0,size[0]):
            img_cor = data_vol[i,:,:]
            contour_cor = data_seg[i,:,:]
            img_cor = np.pad(img_cor, ((int(np.floor((max_padding - size[1])/2)), int(np.ceil((max_padding - size[1])/2))),
                                  (int(np.floor((max_padding - size[2])/2)), int(np.ceil((max_padding - size[2])/2)))), 'constant', constant_values=0)
            contour_cor = np.pad(contour_cor, ((int(np.floor((max_padding - size[1])/2)), int(np.ceil((max_padding - size[1])/2))),
                                  (int(np.floor((max_padding - size[2])/2)), int(np.ceil((max_padding - size[2])/2)))), 'constant', constant_values=255)
            size_img = img_cor.shape
            stacked_cor_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_cor_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            if FLAGS.structure == 'parotids':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_cor_1 = bit_conversion(img_cor, stacked_cor_1, LUT, FLAGS.structure)
            elif FLAGS.structure == 'bladder':
                LUT = np.arange(np.max(data_vol) - np.min(data_vol) + 1)
                stacked_cor_1 = bit_conversion(img_cor, stacked_cor_1, LUT, FLAGS.structure)
            else:
                stacked_cor_1[:,:,0] = img_cor
                stacked_cor_1[:,:,1] = img_cor
                stacked_cor_1[:,:,2] = img_cor

            stacked_cor_2[:,:] = contour_cor
            unique, counts = np.unique(stacked_cor_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if 1 in vals:
                img_name = os.path.join('PNGImages','cor' + str(p_num) + '_' + str(i) + '.png')
                gt_name = os.path.join('SegmentationClass','cor' + str(p_num) + '_' + str(i) + '.png')
                imsave(os.path.join(save_path,img_name), stacked_cor_1)
                imsave(os.path.join(save_path,gt_name), stacked_cor_2)

    return

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

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
        ## Collect list of structure set files in specified path dir, "RS_Files".  Relies on structure
        ## set labeling prepended with "RS" convention.
        RS_Files = find('RS.*.dcm', data_path)

        ## Start loop through each file (p_num = patient number)
        if RS_Files:
            for p_num in range(0, len(RS_Files)):

                ## Read RS file into dicom class, ss
                ss = dicom.read_file(RS_Files[p_num])

                ## k is numerical counter for structures in file (resets for each RS file)
                k = 0
                ## Start loop through each structure in RS file
                for item in ss.StructureSetROISequence[:]:

                    ## Check if structure is equal to specified structure name
                    if FLAGS.structure_match in item.ROIName:
                         ## ss_maxslice: determines maximum number of image slices contour lives on
                        ss_maxslice = len(ss.ROIContours[k].Contours)

                        ## pattern collects referenced SOP for DICOM collection, searched dir for CT_files list
                        pattern = ss.ROIContours[k].Contours[0].ContourImageSequence[0].ReferencedSOPInstanceUID
                        pattern = '*' + '.'.join(pattern.split('.')[:-2])
                        pattern = pattern[:-3] + '*'
                        CT_files = find(pattern, data_path)
                        try:
                             CT_files.remove(RS_Files[p_num])
                        except:
                             print('RS not found in CT list')
                        if CT_files:

                            ## Open first CT image, get size, total number of files and
                            ## initialize Numpy Arrays for data collection
                            ct_maxslice = len(CT_files)
                            img = dicom.read_file(CT_files[0])
                            img_size = np.shape(img.pixel_array)
                            im_mask = np.zeros((img_size[0], img_size[1], ct_maxslice))
                            im_data = np.zeros((img_size[0], img_size[1], ct_maxslice))
                            z0 = img.ImagePositionPatient[2]

                            ## Since DICOM files are not in spatial order, determine
                            ## "z0" or starting z position
                            for slice in range(0, ct_maxslice):
                                img = dicom.read_file(CT_files[slice])
                                if z0 > img.ImagePositionPatient[2]:
                                    z0 = img.ImagePositionPatient[2]

                            ## Start loop through each CT slice found
                            for slice in range(0, ct_maxslice):

                                ## Read pixel array and image location, convert to numpy reference frame
                                ## and place into im_data as appropriate location
                                img = dicom.read_file(CT_files[slice])
                                z_prime = float(img.ImagePositionPatient[2])
                                zsp = float(img.SliceThickness)
                                z = int((z_prime - z0) / zsp)
                                im_data[:, :, z] = img.pixel_array

                                ## Start for loop through strucutre set point lists
                                ## ss_maxslice is the number of contour objects for structure in question
                                for j in range(0, ss_maxslice):

                                    ## check CT file name against reference UID in contour object
                                    if CT_files[slice].split(os.sep)[-1] == ss.ROIContours[k].Contours[j].ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm' or \
                                       CT_files[slice].split(os.sep)[-1] == 'MR.' + ss.ROIContours[k].Contours[j].ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm':
                                        ## Initialize point list and determin x,y,z pixel spacing, dicom positioning
                                        pointList = []
                                        x_y = np.array(img.ImagePositionPatient)
                                        xsp_ysp = np.array(img.PixelSpacing)
                                        zsp = float(img.SliceThickness)
                                        size = len(ss.ROIContours[k].Contours[j].ContourData)

                                        ## For loop converts point list to numpy reference frame at appropriate
                                        ## z locations
                                        for i in range(0, size, 3):
                                            x_prime = float(ss.ROIContours[k].Contours[j].ContourData[i])
                                            y_prime = float(ss.ROIContours[k].Contours[j].ContourData[i+1])
                                            x = (x_prime - x_y[0])/xsp_ysp[0]
                                            y = (y_prime - x_y[1])/xsp_ysp[1]
                                            p = geometry.Point(x , y)
                                            pointList.append(p)

                                        ## Use Shapely package to convert list of points to image mask
                                        ## at slice z, with pixel inside polygon equal 1, else 0.
                                        poly = geometry.Polygon([[pt.x, pt.y] for pt in pointList])
                                        x, y = poly.exterior.coords.xy
                                        pointsList_new = []
                                        for pp in range(0, len(x)):
                                            pointsList_new.append(x[pp])
                                            pointsList_new.append(y[pp])
                                        m = Image.new('L', (img_size[0], img_size[1]), 0)
                                        ImageDraw.Draw(m).polygon(pointsList_new, outline=1, fill=1)
                                        mask = np.array(m)
                                        z = int(np.round(z))
                                        im_mask[:,:,z] = mask

                        sys.stdout.write('\r>> Exporting patient %d of %d' % (p_num+1, len(RS_Files)))
                        sys.stdout.flush()
                        data_export(im_data, im_mask, FLAGS.save_dir, p_num, FLAGS.cerr, FLAGS.structure)

                    ## Iterate over contour
                    k = k + 1
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