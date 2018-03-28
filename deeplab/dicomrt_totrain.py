# Written by: Sharif Elguindi, MS, DABR
# ==============================================================================
#
# This script returns PNG image files and associated masks for 2D training of images
# using FCN architecture
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

flags = tf.app.flags
FLAGS = flags.FLAGS

# Default Inputs

flags.DEFINE_boolean('cerr', False,
                     'Set to true to collect data based on .mat CERR files.')

flags.DEFINE_string('rawdata_dir', r'U:\DATA\datasets\raw_datasets',
                    'absolute path to where raw data is collected from.')

flags.DEFINE_string('save_dir', 'datasets',
                    'absolute path to where processed data is saved.')

flags.DEFINE_string('structure', 'bladder',
                    'string name of structure to export')

flags.DEFINE_string('structure_match', 'Bladder_O',
                    'string name of structure to export')

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

    ## Create folders to store images/masks
    save_path = os.path.join(save_path, struct_name, 'processed')
    if not os.path.exists(os.path.join(save_path,'PNGImages')):
        os.makedirs(os.path.join(save_path,'PNGImages'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClass')):
        os.makedirs(os.path.join(save_path, 'SegmentationClass'))

    ## If CERR flag, rotate image 90 degrees (not needed but easy)
    if cerrIO:
        data_vol = np.rot90(data_vol, axes=(2, 0))
        data_seg = np.rot90(data_seg, axes=(2, 0))

    size = data_vol.shape
    size_msk = data_seg.shape

    if size == size_msk:
        # Axial Slices
        for i in range(0,size[2]):
            img = data_vol[:,:,i]
            contour = data_seg[:,:,i]
            size_img = img.shape
            stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_img_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            stacked_img_1[:,:,0] = img
            stacked_img_1[:,:,1] = img
            stacked_img_1[:,:,2] = img

            stacked_img_2[:,:] = contour
            unique, counts = numpy.unique(stacked_img_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if(vals[1]) > 0):
                img_name = os.path.join('PNGImages','ax' + str(p_num) + '_' + str(i) + '.png')
                gt_name = os.path.join('SegmentationClass','ax' + str(p_num) + '_' + str(i) + '.png')
                imsave(os.path.join(save_path,img_name), stacked_img_1)
                imsave(os.path.join(save_path,gt_name), stacked_img_2)

        # Sagittal
        for i in range(0,size[1]):
            img_sag = data_vol[:,i,:]
            contour_sag = data_seg[:,i,:]
            img_sag = np.pad(img_sag, ((0, 0), (0, size[1] - size[2])), 'constant', constant_values=255)
            contour_sag = np.pad(contour_sag, ((0, 0), (0, size[1] - size[2])), 'constant', constant_values=255)
            size_img = img_sag.shape
            stacked_sag_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_sag_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            stacked_sag_1[:,:,0] = img_sag
            stacked_sag_1[:,:,1] = img_sag
            stacked_sag_1[:,:,2] = img_sag

            stacked_sag_2[:,:] = contour_sag
            unique, counts = numpy.unique(stacked_sag_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if(vals[1]) > 0):
                img_name = os.path.join('PNGImages','sag' + str(p_num) + '_' + str(i) + '.png')
                gt_name = os.path.join('SegmentationClass','sag' + str(p_num) + '_' + str(i) + '.png')
                imsave(os.path.join(save_path,img_name), stacked_sag_1)
                imsave(os.path.join(save_path,gt_name), stacked_sag_2)

        # Coronal
        for i in range(0,size[0]):
            img_cor = data_vol[i,:,:]
            contour_cor = data_seg[i,:,:]
            img_cor = np.pad(img_cor, ((0, 0), (0, size[0] - size[2])), 'constant', constant_values=255)
            contour_cor = np.pad(contour_cor, ((0, 0), (0, size[0] - size[2])), 'constant', constant_values=255)
            size_img = img_cor.shape
            stacked_cor_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
            stacked_cor_2 = np.zeros((size_img[0], size_img[1]), dtype=np.uint8)

            stacked_cor_1[:,:,0] = img_cor
            stacked_cor_1[:,:,1] = img_cor
            stacked_cor_1[:,:,2] = img_cor

            stacked_cor_2[:,:] = contour_cor
            unique, counts = numpy.unique(stacked_cor_2, return_counts=True)
            vals = dict(zip(unique, counts))
            if(vals[1]) > 0):
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

def create_tfrecord(fileList):

    return

def main(unused_argv):

    data_path = FLAGS.rawdata_dir

    if FLAGS.cerr:
        sys.stdout.write('Searching for .mat CERR files')
        p_num = 1
        matFiles = find('*.mat',data_path)
        if len(os.listdir(data_path)) > 0:
            for filename in matFiles:
                file = h5py.File(filename, 'r')
                scan = getScanArray(file)
                mask = getMaskArray(file)
                sys.stdout.write('\r>> Exporting patient %d of %d' % (
                    p_num, len(os.listdir(data_path))-1))
                data_export(scan, mask, FLAGS.save_dir, p_num, FLAGS.cerr, FLAGS.structure)
                p_num = p_num + 1
        else:
            print("No CERR .mat files found")

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
                    if item.ROIName == FLAGS.structure_match:
                         ## ss_maxslice: determines maximum number of image slices contour lives on
                        ss_maxslice = len(ss.ROIContours[k].Contours)

                        ## pattern collects referenced SOP for DICOM collection, searched dir for CT_files list
                        pattern = ss.ROIContours[k].Contours[0].ContourImageSequence[0].ReferencedSOPInstanceUID
                        pattern = '*' + '.'.join(pattern.split('.')[:-2])
                        pattern = pattern[:-3] + '*'
                        CT_files = find(pattern, data_path)

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
                                    if CT_files[slice].split(os.sep)[-1] == ss.ROIContours[k].Contours[j].ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm':
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

                        sys.stdout.write('\r>> Exporting patient %d of %d' % (p_num+1, len(RS_Files) - 1))
                        data_export(im_data, im_mask, FLAGS.save_dir, p_num, FLAGS.cerr, FLAGS.structure)

                    ## Iterate over contour
                    k = k + 1


        else:
            print('Directory specified contains no RS files')


if __name__ == '__main__':
  # flags.mark_flag_as_required('rawdata_dir')
  # flags.mark_flag_as_required('save_dir')
  # flags.mark_flag_as_required('structure')
  tf.app.run()