from __future__ import print_function
import numpy as np
import os, fnmatch
import dicom
from shapely import geometry
from PIL import Image, ImageDraw
from collections import defaultdict
import sys
import cv2
from random import random
import math


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

# Takes RS file path, directory of DICOM images and structure, returns scan and mask
def load_dicom_toarray(RS_File, img_list, structure_match):

    ## Read RS file into dicom class, ss
    ss = dicom.read_file(RS_File)
    item_found = False
    if not structure_match:
        k = 0
    else:
        ## Start loop through each structure in RS file
        k = 0
        for item in ss.StructureSetROISequence[:]:
            ## Check if structure is equal to specified structure name
            if item.ROIName == structure_match:
                item_found = True
                ss_maxslice = len(ss.ROIContours[k].Contours)
                break;
            k = k + 1

    if img_list:

        ## Open first CT image, get size, total number of files and
        ## initialize Numpy Arrays for data collection
        ct_maxslice = len(img_list)
        img = dicom.read_file(img_list[0])
        img_size = np.shape(img.pixel_array)
        im_mask = np.zeros((img_size[0], img_size[1], ct_maxslice))
        im_data = np.zeros((img_size[0], img_size[1], ct_maxslice))
        z0 = img.ImagePositionPatient[2]

        ## Since DICOM files are not in spatial order, determine
        ## "z0" or starting z position
        for slice in range(0, ct_maxslice):
            img = dicom.read_file(img_list[slice])
            if z0 > img.ImagePositionPatient[2]:
                z0 = np.float(img.ImagePositionPatient[2])

        ## Start loop through each CT slice found
        for slice in range(0, ct_maxslice):

            ## Read pixel array and image location, convert to numpy reference frame
            ## and place into im_data at appropriate location
            img = dicom.read_file(img_list[slice])
            z_prime = float(img.ImagePositionPatient[2])
            zsp = float(img.SliceThickness)
            z = int(np.round((z_prime - z0) / zsp))
            im_data[:, :, z] = img.pixel_array


            if item_found:
                ## Start for loop through strucutre set point lists
                ## ss_maxslice is the number of contour objects for structure in question
                for j in range(0, ss_maxslice):

                    ## check CT file name against reference UID in contour object
                    if img_list[slice].split(os.sep)[-1] == ss.ROIContours[k].Contours[j].ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm' or \
                       img_list[slice].split(os.sep)[-1] == 'MR.' + ss.ROIContours[k].Contours[j].ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm':
                        ## Initialize point list and determin x,y,z pixel spacing, dicom positioning
                        pointList = []
                        x_y = np.array(img.ImagePositionPatient)
                        xsp_ysp = np.array(img.PixelSpacing)
                        size = len(ss.ROIContours[k].Contours[j].ContourData)

                        ## For loop converts point list to numpy reference frame at appropriate
                        ## z locations
                        for i in range(0, size, 3):
                            x_prime = float(ss.ROIContours[k].Contours[j].ContourData[i])
                            y_prime = float(ss.ROIContours[k].Contours[j].ContourData[i + 1])
                            x = (x_prime - x_y[0]) / xsp_ysp[0]
                            y = (y_prime - x_y[1]) / xsp_ysp[1]
                            p = geometry.Point(x, y)
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
                        im_mask[:, :, z] = mask

    return im_data, im_mask

def collect_dicom(pattern, path):
    sys.stdout.write('\n')
    dicom_files = find(pattern, path)
    d_img = defaultdict(list)
    d_ss = defaultdict(list)
    counter = 1
    for f in dicom_files:
        sys.stdout.write('\r>> Reading patient file %d of %d' % (counter, len(dicom_files)))
        sys.stdout.flush()
        img = dicom.read_file(f)
        if img.Modality != 'RTSTRUCT':
            d_img[img.StudyInstanceUID].append(f)
        elif img.Modality == 'RTSTRUCT':
            d_ss[img.StudyInstanceUID].append(f)
        else:
            print(f[:-1] + ' is not used')
        counter = counter + 1

    return d_img, d_ss

def bit_conversion(img, stacked_img_1, LUT, structure):

    if structure == 'parotid_L':
        # w1_low = 850
        # w1_high = 1350
        # w2_low = 850
        # w2_high = 1350
        # w3_low = 850
        # w3_high = 1350
        w1_low = 0
        w1_high = 1700
        w2_low = 500
        w2_high = 1700
        w3_low = 1000
        w3_high = 1700
        LUT_1 = np.clip(LUT, w1_low, w1_high)
        LUT_2 = np.clip(LUT,  w2_low, w2_high)
        LUT_3 = np.clip(LUT,  w3_low, w3_high)
        for i in range(0, len(LUT)):
            LUT_1[i] = np.int((255. / (w1_high - w1_low)) * LUT_1[i] - ((255. / (w1_high - w1_low))*w1_low))
            LUT_2[i] = np.int((255. / (w2_high - w2_low)) * LUT_2[i] - ((255. / (w2_high - w2_low))*w2_low))
            LUT_3[i] = np.int((255. / (w2_high - w2_low)) * LUT_3[i] - ((255. / (w3_high - w3_low))*w3_low))

    elif structure == 'bladder':
        w1_low = 0
        w1_high = 1700
        w2_low = 500
        w2_high = 1200
        w3_low = 1000
        w3_high = 1700

        LUT_1 = np.clip(LUT, w1_low, w1_high)
        LUT_2 = np.clip(LUT,  w2_low, w2_high)
        LUT_3 = np.clip(LUT,  w3_low, w3_high)
        for i in range(0, len(LUT)):
            LUT_1[i] = np.int((255. / (w1_high - w1_low)) * LUT_1[i] - ((255. / (w1_high - w1_low))*w1_low))
            LUT_2[i] = np.int((255. / (w2_high - w2_low)) * LUT_2[i] - ((255. / (w2_high - w2_low))*w2_low))
            LUT_3[i] = np.int((255. / (w2_high - w2_low)) * LUT_3[i] - ((255. / (w3_high - w3_low))*w3_low))

    elif structure == 'rectum':
        w1_low = 150
        w1_high = 1800
        w2_low = 150
        w2_high = 1800
        w3_low = 150
        w3_high = 1800

        LUT_1 = np.clip(LUT, w1_low, w1_high)
        LUT_2 = np.clip(LUT,  w2_low, w2_high)
        LUT_3 = np.clip(LUT,  w3_low, w3_high)
        for i in range(0, len(LUT)):
            LUT_1[i] = np.int((255. / (w1_high - w1_low)) * LUT_1[i] - ((255. / (w1_high - w1_low))*w1_low))
            LUT_2[i] = np.int((255. / (w2_high - w2_low)) * LUT_2[i] - ((255. / (w2_high - w2_low))*w2_low))
            LUT_3[i] = np.int((255. / (w2_high - w2_low)) * LUT_3[i] - ((255. / (w3_high - w3_low))*w3_low))

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

def normalize_array(arr):

    norm_arr = np.zeros(np.shape(arr), dtype='uint8')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 255, cv2.NORM_MINMAX)
    return norm_arr

def equalize(img, stacked_img, clahe):

    eq_img = clahe.apply(img)
    stacked_img[:, :, 0] = eq_img
    stacked_img[:, :, 1] = eq_img
    stacked_img[:, :, 2] = eq_img

    return eq_img, stacked_img.astype('uint8')
