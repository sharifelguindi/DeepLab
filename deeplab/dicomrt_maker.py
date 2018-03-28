from __future__ import print_function
import dicom
from dicom.sequence import Sequence
from dicom.dataset import Dataset
from shapely import geometry
import numpy as np
from PIL import Image, ImageDraw
import os, fnmatch
from scipy.misc import imsave, imrotate
import matplotlib.pyplot as plt
from skimage import measure

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

contour_name = 'Rectum_O'
data_path = os.path.join('bladder','test','p19')
RS_Files = find('RS.*',data_path)
ss = dicom.read_file(RS_Files[0])
im_mask_ax = np.load(os.path.join(data_path,'ax.npy'))
# im_mask_sag = np.load(os.path.join(data_path,'sag.npy'))
# im_mask_cor = np.load(os.path.join(data_path,'cor.npy'))

## Add Contour
ss = dicom.read_file(RS_Files[0])
UID = ss.SOPInstanceUID
UID_NEW = UID[:-1] + '4'
ss.SOPInstanceUID = UID_NEW
ss.StructureSetName = 'DeepLabV3'
ss.StructureSetLabel = 'DeepLabV3'
ss.InstanceCreationDate = '20180325'
c_num = len(ss.ROIContourSequence)

## Add StructureSetROISequence
ss_new = Dataset()
ss_new.ROINumber = c_num + 1
ss_new.ReferencedFrameOfReferenceUID = ss.StructureSetROISequence[c_num - 1].ReferencedFrameOfReferenceUID
ss_new.ROIName = 'Bladder_Test'
ss_new.ROIDescription = ''
ss_new.ROIGenerationAlgorithm = 'MANUAL'
ss.StructureSetROISequence.append(ss_new)

## Add RTROIObservationsSequence
ss_new = Dataset()
ss_new.ObservationNumber = c_num + 1
ss_new.ReferencedROINumber = c_num + 1
ss_new.ROIObservationDescription = 'Type:Soft, Range:*/*, Fill:0, Opacity:0.0, Thickness:1, LineThickness:2'
ss_new.RTROIInterpretedType = ''
ss_new.ROIInterpreter = ''
ss.RTROIObservationsSequence.append(ss_new)

## Add ROIContourSequence
ss_new = Dataset()
ss_new.ReferencedROINumber = 9
ss_new.ROIDisplayColor = ['255','0','0']
ss_new.ContourSequence = Sequence()

for item in ss.StructureSetROISequence[:]:
    if item.ROIName == contour_name:
        ss_maxslice = len(ss.ROIContours[0].Contours)
        ss_referenceclass = ss.ROIContours[0].Contours[0].ContourImageSequence[0].ReferencedSOPClassUID
        pattern = ss.ROIContours[0].Contours[0].ContourImageSequence[0].ReferencedSOPInstanceUID
        pattern = '.'.join(pattern.split('.')[:-2])
        pattern = pattern[:-2] + '*'
        CT_files = find(pattern, data_path)
        ct_maxslice = len(CT_files)
        img = dicom.read_file(CT_files[0])
        img_size = np.shape(img.pixel_array)
        z0 = img.ImagePositionPatient[2]
        for slice in range(0, ct_maxslice):
            img = dicom.read_file(CT_files[slice])
            if z0 > img.ImagePositionPatient[2]:
                z0 = img.ImagePositionPatient[2]
        for slice in range(0, ct_maxslice):
            contour_dicom = Dataset()
            img = dicom.read_file(CT_files[slice])
            x_y = np.array(img.ImagePositionPatient)
            xsp_ysp = np.array(img.PixelSpacing)
            z_prime = float(img.ImagePositionPatient[2])
            zsp = float(img.SliceThickness)
            z = int((z_prime - z0) / zsp)
            if np.max(im_mask_ax[:,:,z]) > 0 and np.max(im_mask_ax[:,:,z]) > 0:
                r = im_mask_ax[:,:,z]
                contours = measure.find_contours(r, 0.5)
                for n, contour in enumerate(contours):
                    pointList = []
                    contour_dicom = Dataset()
                    contour_dicom.ContourGeometricType = 'CLOSED_PLANAR'
                    for i in range(0, len(contour)):
                        y = contour[i][0]
                        x = contour[i][1]
                        x_prime = x*xsp_ysp[0] + x_y[0]
                        y_prime = y * xsp_ysp[1] + x_y[1]
                        pointList.append(x_prime)
                        pointList.append(y_prime)
                        pointList.append(z_prime)

                    if len(pointList) > 0:
                        contour_dicom.NumberOfContourPoints = len(contour)
                        contour_dicom.ContourData = pointList
                        contour_dicom.ContourImageSequence = Sequence()
                        img_seq = Dataset()
                        img_seq.ReferencedSOPClassUID = ss_referenceclass
                        img_seq.ReferencedSOPInstanceUID = CT_files[slice].split(os.sep)[-1].replace('.dcm','')
                        contour_dicom.ContourImageSequence.append(img_seq)
                        ss_new.ContourSequence.append(contour_dicom)

ss.ROIContourSequence.append(ss_new)
filename = RS_Files[0].split(os.sep)
filename[-1] = 'test.dcm'
print(filename)
ss.save_as(os.sep.join(filename))





