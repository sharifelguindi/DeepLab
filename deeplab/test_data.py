from __future__ import print_function
import collections
import os
import sys
import tarfile
import tempfile
import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import dicom
from shapely import geometry
import numpy as np
from PIL import Image, ImageDraw
import os, fnmatch
from scipy.misc import imsave, imrotate
import matplotlib.pyplot as plt

import tensorflow as tf

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

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

# _FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 384

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        file_handle = open(tarball_path, 'rb')
        graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

base = "\\\\VPENSMPH\\DeasyLab1\\Sharif\\DATA"
contour_name = 'Rectum_O'
data_path = os.path.join('datasets','bladder','test','p19')
RS_Files = find('RS.*',data_path)

for p_num in range(0, len(RS_Files)):
    k = 0
    ss = dicom.read_file(RS_Files[p_num])
    print(RS_Files[p_num])
    for item in ss.StructureSetROISequence[:]:
        if item.ROIName == contour_name:
            print("Contour Found, collecting DICOM info")
            pointList = []
            ss_maxslice = len(ss.ROIContours[k].Contours)
            pattern = ss.ROIContours[k].Contours[0].ContourImageSequence[0].ReferencedSOPInstanceUID
            pattern = '.'.join(pattern.split('.')[:-2])
            pattern = pattern[:-2] + '*'
            CT_files = find(pattern, data_path)
            ct_maxslice = len(CT_files)
            img = dicom.read_file(CT_files[0])
            img_size = np.shape(img.pixel_array)
            im_mask = np.zeros((img_size[0], img_size[1], img_size[1]))
            im_data = np.zeros((img_size[0], img_size[1], img_size[1]))
            z0 = img.ImagePositionPatient[2]
            for slice in range(0, ct_maxslice):
                img = dicom.read_file(CT_files[slice])
                if z0 > img.ImagePositionPatient[2]:
                    z0 = img.ImagePositionPatient[2]
            for slice in range(0, ct_maxslice):
                img = dicom.read_file(CT_files[slice])
                z_prime = float(img.ImagePositionPatient[2])
                zsp = float(img.SliceThickness)
                z = int((z_prime - z0) / zsp)
                im_data[:, :, z] = img.pixel_array

model_path = os.path.join(base, 'datasets','rectum','exp','axial032518_1530','export','frozen_inference_graph.pb')
# model_path = os.path.join(base, 'datasets','bladder','exp','axial032618','export','frozen_inference_graph.pb')

im_mask = np.zeros((img_size[0], img_size[1], img_size[1]))
model = DeepLabModel(model_path)
for i in range(0,ct_maxslice):
    img = im_data[:, :, i]
    size_img = img.shape
    stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
    stacked_img_1[:, :, 0] = img
    stacked_img_1[:, :, 1] = img
    stacked_img_1[:, :, 2] = img
    imsave('hold.png', stacked_img_1)
    image = Image.open('hold.png')
    r_im, seg = model.run(image)
    im_mask[:,:,i] = seg

np.save(os.path.join(data_path,'ax_rec.npy'), im_mask)

# model_path = os.path.join(base, 'datasets','bladder','exp','saggital032618','export','frozen_inference_graph.pb')
# model_path = os.path.join(base, 'datasets','bladder','exp','saggital032618','export','frozen_inference_graph.pb')
#
# im_mask = np.zeros((img_size[0], img_size[1], img_size[1]))
# model = DeepLabModel(model_path)
# for i in range(0,size_img[1]):
#     img = im_data[:, i, :]
#     size_img = img.shape
#     stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
#     stacked_img_1[:, :, 0] = img
#     stacked_img_1[:, :, 1] = img
#     stacked_img_1[:, :, 2] = img
#     imsave('hold.png', stacked_img_1)
#     image = Image.open('hold.png')
#     r_im, seg = model.run(image)
#     im_mask[:,i,:] = seg
#
# np.save(os.path.join(data_path,'sag.npy'), im_mask)
#
# model_path = os.path.join(base, 'datasets','bladder','exp','coronal032618','export','frozen_inference_graph.pb')
# model_path = os.path.join(base, 'datasets','bladder','exp','coronal032618','export','frozen_inference_graph.pb')
#
# im_mask = np.zeros((img_size[0], img_size[1], img_size[1]))
# model = DeepLabModel(model_path)
# for i in range(0,img_size[1]):
#     img = im_data[i, :, :]
#     size_img = img.shape
#     stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
#     stacked_img_1[:, :, 0] = img
#     stacked_img_1[:, :, 1] = img
#     stacked_img_1[:, :, 2] = img
#     imsave('hold.png', stacked_img_1)
#     image = Image.open('hold.png')
#     r_im, seg = model.run(image)
#     im_mask[i,:,:] = seg
#
# np.save(os.path.join(data_path,'cor.npy'), im_mask)
