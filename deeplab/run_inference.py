from __future__ import print_function
import dicom
from dicom.sequence import Sequence
from dicom.dataset import Dataset
import numpy as np
from PIL import Image
import os, fnmatch
from scipy.misc import imsave, imrotate
import tensorflow as tf
import datetime
from skimage import measure
import h5py

## Defined variables, flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# Default Inputs
flags.DEFINE_boolean('cerr', False,
                     'Set to true to create/append .mat CERR file instead .')

flags.DEFINE_string('graph_name', 'frozen_inference_graph',
                    'inference graph name, standard is set by default')

flags.DEFINE_string('data_dir', 'datasets/MR_VALIDATION/p37',
                    'absolute path patient DICOM data, including RS object to append')

flags.DEFINE_string('save_dir', 'datasets/MR_VALIDATION/p37',
                    'absolute path to save RS object, typically same folder')

flags.DEFINE_string('model_dir', os.path.join('datasets','bladder','exp'),
                    'path to saved model directory (axial, coronal, saggital)')

flags.DEFINE_string('model_val', '',
                    'Identifier for models, typically date of training run (ex. 032818)')

flags.DEFINE_integer('inference_size', 320,
                    'Size of image used in training')

flags.DEFINE_string('structure', 'bladder',
                    'string name of structure folder')

flags.DEFINE_string('structure_match', 'Bladder_O',
                    'string name of structure to create')

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

def bbox2_3D(img, pad):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin - pad, rmax + pad, cmin - pad, cmax + pad, zmin - pad, zmax + pad

## DeepLabV3 class, uses frozen graph to load weights, make predictions
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = FLAGS.inference_size

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


def create_rtstruct(RS_File, im_mask_ax, im_mask_sag, im_mask_cor):

    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(im_mask_sag, 2)
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if zmin < 0:
        zmin = 0

    im_mask_ax_adj = np.zeros(np.shape(im_mask_ax))
    im_mask_ax_adj[rmin:rmax, cmin:cmax, zmin:zmax] = im_mask_ax[rmin:rmax, cmin:cmax, zmin:zmax]

    ss = dicom.read_file(RS_File)
    contour_name = FLAGS.structure
    data_path = FLAGS.data_dir

    ## Add Contour
    UID = ss.SOPInstanceUID.split('.')
    UID_NEW = UID[:-1]
    UID_NEW.append(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[0:19])
    ss.SOPInstanceUID = '.'.join(UID_NEW)
    ss.StructureSetName = 'DeepLabV3'
    ss.StructureSetLabel = 'DeepLabV3'
    ss.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    ss.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S.%f")
    ROINumList = []
    for s in ss.ROIContourSequence:
        ROINumList.append(s.ReferencedROINumber)

    ## Add StructureSetROISequence
    ss_new = Dataset()
    ss_new.ROINumber = np.int(max(ROINumList)) + 1
    ss_new.ReferencedFrameOfReferenceUID = ss.StructureSetROISequence[len(ROINumList) - 1].ReferencedFrameOfReferenceUID
    ss_new.ROIName = FLAGS.structure_match + '_DLV3'
    ss_new.ROIDescription = ''
    ss_new.ROIGenerationAlgorithm = 'MANUAL'
    ss.StructureSetROISequence.append(ss_new)

    ## Add RTROIObservationsSequence
    ss_new = Dataset()
    ss_new.ObservationNumber = np.int(max(ROINumList)) + 1
    ss_new.ReferencedROINumber = np.int(max(ROINumList)) + 1
    ss_new.ROIObservationDescription = 'Type:Soft, Range:*/*, Fill:0, Opacity:0.0, Thickness:1, LineThickness:2'
    ss_new.RTROIInterpretedType = ''
    ss_new.ROIInterpreter = ''
    ss.RTROIObservationsSequence.append(ss_new)

    ## Add ROIContourSequence
    ss_new = Dataset()
    ss_new.ReferencedROINumber = np.int(max(ROINumList)) + 1
    ss_new.ROIDisplayColor = ['255', '0', '0']
    ss_new.ContourSequence = Sequence()

    k = 0
    ss_referenceclass = ss.ROIContours[0].Contours[0].ContourImageSequence[0].ReferencedSOPClassUID
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
            try:
                CT_files.remove(RS_File)
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
                    if 'RS' in CT_files[slice]:
                        print('not structure')
                    else:
                        if z0 > img.ImagePositionPatient[2]:
                            z0 = img.ImagePositionPatient[2]

            for slice in range(0, ct_maxslice):
                if 'RS' in CT_files[slice]:
                    print('not structure')
                else:
                    contour_dicom = Dataset()
                    img = dicom.read_file(CT_files[slice])
                    x_y = np.array(img.ImagePositionPatient)
                    xsp_ysp = np.array(img.PixelSpacing)
                    z_prime = float(img.ImagePositionPatient[2])
                    zsp = float(img.SliceThickness)
                    z = int((z_prime - z0) / zsp)
                    if np.max(im_mask_ax_adj[:, :, z]) > 0 and (np.max(im_mask_sag[:, :, z] > 0)):
                        r = im_mask_ax_adj[:, :, z]
                        contours = measure.find_contours(r, 0.5)
                        for n, contour in enumerate(contours):
                            pointList = []
                            contour_dicom = Dataset()
                            contour_dicom.ContourGeometricType = 'CLOSED_PLANAR'
                            for i in range(0, len(contour)):
                                y = contour[i][0]
                                x = contour[i][1]
                                x_prime = x * xsp_ysp[0] + x_y[0]
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
                                img_seq.ReferencedSOPInstanceUID = CT_files[slice].split(os.sep)[-1].replace('.dcm','').replace('MR.', '')
                                contour_dicom.ContourImageSequence.append(img_seq)
                                ss_new.ContourSequence.append(contour_dicom)
            k = k + 1

    ss.ROIContourSequence.append(ss_new)
    filename = RS_File.split(os.sep)
    filename[-1] = contour_name + str(datetime.datetime.now().strftime("%Y%m%d")) + '.dcm'
    print(filename)
    ss.save_as(os.sep.join(filename))

    return

def main(unused_argv):
    if tf.__version__ < '1.5.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

    contour_name = FLAGS.structure
    data_path = FLAGS.data_dir
    RS_Files = find('RS.*.dcm', data_path)

    ## Start loop through each file (p_num = patient number)
    if RS_Files:
        print('Test data found...')
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
                    try:
                        CT_files.remove(RS_Files[p_num])
                    except:
                        print('RS not found in CT list')
                    if CT_files:


                        ## Open first CT image, get size, total number of files and
                        ## initialize Numpy Arrays for data collection
                        ct_maxslice = len(CT_files)


                        if 'RS' in CT_files[0]:
                            img = dicom.read_file(CT_files[1])
                            img_size = np.shape(img.pixel_array)
                            im_data = np.zeros((img_size[0], img_size[1], ct_maxslice))
                            z0 = img.ImagePositionPatient[2]
                        else:
                            img = dicom.read_file(CT_files[0])
                            img_size = np.shape(img.pixel_array)
                            im_data = np.zeros((img_size[0], img_size[1], ct_maxslice))
                            z0 = img.ImagePositionPatient[2]

                        ## Since DICOM files are not in spatial order, determine
                        ## "z0" or starting z position
                        for slice in range(0, ct_maxslice):
                            if 'RS' in CT_files[slice]:
                                print('not structure')
                            else:
                                img = dicom.read_file(CT_files[slice])
                                if z0 > img.ImagePositionPatient[2]:
                                    z0 = img.ImagePositionPatient[2]

                        ## Start loop through each CT slice found
                        for slice in range(0, ct_maxslice):
                            ## Read pixel array and image location, convert to numpy reference frame
                            ## and place into im_data as appropriate location
                            if 'RS' in CT_files[slice]:
                                print('not structure')
                            else:
                                img = dicom.read_file(CT_files[slice])
                                z_prime = float(img.ImagePositionPatient[2])
                                zsp = float(img.SliceThickness)
                                z = int((z_prime - z0) / zsp)
                                im_data[:, :, z] = img.pixel_array

            ## Defined Imaging planes
            planeList = ['axial', 'coronal', 'saggital']
            size = im_data.shape
            model_val = FLAGS.model_val
            save_dir = FLAGS.save_dir
            infer_size = FLAGS.inference_size
            strt = int(np.floor((size[0] - infer_size) / 2 ))
            stop = int(np.ceil((size[0] - infer_size) / 2 )) + infer_size
            im_data_resized = im_data[strt:stop,strt:stop,:]
            im_data_resized = np.pad(im_data_resized,((0,0),(0,0),(0,int(infer_size - size[2]))),'constant', constant_values=0)

            ## Loop through each plane and load subsequence model
            for plane in planeList:
                model_path = os.path.join(FLAGS.model_dir, plane + model_val, 'export', 'frozen_inference_graph.pb')
                model = DeepLabModel(model_path)
                if plane == 'axial':
                    print('Computing Axial Model...')
                    im_mask_ax = np.zeros((img_size[0], img_size[1], img_size[1]))
                    for i in range(0,size[2]):
                        img = im_data_resized[:,:,i]
                        size_img = img.shape
                        stacked_img_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
                        stacked_img_1[:,:,0] = img
                        stacked_img_1[:,:,1] = img
                        stacked_img_1[:,:,2] = img
                        imsave('hold.png', stacked_img_1)
                        image = Image.open('hold.png')
                        r_im, seg = model.run(image)
                        im_mask_ax[strt:stop,strt:stop, i] = seg
                    with h5py.File(os.path.join(save_dir, 'axial.h5'), 'w') as hf:
                        hf.create_dataset("im_mask_ax", data=im_mask_ax, compression="lzf")
                elif plane == 'coronal':
                    print('Computing Coronal Model...')
                    im_mask_cor = np.zeros((img_size[0], img_size[1], img_size[1]))
                    for i in range(0, infer_size):
                        img_cor = im_data_resized[i, :, :]
                        # img_cor = np.pad(img_cor, ((0, 0), (0, size[0] - size[2])), 'constant', constant_values=255)
                        size_img = img_cor.shape
                        stacked_cor_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
                        stacked_cor_1[:, :, 0] = img_cor
                        stacked_cor_1[:, :, 1] = img_cor
                        stacked_cor_1[:, :, 2] = img_cor

                        imsave('hold.png', stacked_cor_1)
                        image = Image.open('hold.png')
                        r_im, seg = model.run(image)
                        im_mask_cor[i,strt:stop,0:infer_size] = seg
                    with h5py.File(os.path.join(save_dir, 'coronal.h5'), 'w') as hf:
                        hf.create_dataset("im_mask_cor", data=im_mask_cor, compression="lzf")
                elif plane == 'saggital':
                    print('Computing Saggital Model...')
                    im_mask_sag = np.zeros((img_size[0], img_size[1], img_size[1]))
                    for i in range(0, infer_size):
                        img_sag = im_data_resized[:, i, :]
                        # img_sag = np.pad(img_sag, ((0, 0), (0, size[0] - size[2])), 'constant', constant_values=255)
                        size_img = img_cor.shape
                        stacked_sag_1 = np.zeros((size_img[0], size_img[1], 3), dtype=np.int16)
                        stacked_sag_1[:, :, 0] = img_sag
                        stacked_sag_1[:, :, 1] = img_sag
                        stacked_sag_1[:, :, 2] = img_sag

                        imsave('hold.png', stacked_sag_1)
                        image = Image.open('hold.png')
                        r_im, seg = model.run(image)
                        im_mask_sag[strt:stop,i,0:infer_size] = seg
                    with h5py.File(os.path.join(save_dir, 'saggital.h5'), 'w') as hf:
                        hf.create_dataset("im_mask_sag", data=im_mask_sag, compression="lzf")
            print('Creating DICOM object')
            create_rtstruct(RS_Files[p_num], im_mask_ax, im_mask_sag, im_mask_cor)

if __name__ == '__main__':
  tf.app.run()