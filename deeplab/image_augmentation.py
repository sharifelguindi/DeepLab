import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
from math import ceil, floor
import cv2
from math import pi
from scipy.misc import imsave, toimage
import os
from math import ceil, floor
from random import random
import math

IMAGE_SIZE = 256

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs

def bbox2(img, scales):
    pad = 50
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = rmin - pad
    rmax = rmax + pad
    cmin = cmin - pad
    cmax = cmax + pad

    return np.array([[(rmin*scales[0], cmax),
                      (rmin*scales[1], cmin),
                      (rmax*scales[2], cmin),
                      (rmax*scales[3], cmax)]],dtype=np.int32)

def flip_images(X_imgs, n_channels):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype=np.float32)
    return X_flip

def get_perspective_matrices(X_img, seg, scales):
    offset = 0
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(bbox2(seg, scales))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix

def perspective_transform(X_img, seg, scales):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img, seg, scales)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    return warped_img

def central_scale_images(X_imgs, scales, n_channels, img_scale):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = img_scale - img_scale * scale  # To scale centrally
        x2 = y2 = img_scale + img_scale * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, n_channels))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data

def rotate_images(X_imgs, start_angle, end_angle, n_images, n_channels):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, n_channels))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate

def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]

  with tf.Session() as sess:
      crop = tf.random_crop(
          combined_pad,
          size=tf.concat([size, [last_label_dim + last_image_dim]],
                         axis=0))
      combined_crop = crop.eval()

  return (combined_crop[:, :, 0:3],
          combined_crop[:, :, 3])

def save_augmentations(img_aug, seg_aug, save_path, img_name, gt_name):

    seg_reshape = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    seg_reshape[0,:,:,0] = seg_aug[0]

    seg_rot = rotate_images(seg_reshape, 90, -90, 14, 1)
    img_rot = rotate_images(img_aug, 90, -90, 14, 3)

    bbox2(seg_reshape, [1,1,1,1])
    seg_scale = central_scale_images(seg_reshape, [1.8, 1.5, 1.2, 0.9, 0.6, 0.3], 1, 0.5)
    img_scale = central_scale_images(img_aug, [1.8, 1.5, 1.2, 0.9, 0.6, 0.3], 3, 0.5)

    seg_flip = flip_images(seg_reshape, 1)
    img_flip = flip_images(img_aug, 3)


    img_crop = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3), dtype='float32')
    seg_crop = np.zeros((IMAGE_SIZE,IMAGE_SIZE,1), dtype='float32')

    for i in range(0,20):
        cropped_image, cropped_labels = random_crop_and_pad_image_and_labels(
            image=tf.convert_to_tensor(img_aug[0], dtype='float64'),
            labels=tf.convert_to_tensor(seg_reshape[0, :, :, :]),
            size=[160, 160])
        for j in range(0,2):
            img_crop[:,:,j] =  cv2.resize(cropped_image[:,:,j],(256,256))
        seg_crop[:,:,0] =  cv2.resize(cropped_labels[:,:],(256,256))
        toimage(img_crop, cmin=0, cmax=255).save(os.path.join(save_path, img_name + '_crop_' + str(i) + '.png'))
        toimage(seg_crop[:,:,0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '_crop_' + str(i) + '.png'))

    for i in range(0, len(seg_flip)):
        toimage(img_flip[i], cmin=0, cmax=255).save(os.path.join(save_path, img_name + '_flip_' + str(i) + '.png'))
        toimage(seg_flip[i,:,:,0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '_flip_' + str(i) + '.png'))


    for i in range(0, len(seg_scale)):
        toimage(img_scale[i], cmin=0, cmax=255).save(os.path.join(save_path, img_name + '_scale_' + str(i) + '.png'))
        toimage(seg_scale[i,:,:,0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '_scale_' + str(i) + '.png'))

    for i in range(0, len(seg_rot)):
        toimage(img_rot[i], cmin=0, cmax=255).save(os.path.join(save_path, img_name + '_rot_' + str(i) + '.png'))
        toimage(seg_rot[i,:,:,0], cmin=0, cmax=255).save(os.path.join(save_path, gt_name + '_rot_' + str(i) + '.png'))

    return