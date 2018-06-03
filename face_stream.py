from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from . import facenet
from . import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

print('Creating networks and loading parameters')


class FaceStream():

    def __init__(self,
            input_dir = './input_dir',
            modelfile = './pre_model/20180402-114759/20180402-114759.pb',
            mtcnn_dir = './d_npy',
            classifier_filename = './face_stream/my_class/my_classifier.pkl'):
        self.input_dir = input_dir
        self.modeldir = modelfile
        self.mtcnn_dir = mtcnn_dir
        self.classifier_filename = classifier_filename
        
        self.init()

    def init(self):
        self.define_const()
        self.load_model()

    def define_const(self):
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.frame_interval = 3
        self.batch_size = 1000
        self.image_size = 182
        self.input_image_size = 160
        self.HumanNames = os.listdir(self.input_dir)
        self.HumanNames.sort()

    def load_model(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(
                    self.sess, self.mtcnn_dir)
                
                self.pnet = pnet
                self.rnet = rnet
                self.onet = onet
                
                print('Loading feature extraction model')
                facenet.load_model(self.modeldir)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                classifier_filename_exp = os.path.expanduser(
                    self.classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    self.model = model
                    self.class_names = class_names

                    print('load classifier file-> %s' %
                          classifier_filename_exp)

                print('Load model soccess!')

    def detect_faces(self, input_img, output_img):
        frame = cv2.imread(input_img)

        # resize frame (optional)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # find_results = []

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)

        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(
            frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        # print('Detected_FaceNum: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            # may be this is inout_image_size
            # TODO: figure it out
            # img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, self.embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('face is inner of range!')
                    continue

                cropped.append(
                    frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(misc.imresize(
                    cropped[i], (self.image_size, self.image_size), interp='bilinear'))
                scaled[i] = cv2.resize(scaled[i], (self.input_image_size, self.input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(
                    scaled[i].reshape(-1, self.input_image_size, self.input_image_size, 3))
                feed_dict = {
                    self.images_placeholder: scaled_reshape[i], self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                # print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                # print(best_class_indices)
                # best_class_probabilities = predictions[np.arange(
                #     len(best_class_indices)), best_class_indices]
                # print(best_class_probabilities)
                # boxing face
                cv2.rectangle(
                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                # plot result idx under box
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                # print('detected: ', best_class_indices[0])
                # print(best_class_indices)
                # print(HumanNames)
                for H_i in self.HumanNames:
                    # print(H_i)
                    if self.HumanNames[best_class_indices[0]] == H_i:
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        result_names = self.HumanNames[best_class_indices[0]]
                        textsize = cv2.getTextSize(result_names, font, 1, 2)[0]

                        bg_x = text_x
                        bg_y = text_y - textsize[1]
                        bg_width = textsize[0]
                        bg_height = textsize[1]

                        cv2.rectangle(
                            frame, (bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height), (0, 0, 0), thickness=-1)
                        cv2.putText(frame, result_names, (text_x, text_y), font,
                                    1, (255, 255, 255), thickness=1, lineType=2)
        else:
            print('Unable to align')

        cv2.imwrite(output_img, frame)

        return frame
