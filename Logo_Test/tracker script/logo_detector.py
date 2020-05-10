import numpy as np
import tensorflow.compat.v1 as tf
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
import cv2
#from extract_people import PersonDetector


class LogoDetector(object):
    def __init__(self):

        self.logo_boxes = []


        #Tensorflow localization/detection model
        # Single-shot-dectection with mobile net architecture trained on COCO dataset

        detect_model_name = 'person_based_logo_model_v1'

        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'

        # setup tensorflow graph
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def get_localization(self, image, visual=False):

        """Determines the locations of the cars in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        category_index={1: {'id': 1, 'name': u'attlogo'}}

        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})

              if visual == True:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)

                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()

              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)

              cls = classes.tolist()

              # The ID for car in COCO data set is 3
              idx_vec = [i for i, v in enumerate(cls) if ((v==1) and (scores[i]>0.8))]

              if len(idx_vec) ==0:
                  print('no detection!')
                  self.logo_boxes = []
              else:
                  tmp_logo_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)

                      tmp_logo_boxes.append(box)


                  self.logo_boxes = tmp_logo_boxes

        return self.logo_boxes

if __name__ == '__main__':
    vidname = "Test2B_ims"
    det = PersonDetector()
    logo_det = LogoDetector()
    images = [file for file in sorted(glob.glob('./{}/*.png'.format(vidname)))]
    print(len(images))
    for i in range(250,260):
        image = plt.imread(images[i])
        if np.max(image) == 1:
            image = (image*255).astype("uint8")
        z_box = det.get_localization(image)
        plt.imshow(image)
        plt.show()
        for j in range(len(z_box)):
            ymin, xmin, ymax, xmax = z_box[j]
            im_person = image[ymin:ymax, xmin:xmax]
            #im_person = cv2.cvtColor(im_person, cv2.COLOR_BGR2RGB)
            logo_boxes = logo_det.get_localization(im_person)
            print("there are {} logos".format(len(logo_boxes)))
            if(len(logo_boxes) > 0):
                l_ymin, l_xmin, l_ymax, l_xmax = logo_boxes[0]
                xdiff  = l_xmax - l_xmin
                ydiff = l_ymax - l_ymin
                rect = patches.Rectangle((l_xmin+xmin,l_ymin+ymin),xdiff,ydiff,linewidth=1,edgecolor='r',facecolor='none')

                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.add_patch(rect)
                plt.show()
