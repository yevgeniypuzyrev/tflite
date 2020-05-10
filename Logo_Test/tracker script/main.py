#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: kyleguan
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR
import cv2
import argparse
import pickle

import helpers
import detector
import tracker
import logo_detector

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 10  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

dead_tracker_list = [] # list for trackers even after they're dead

tracker_list =[] # list for trackers
# list for track ID
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
track_id_start_value = 1
debug = True

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    print(IOU_mat)
    matched_idx = linear_assignment(-IOU_mat)
    print(matched_idx)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global dead_tracker_list
    global track_id_start_value
    global max_age
    global min_hits
    global track_id_list
    global debug
    global args

    frame_count+=1

    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img) # measurement
    img_raw = np.copy(img)
    if debug:
       print('Frame:', frame_count)

    x_box =[]
    if debug:
        for i in range(len(z_box)):
            if not args['dots']:
                img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
            #plt.imshow(img1)
        plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            trk.predict_only()
            xx = trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            trk.box = xx
            x_box.append(trk.box)


    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.1)
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_color = (255, 255, 255)
    cv2.putText(img,str(len(z_box)),(100,100), font, font_size, font_color, 1, cv2.LINE_AA)
    pos = 30
    for trk_idx, det_idx in matched:
        iou = helpers.box_iou2(tracker_list[trk_idx].box,z_box[det_idx])
        cv2.putText(img,tracker_list[trk_idx].id + " " + str(iou),(100,100+pos), font, font_size, font_color, 1, cv2.LINE_AA)
        pos+=30
    cv2.putText(img,str(frame_count),(100,100+pos), font, font_size, font_color, 1, cv2.LINE_AA)
    # Deal with matched detections
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            ymin, xmin, ymax, xmax = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            person_im = img_raw[ymin:ymax, xmin:xmax]
            logo_boxes = logo_det.get_localization(person_im)
            if(len(logo_boxes) > 0):
                l_ymin, l_xmin, l_ymax, l_xmax = logo_boxes[0]
                img= helpers.draw_box_label(img, [l_ymin+z[0], l_xmin+z[1], l_ymax+z[0], l_xmax+z[1]], id = "logo", box_color = (255, 0, 0))
                logo_x = (l_xmax-l_xmin)/2 + (l_xmin + z[1])
                logo_y = (l_ymax-l_ymin)/2 + (l_ymin + z[0])
                img = cv2.circle(img, (logo_x,logo_y), 5, trk.color, 2)
                tmp_trk.logo_x_coords.append(logo_x)
                tmp_trk.logo_y_coords.append(logo_y)
            if not args['dots']:
                img= helpers.draw_box_label(img, tmp_trk.box, id = tmp_trk.id, box_color = (0, 255, 0))
            #tmp_trk.kalman_filter(z)
            tmp_trk.update_only(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.y_coords.append(int((xx[2] - xx[0])/2 + xx[0]))
            tmp_trk.x_coords.append(int((xx[3] - xx[1])/2 + xx[1]))
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            ymin, xmin, ymax, xmax = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            person_im = img_raw[ymin:ymax, xmin:xmax]
            logo_boxes = logo_det.get_localization(person_im)
            if(len(logo_boxes) > 0):
                l_ymin, l_xmin, l_ymax, l_xmax = logo_boxes[0]
                img= helpers.draw_box_label(img, [l_ymin+z[0], l_xmin+z[1], l_ymax+z[0], l_xmax+z[1]], id = "logo", box_color = (255, 0, 0))
                logo_x = (l_xmax-l_xmin)/2 + (l_xmin + z[1])
                logo_y = (l_ymax-l_ymin)/2 + (l_ymin + z[0])
                img = cv2.circle(img, (logo_x,logo_y), 5, trk.color, 2)
                tmp_trk.logo_x_coords.append(logo_x)
                tmp_trk.logo_y_coords.append(logo_y)
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.y_coords.append(int((xx[2] - xx[0])/2 + xx[0]))
            tmp_trk.x_coords.append(int((xx[3] - xx[1])/2 + xx[1]))
            tmp_trk.id = str(track_id_start_value) # assign an ID for the tracker
            track_id_start_value += 1
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            if not args['dots']:
                img= helpers.draw_box_label(img, tmp_trk.box, id = tmp_trk.id, box_color = (255, 255, 0))
            tmp_trk.no_losses += 1
            #tmp_trk.predict_only()
            #xx = tmp_trk.x_state
            #xx = xx.T[0].tolist()
            #xx =[xx[0], xx[2], xx[4], xx[6]]
            #tmp_trk.box =xx
            #x_box[trk_idx] = xx


    # The list of tracks to be annotated
    good_tracker_list =[]
    for trk in tracker_list:
        print(trk.id)
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             if args['dots']:
                 print("COLOR IS: {}".format(trk.color))
                 for i in zip(trk.x_coords, trk.y_coords):
                     img = cv2.circle(img, i, 10, trk.color, 5)
             if not args['dots']:
                 img= helpers.draw_box_label(img, x_cv2, id = trk.id) # Draw the bounding boxes on the
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)

    for trk in deleted_tracks:
            dead_tracker_list.append(trk)

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))


    return img

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", action="store_true",
	help="If set, output video file")
    ap.add_argument("-d", "--debug", action="store_true",
	help="If set, show frame by frame")
    ap.add_argument("-t", "--dots", action="store_true",
	help="If set, draw dots each frame instead of boxes")
    args = vars(ap.parse_args())

    det = detector.CarDetector()
    logo_det = logo_detector.LogoDetector()
    vidname = "Test2A"
    if debug: # test on a sequence of images
        images = [file for file in sorted(glob.glob('./{}_ims/*.png'.format(vidname)))]
        print(len(images))
        width = plt.imread(images[0]).shape[1]
        height = plt.imread(images[0]).shape[0]
        FPS = 4
        seconds = len(images)/4
        if args['video']:
            fourcc = VideoWriter_fourcc(*'MP42')
            video = VideoWriter('./output/{}_logo_test_old_lab.avi'.format(vidname), fourcc, float(FPS), (width, height))
        for i in range(len(images)):
             image = plt.imread(images[i])
             if np.max(image) == 1:
                 image = (image*255).astype("uint8")
             image_box = pipeline(image)
             if args['debug']:
                 plt.imshow(image_box)
                 plt.show()
             if args['video']:
                video.write(cvtColor(image_box, COLOR_RGB2BGR))
        if args['video']:
            video.release()
        image = plt.imread(images[0])
        for trk in dead_tracker_list:
            print(trk.id)
            print(trk.x_coords)
            print(trk.y_coords)
            if(len(trk.x_coords) > 20):
                color = (np.random.randint(0,256)/255, np.random.randint(0,256)/255, np.random.randint(0,256)/255)
                print(color)
                for i in zip(trk.y_coords, trk.x_coords):
                    image = cv2.circle(image, i, 10, color, 5)
        #plt.imshow(image)
        #plt.show()
        pickle.dump([i for i in dead_tracker_list if len(i.x_coords) > 20], open("./output/{}_trackers_with_logo.p".format(vidname), "wb") )

    else: # test on a video file.

        start=time.time()
        output = 'test_v7.mp4'
        clip1 = VideoFileClip("project_video.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(output, audio=False)
        end  = time.time()

        print(round(end-start, 2), 'Seconds to finish')
