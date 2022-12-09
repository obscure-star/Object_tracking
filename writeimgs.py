# Author: Jason Grant, Villanova University
# Read Ground Truth Files
# CSC 5930/9010 - Computer Vision
# MOT Challenge
# Date Created: December 2, 2021

import cv2
import numpy as np
import sys
import os

def getboxes(filename):
    """
    Reads in the ground truth file and returns a dictionary
    key of the dictionary is the frame and the value is a
    list of all bounding boxes in the frame
    Note: other data fields ignored
    """

    boxes = dict()

    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            data = line.split(',')
            frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = data

            frame = int(frame)
            id = int(id)
            bb_left = int(bb_left)
            bb_top = int(bb_top)
            bb_width = int(bb_width)
            bb_height = int(bb_height)

            if frame not in boxes:
                boxes[frame] = [(bb_left, bb_top, bb_width, bb_height, id)]
            else:
                boxes[frame].append( (bb_left, bb_top, bb_width, bb_height, id) )

    return boxes

def main():

    if len(sys.argv) < 3:
        print(f'{sys.argv[0]} [groundtruth] [img dir] [output dir]')
        sys.exit()

    groundtruth = sys.argv[1]
    indir = sys.argv[2]
    outdir = sys.argv[3]

    # Read in the file and get all of the bounding box data
    bb = getboxes(groundtruth)

    cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)

    framenum = 1
    with os.scandir(indir) as it:

        # sort the files in alphabetical order so the video
        # appears in sequential order
        it = list(it)
        it.sort(key=lambda x: x.name)

        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():

                # Read in the image
                src = cv2.imread(indir + "/" + entry.name)

                # for each frame, go through the dictionary and find
                # the corresponding bounding boxes for each frame
                # Then, draw them on the frame
                boxes = bb[framenum]
                for box in boxes:
                    cv2.rectangle(src, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,255), 2)
                    cv2.putText(src, str(box[4]), (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0,0,255),
                                2)

                cv2.imshow("Ground Truth", src)
                cv2.waitKey(30)
                cv2.imwrite("{}/{:06d}.jpg".format(outdir,framenum), src)

                framenum += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
