# Author: Ifeanyi Ibeanusi, Villanova University
# Read Ground Truth Files
# CSC 5930/9010 - Computer Vision
# MOT Challenge
# Date Created: December 2, 2021
import argparse
import math
from collections import defaultdict
import random
import hungarian

import cv2
import numpy as np
import sys
import os


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--groundtruth', required=True,
                    help='path to ground truth file')
    ap.add_argument('-i', '--image', required=True,
                    help='path to image file')
    ap.add_argument('-c', '--config', required=True,
                    help='path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help='path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help='path to text file containing class names')
    args = ap.parse_args()
    return args


class DetectedObject:
    def __init__(self, name="0", color=None, location=None, img=None, class_id=None, missed_times=0):
        """
        :param name: String
        :param color: List
        :param location: List
        :param img: List[List]
        """
        if color is None:
            color = np.random.uniform(0, 255, size=(1, 3))[0]
        if location is None:
            location = [0.0, 0.0, 0.0, 0.0]
        self.name = name
        self.color = color
        self.location = location
        self.img = img
        self.class_id = class_id
        self.missed_times = missed_times  # don't know what to do with missed_times yet


class DetectedObjects:
    def __init__(self, yolo, detected=None, ious=None, max_len=0):
        """
        :param yolo: YOLO

        """
        self.yolo = yolo
        self.detected = detected  # output dictionary of DetectedObject
        self.max_len = max_len  # to track max number of objects
        self.ious = ious

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : list to dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : list to dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        # if bb1 and bb2 is of type list of [x, y, w, h] convert to dict
        bb1_dict = {'x1': bb1[0], 'x2': bb1[0] + bb1[2], 'y1': bb1[1], 'y2': bb1[1] + bb1[3]}
        bb2_dict = {'x1': bb2[0], 'x2': bb2[0] + bb2[2], 'y1': bb2[1], 'y2': bb2[1] + bb2[3]}

        assert bb1_dict['x1'] <= bb1_dict['x2']
        assert bb1_dict['y1'] <= bb1_dict['y2']
        assert bb2_dict['x1'] <= bb2_dict['x2']
        assert bb2_dict['y1'] <= bb2_dict['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1_dict['x1'], bb2_dict['x1'])
        y_top = max(bb1_dict['y1'], bb2_dict['y1'])
        x_right = min(bb1_dict['x2'], bb2_dict['x2'])
        y_bottom = min(bb1_dict['y2'], bb2_dict['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1_dict['x2'] - bb1_dict['x1']) * (bb1_dict['y2'] - bb1_dict['y1'])
        bb2_area = (bb2_dict['x2'] - bb2_dict['x1']) * (bb2_dict['y2'] - bb2_dict['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        try:  # for zero devision error
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        except ZeroDivisionError:
            iou = 0.0
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def update_detected(self):
        self.detected = YOLO.YOLO_detector(self.yolo).copy()

    def get_ious_objects(self):
        """
        Function to get the ious for each detected object and tracking object
        :param tracking: dict with objects from previous frame
        :param detected: dict with objects from current frame
        :return: 2d array with ious
        """
        min_len = min(len(self.tracking), len(self.detected))
        ious = []
        for object_id_detected in range(min_len):
            detected_row = []
            sum_detected_row = 0
            for object_id_tracking in range(min_len):
                iou = self.get_iou(self.detected[object_id_detected].location,
                                   self.tracking[object_id_tracking].location)
                sum_detected_row += iou
                detected_row.append(iou)
            if sum_detected_row == 0:
                self.detected[object_id_detected].missed_times += 1
            else:
                self.detected[object_id_detected].missed_times = 0
            ious.append(detected_row)
        self.ious = ious


# used for YOLO
class YOLO:
    def __init__(self, src, args):
        """
        Input is a source image and arguments from terminal
        """
        self.image = src.copy()
        self.gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self.args = args

    def get_output_layers(self, net):
        """
        :param net: {getLayerNames, getUnconnectedOutLayers}
        :return: List
        """
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    # output array of boxes with box being [x, y, w, h]
    def YOLO_detector(self):
        """
        YOLO detector receives an input image called src and outputs a list of detected objects
        It also stores the image of the object in an array: object_img
        :param args: parser
        :param src: List[List]
        :return: objects
        """
        image = self.image.copy()
        src_gray = self.gray_image

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(self.args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        net = cv2.dnn.readNet(self.args.weights, self.args.config)  # object detection for openCV dnn

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(self.get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # outs is a 2d matrix
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]  # going to scores of 25/26
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        object_img = []  # array to store object image
        detected_objects = {}

        for object_ID, i in enumerate(indices):
            # assign name and color to object
            # object_detected = DetectedObject(name=classes[class_ids[i]] + str(object_ID),
            # color=np.random.uniform(0, 255, size=(1, 3))[0])
            if classes[class_ids[i]] != 'person':
                continue
            object_detected = DetectedObject(name=str(object_ID))
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            object_detected.location = box

            # I will be tracking object in objects
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            # get the initial images turn into binary and store in array
            box_img = self.apply_brightness_contrast(image[y:y + h, x:x + w], contrast=100)
            box_img_gray = self.apply_brightness_contrast(src_gray[y:y + h, x:x + w], contrast=100)
            (T, binary) = cv2.threshold(box_img_gray, 100, 255, cv2.THRESH_BINARY)

            object_detected.img = binary

            cv2.imshow("object image " + str(object_detected.name), object_detected.img)
            object_detected.class_id = classes[class_ids[i]]
            object_img.append(src_gray[y:y + h, x:x + w])

            detected_objects[object_ID] = object_detected

        return detected_objects

        # draw bounding boxes

    def apply_brightness_contrast(self, input_img, brightness=0, contrast=0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def draw_prediction(self, objects_detected, name_window="detected"):
        """
        :param objects_detected: Dict
        :param name_window: String
        :return:
        """
        img = self.image.copy()
        for object_Id in objects_detected:
            # label = str(classes[class_id]) remove
            curr_object = objects_detected[object_Id]
            label = str(curr_object.class_id)
            missed_times = curr_object.missed_times
            if label == 'person' and missed_times < 3:
                color = curr_object.color
                x = round(curr_object.location[0])
                y = round(curr_object.location[1])
                w = round(curr_object.location[2])
                h = round(curr_object.location[3])
                x_plus_w = x + w
                y_plus_h = y + h
                cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

                cv2.putText(img, label + curr_object.name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color,
                            2)
        cv2.imshow(name_window, img)


class Tracker:
    def __init__(self, image, detected):
        self.img = image
        self.gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.detected = detected

    def track_object(self, detected_object):
        """
        Tracks an object specified in detected_object
        """

        roi = detected_object.location
        gray_img = self.gray_img

        # get the region of interest coordinates
        roi_column = int(roi[0])
        roi_row = int(roi[1])
        roi_column_width = int(roi[2])
        roi_row_height = int(roi[3])

        # offset to start search from founding position
        offset = 3

        # threshold to ensure tracker moves only when there is a major movement of object
        threshold = 0.0001

        min_score = math.inf

        # iterate through neighboring boxes
        for col in range(roi_column - offset, roi_column + offset):
            for row in range(roi_row - offset, roi_row + offset):
                compare_img = gray_img[row: row + roi_row_height,
                              col: col + roi_column_width]
                compare_score = cv2.matchShapes(detected_object.img, compare_img, cv2.CONTOURS_MATCH_I1, 0)
                if min_score > compare_score:
                    min_score = compare_score
                    print(min_score)
                    best_match_contour = (col, row, roi_column_width, roi_row_height)

        newImg = self.yolo.copy()
        if min_score >= threshold:
            roi = best_match_contour
        cv2.rectangle(newImg, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

        # display the image
        cv2.imshow("frame", newImg)

    def track_objects(self):
        """
        Run track object in each instance of DetectedObject
        :return:
        """
        for detected_object in self.detected:
            self.track_object(detected_object)


class HungarianError(Exception):
    pass


class Hungarian:
    """
    Implementation of the Hungarian (Munkres) Algorithm using np.
    Usage:
        hungarian = Hungarian(cost_matrix)
        hungarian.calculate()
    or
        hungarian = Hungarian()
        hungarian.calculate(cost_matrix)
    Handle Profit matrix:
        hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    or
        cost_matrix = Hungarian.make_cost_matrix(profit_matrix)
    The matrix will be automatically padded if it is not square.
    For that numpy's resize function is used, which automatically adds 0's to any row/column that is added
    Get results and total potential after calculation:
        hungarian.get_results()
        hungarian.get_total_potential()
    """

    def __init__(self, input_matrix=None, is_profit_matrix=False):
        """
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        if input_matrix is not None:
            # Save input
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxColumn = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            # Adds 0s if any columns/rows are added. Otherwise stays unaltered
            matrix_size = max(self._maxColumn, self._maxRow)
            pad_columns = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxColumn
            my_matrix = np.pad(my_matrix, ((0, pad_columns), (0, pad_rows)), 'constant', constant_values=(0))

            # Convert matrix to profit matrix if necessary
            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)

            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            # Results from algorithm.
            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def get_results(self):
        """Get results after calculation."""
        return self._results

    def get_total_potential(self):
        """Returns expected value after calculation."""
        return self._totalPotential

    def calculate(self, input_matrix=None, is_profit_matrix=False):
        """
        Implementation of the Hungarian (Munkres) Algorithm.
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        # Handle invalid and new matrix inputs.
        if input_matrix is None and self._cost_matrix is None:
            raise HungarianError("Invalid input")
        elif input_matrix is not None:
            self.__init__(input_matrix, is_profit_matrix)

        result_matrix = self._cost_matrix.copy()

        # Step 1: Subtract row mins from each row.
        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        # Step 2: Subtract column mins from each column.
        for index, column in enumerate(result_matrix.T):
            result_matrix[:, index] -= column.min()

        # Step 3: Use minimum number of lines to cover all zeros in the matrix.
        # If the total covered rows+columns is not equal to the matrix size then adjust matrix and repeat.
        total_covered = 0
        while total_covered < self._size:
            # Find minimum number of lines to cover all zeros in the matrix and find total covered rows and columns.
            # print("result_matrix is")
            # print(result_matrix)
            cover_zeros = CoverZeros(result_matrix)
            covered_rows = cover_zeros.get_covered_rows()
            covered_columns = cover_zeros.get_covered_columns()
            total_covered = len(covered_rows) + len(covered_columns)

            # if the total covered rows+columns is not equal to the matrix size then adjust it by min uncovered num (m).
            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_columns)

        # Step 4: Starting with the top row, work your way downwards as you make assignments.
        # Find single zeros in rows or columns.
        # Add them to final result and remove them and their associated row/column from the matrix.
        expected_results = min(self._maxColumn, self._maxRow)
        zero_locations = (result_matrix == 0)
        while len(self._results) != expected_results:

            # If number of zeros in the matrix is zero before finding all the results then an error has occurred.
            if not zero_locations.any():
                raise HungarianError("Unable to find results. Algorithm has failed.")

            # Find results and mark rows and columns for deletion
            matched_rows, matched_columns = self.__find_matches(zero_locations)

            # Make arbitrary selection
            total_matched = len(matched_rows) + len(matched_columns)
            if total_matched == 0:
                matched_rows, matched_columns = self.select_arbitrary_match(zero_locations)

            # Delete rows and columns
            for row in matched_rows:
                zero_locations[row] = False
            for column in matched_columns:
                zero_locations[:, column] = False

            # Save Results
            self.__set_results(zip(matched_rows, matched_columns))

        # Calculate total potential
        value = 0
        for row, column in self._results:
            value += self._input_matrix[row, column]
        self._totalPotential = value

    @staticmethod
    def make_cost_matrix(profit_matrix):
        """
        Converts a profit matrix into a cost matrix.
        Expects NumPy objects as input.
        """
        # subtract profit matrix from a matrix made of the max value of the profit matrix
        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix
        return cost_matrix

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_columns):
        """Subtract m from every uncovered number and add m to every element covered with two lines."""
        # Calculate minimum uncovered number (m)
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        elements.append(element)
        min_uncovered_num = min(elements)

        # Add m to every covered element
        adjusted_matrix = result_matrix
        for row in covered_rows:
            adjusted_matrix[row] += min_uncovered_num
        for column in covered_columns:
            adjusted_matrix[:, column] += min_uncovered_num

        # Subtract m from every element
        m_matrix = np.ones(self._shape, dtype=int) * min_uncovered_num
        adjusted_matrix -= m_matrix

        return adjusted_matrix

    def __find_matches(self, zero_locations):
        """Returns rows and columns with matches in them."""
        marked_rows = np.array([], dtype=int)
        marked_columns = np.array([], dtype=int)

        # Mark rows and columns with matches
        # Iterate over rows
        for index, row in enumerate(zero_locations):
            row_index = np.array([index])
            if np.sum(row) == 1:
                column_index, = np.where(row)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        # Iterate over columns
        for index, column in enumerate(zero_locations.T):
            column_index = np.array([index])
            if np.sum(column) == 1:
                row_index, = np.where(column)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        return marked_rows, marked_columns

    @staticmethod
    def __mark_rows_and_columns(marked_rows, marked_columns, row_index, column_index):
        """Check if column or row is marked. If not marked then mark it."""
        new_marked_rows = marked_rows
        new_marked_columns = marked_columns
        if not (marked_rows == row_index).any() and not (marked_columns == column_index).any():
            new_marked_rows = np.insert(marked_rows, len(marked_rows), row_index)
            new_marked_columns = np.insert(marked_columns, len(marked_columns), column_index)
        return new_marked_rows, new_marked_columns

    @staticmethod
    def select_arbitrary_match(zero_locations):
        """Selects row column combination with minimum number of zeros in it."""
        # Count number of zeros in row and column combinations
        rows, columns = np.where(zero_locations)
        zero_count = []
        for index, row in enumerate(rows):
            total_zeros = np.sum(zero_locations[row]) + np.sum(zero_locations[:, columns[index]])
            zero_count.append(total_zeros)

        # Get the row column combination with the minimum number of zeros.
        indices = zero_count.index(min(zero_count))
        row = np.array([rows[indices]])
        column = np.array([columns[indices]])

        return row, column

    def __set_results(self, result_lists):
        """Set results during calculation."""
        # Check if results values are out of bound from input matrix (because of matrix being padded).
        # Add results to results list.
        for result in result_lists:
            row, column = result
            if row < self._maxRow and column < self._maxColumn:
                new_result = (int(row), int(column))
                self._results.append(new_result)


class CoverZeros:
    """
    Use minimum number of lines to cover all zeros in the matrix.
    Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
    """

    def __init__(self, matrix):
        """
        Input a matrix and save it as a boolean matrix to designate zero locations.
        Run calculation procedure to generate results.
        """
        # Find zeros in matrix
        self._zero_locations = (matrix == 0)
        self._shape = matrix.shape

        # Choices starts without any choices made.
        self._choices = np.zeros(self._shape, dtype=bool)

        self._marked_rows = []
        self._marked_columns = []

        # marks rows and columns
        self.__calculate()

        # Draw lines through all unmarked rows and all marked columns.
        self._covered_rows = list(set(range(self._shape[0])) - set(self._marked_rows))
        self._covered_columns = self._marked_columns

    def get_covered_rows(self):
        """Return list of covered rows."""
        return self._covered_rows

    def get_covered_columns(self):
        """Return list of covered columns."""
        return self._covered_columns

    def __calculate(self):
        """
        Calculates minimum number of lines necessary to cover all zeros in a matrix.
        Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
        """
        while True:
            # Erase all marks.
            # print("erase all marks...")
            self._marked_rows = []
            self._marked_columns = []

            # Mark all rows in which no choice has been made.
            for index, row in enumerate(self._choices):
                if not row.any():
                    # print("mark row "+ str(index))
                    self._marked_rows.append(index)

            # If no marked rows then finish.
            if not self._marked_rows:
                # print("finish since no marked rows")
                # print(self._marked_rows)
                # print(self._marked_columns)

                return True

            # Mark all columns not already marked which have zeros in marked rows.
            # print("Mark all columns not already marked which have zeros in marked rows.")
            num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

            # If no new marked columns then finish.
            if num_marked_columns == 0:
                # print("finish since no marked columns")
                # print(self._marked_rows)
                # print(self._marked_columns)

                return True

            # While there is some choice in every marked column.
            while self.__choice_in_all_marked_columns():
                # print("when there is some choice in every marked column")
                # Some Choice in every marked column.

                # Mark all rows not already marked which have choices in marked columns.
                num_marked_rows = self.__mark_new_rows_with_choices_in_marked_columns()

                # If no new marks then Finish.
                if num_marked_rows == 0:
                    # print("finish since no new marked row")
                    # print(self._marked_rows)
                    # print(self._marked_columns)

                    return True

                # Mark all columns not already marked which have zeros in marked rows.
                num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

                # If no new marked columns then finish.
                if num_marked_columns == 0:
                    # print("finish since no new marked columns")
                    # print(self._marked_rows)
                    # print(self._marked_columns)

                    return True

            # No choice in one or more marked columns.
            # print("No choice in some marked columns now")
            # Find a marked column that does not have a choice.
            choice_column_index = self.__find_marked_column_without_choice()
            # print("choice:")
            # print(self._choices)

            while choice_column_index is not None:
                # print(str(choice_column_index) + " marked column does not have a choice")
                # Find a zero in the column indexed that does not have a row with a choice.
                choice_row_index = self.__find_row_without_choice(choice_column_index)
                # print(str(choice_row_index)+" row does not have a choice")

                # Check if an available row was found.
                new_choice_column_index = None
                if choice_row_index is None:
                    # Find a good row to accomodate swap. Find its column pair.
                    choice_row_index, new_choice_column_index = \
                        self.__find_best_choice_row_and_new_column(choice_column_index)

                    # Delete old choice.
                    self._choices[choice_row_index, new_choice_column_index] = False
                    # print("delete old choice at "+str(choice_row_index)+","+str(new_choice_column_index))

                # Set zero to choice.
                self._choices[choice_row_index, choice_column_index] = True
                # print("set choice at " + str(choice_row_index) + "," + str(choice_column_index))

                # Loop again if choice is added to a row with a choice already in it.
                choice_column_index = new_choice_column_index
                # print("now for the column "+str(choice_column_index))

    def __mark_new_columns_with_zeros_in_marked_rows(self):
        """Mark all columns not already marked which have zeros in marked rows."""
        num_marked_columns = 0
        for index, column in enumerate(self._zero_locations.T):
            if index not in self._marked_columns:
                if column.any():
                    row_indices, = np.where(column)
                    zeros_in_marked_rows = (set(self._marked_rows) & set(row_indices)) != set([])
                    if zeros_in_marked_rows:
                        # print("mark column "+str(index))
                        self._marked_columns.append(index)
                        num_marked_columns += 1
        # print(str(num_marked_columns)+" columns marked")
        return num_marked_columns

    def __mark_new_rows_with_choices_in_marked_columns(self):
        """Mark all rows not already marked which have choices in marked columns."""
        num_marked_rows = 0
        for index, row in enumerate(self._choices):
            if index not in self._marked_rows:
                if row.any():
                    column_index, = np.where(row)
                    if column_index in self._marked_columns:
                        # print("mark row "+str(index))
                        self._marked_rows.append(index)
                        num_marked_rows += 1
        # print(str(num_marked_rows)+" row marked")
        return num_marked_rows

    def __choice_in_all_marked_columns(self):
        """Return Boolean True if there is a choice in all marked columns. Returns boolean False otherwise."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return False
        return True

    def __find_marked_column_without_choice(self):
        """Find a marked column that does not have a choice."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return column_index

        raise HungarianError(
            "Could not find a column without a choice. Failed to cover matrix zeros. Algorithm has failed.")

    def __find_row_without_choice(self, choice_column_index):
        """Find a row without a choice in it for the column indexed. If a row does not exist then return None."""
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            if not self._choices[row_index].any():
                return row_index

        # All rows have choices. Return None.
        return None

    def __find_best_choice_row_and_new_column(self, choice_column_index):
        """
        Find a row index to use for the choice so that the column that needs to be changed is optimal.
        Return a random row and column if unable to find an optimal selection.
        """
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            column_indices, = np.where(self._choices[row_index])
            column_index = column_indices[0]
            if self.__find_row_without_choice(column_index) is not None:
                return row_index, column_index

        # Cannot find optimal row and column. Return a random row and column.
        from random import shuffle

        shuffle(row_indices)
        column_index, = np.where(self._choices[row_indices[0]])
        return row_indices[0], column_index[0]


class KalmanFilter:
    def __init__(self):
        pass


def main():
    global src, previous_objects
    args = parser()

    if len(sys.argv) < 3:
        print(f'{sys.argv[0]} [groundtruth] [img dir]')
        sys.exit()

    groundtruth = args.groundtruth
    dir = args.image

    # Read in the file and get all of the bounding box data
    winname = "output"
    cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)
    frame = []

    framenum = 1
    with os.scandir(dir) as it:

        # sort the files in alphabetical order so the video
        # appears in sequential order
        it = list(it)
        it.sort(key=lambda x: x.name)
        first_frame = cv2.imread(dir + "/" + it[0].name)

        first_frame_yolo = YOLO(first_frame, args)  # the first YOLO object for first frame
        # contains self.detected_objects = {}
        detected_objects = DetectedObjects(
            yolo=first_frame_yolo)  # initialize first of detected objects based off of yolo detection

        first_frame_yolo.image = first_frame.copy()  # reset the image
        detected_objects.update_detected()

        first_frame_yolo.draw_prediction(detected_objects.detected)
        # objects contains tracking list (t-1)

        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():

                # Read in the image
                src = cv2.imread(dir + "/" + entry.name)

                # I have my detected objects thanks to yolo

                # each object has an image stored in them
                # the image has to be a contour
                # compare the image stored in them with the image of the neighbors of the object with central moments

                # if central moment output the best make that the new location of the object
                # to account for occlusion need to call Hungarian algorithm between detectors to decide which detector
                # is correct
                # assign detector at pair[0] the properties of detector at pair[1]

                # detected_objects.update_tracking_locations()

                key = cv2.waitKey(1000)
                if key == ord("q"):
                    break

                framenum += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
