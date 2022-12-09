# Author: Ifeanyi Ibeanusi, Villanova University
# Read Ground Truth Files
# CSC 5930/9010 - Computer Vision
# MOT Challenge
# Date Created: December 2, 2022

# implementing with an array

"""
To run example: python3 readgt_yolo_classes_2.py -g [directory containing gt files] --image [directory containing image files]
--config yolov3.cfg --weights yolov3.weights --classes yolov3.txt -pf [parent directory for gt and image files]
"""

import argparse
from collections import defaultdict
import csv
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
    ap.add_argument('-pf', '--parent_folder', required=True,
                    help='path to parent folder')
    args = ap.parse_args()
    return args


class DetectedObject:
    def __init__(self, name=None, color=None, location=None, class_id='person', missed_times=0,
                 faulty_tracking_times=0, show=True):
        """
        :param name: String
        :param color: List
        :param location: List
        """
        if color is None:
            color = np.random.uniform(0, 255, size=(1, 3))[0]
        if location is None:
            location = [0.0, 0.0, 0.0, 0.0]

        self.name = name
        self.color = color
        self.location = location
        self.class_id = class_id
        self.missed_times = missed_times
        self.faulty_tracking_times = faulty_tracking_times
        self.show = show


class DetectedObjects:
    def __init__(self, yolo, tracking=None, detected=None, ious=None, max_len=0, faulty_trackings=None):
        """
        :param yolo: YOLO

        """
        if tracking is None:
            tracking = []
        if faulty_trackings is None:
            faulty_trackings = []
        self.yolo = yolo
        self.tracking = tracking
        self.detected = detected  # output dictionary of DetectedObject
        self.max_len = max_len  # to track max number of objects
        self.ious = ious
        self.faulty_trackings = faulty_trackings

    @staticmethod
    def get_iou(bb1, bb2):
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

    @staticmethod
    def add_missing_object(detected_objects, object_before):
        """
        Add missing object from tracking objects array to detected objects array
        :param detected_objects: List of DetectedObject (detected)
        :param object_before: DetectedObject in tracking
        :return: None
        """
        missing_object = DetectedObject(name=object_before.name + str(object_before.missed_times), color=object_before.color,
                                        missed_times=1)
        detected_objects.append(missing_object)

    @staticmethod
    def add_new_object(tracking_objects, object_after):
        """
        Add new object from detected objects array to tracking objects array
        :param tracking_objects: List of DetectedObject (tracking)
        :param object_after: DetectedObject in detected
        :return: None
        """
        new_object = DetectedObject(name=object_after.name, color=object_after.color)
        tracking_objects.append(new_object)

    def get_ious_objects(self):
        """
        Function to get the ious for each detected object and tracking object
        :return: None
        """
        max_len = max(len(self.tracking), len(self.detected))
        ious = np.array([0] * max_len)
        for object_id_detected in range(max_len):
            detected_row = []
            sum_detected_row = 0
            for object_id_tracking in range(max_len):
                try:
                    iou = self.get_iou(self.detected[object_id_detected].location,
                                       self.tracking[object_id_tracking].location)
                except IndexError:
                    while len(self.detected) < len(self.tracking):
                        missed_object = len(self.detected)
                        # add missing object with increment missed_times
                        self.add_missing_object(detected_objects=self.detected,
                                                object_before=self.tracking[missed_object])
                    while len(self.tracking) < len(self.detected):
                        new_object = len(self.tracking)
                        # add new object
                        self.add_new_object(tracking_objects=self.tracking,
                                            object_after=self.detected[new_object])
                    iou = self.get_iou(self.detected[object_id_detected].location,
                                       self.tracking[object_id_tracking].location)
                sum_detected_row += iou
                detected_row.append(iou)
            ious = np.vstack([ious, detected_row])
        ious = ious[1:]
        # if column sums to 0
        tracking_sums = ious.sum(axis=0)
        for column, sum_column in enumerate(tracking_sums):
            if sum_column == 0:
                self.faulty_trackings.append(column)
                self.tracking[column].faulty_tracking_times += 1

        self.ious = ious

    def update_detected(self):
        """
        Update the detected array with list of DetectedObjects with new YOLO detections
        :return: None
        """
        self.detected = YOLO.yolo_detector(self.yolo)

    def update_tracking(self):
        """
        Update the tracking array with list of DetectedObjects with new YOLO detections
        :return: None
        """
        self.tracking = YOLO.yolo_detector(self.yolo)

    def update_ious(self):
        """
        Update the ious attribute for DetectedObjects
        :return: None
        """
        self.get_ious_objects()

    def update_detected_property(self, property, detected_key, tracking_key):
        """
        Menu for updating detected DetectedObject property
        :param property:  location, color, name, class_id
        :param detected_key: key of detected objected to update
        :param tracking_key: key of tracking objected to update
        :return: None
        """
        if property == "location":
            self.detected[detected_key].location = self.tracking[tracking_key].location
        if property == "color":
            self.detected[detected_key].color = self.tracking[tracking_key].color
        if property == "name":
            self.detected[detected_key].name = self.tracking[tracking_key].name
        if property == "class_id":
            self.detected[detected_key].class_id = self.tracking[tracking_key].class_id
        if property == "missed_times":
            self.detected[detected_key].missed_times = self.tracking[tracking_key].missed_times

    def update_tracking_property(self, property, detected_key, tracking_key):
        """
        Tracking stores the new location of detected after Hungarian method
        :param property: the property to change in DetectedObject
        :param detected_key: key of detected objected to update
        :param tracking_key: key of tracking objected to update
        :return: None
        """
        if property == "location":
            self.tracking[tracking_key].location = self.detected[detected_key].location
        if property == "color":
            self.tracking[tracking_key].color = self.detected[detected_key].color
        if property == "name":
            self.tracking[tracking_key].name = self.detected[detected_key].name
        if property == "class_id":
            self.tracking[tracking_key].class_id = self.detected[detected_key].class_id
        if property == "missed_times":
            if self.detected[detected_key].missed_times == 0:
                self.tracking[tracking_key].missed_times = 0
            self.tracking[tracking_key].missed_times += self.detected[detected_key].missed_times
        if property == "show":
            if self.tracking[tracking_key].faulty_tracking_times == 0:
                self.tracking[tracking_key].show = True

    def hungarian_update_detected(self, hungarian_object):
        """
        Map detected DetectedObject with tracking DetectedObject to ensure that bounding box stays on the same person.
        Use Hungarian algorithm to find closest map
        :param hungarian_object: Hungarian
        :return: None
        """
        hungarian_object.calculate()
        result = hungarian_object.get_results()

        for pair in result:  # update detected

            self.update_detected_property("name", pair[0], pair[1])
            self.update_detected_property("color", pair[0], pair[1])

            # if missed object
            if self.detected[pair[0]].missed_times > 0 and \
                    self.tracking[pair[1]].missed_times <= 3 or \
                    (pair[1] in self.faulty_trackings and
                     self.tracking[pair[1]].faulty_tracking_times <= 3):
                self.update_detected_property("location", pair[0], pair[1])

            if self.tracking[pair[1]].faulty_tracking_times > 3:
                self.tracking[pair[1]].show = False

            self.update_tracking_property("location", pair[0], pair[1])
            self.update_tracking_property("missed_times", pair[0], pair[1])
            #self.update_tracking_property("show", pair[0], pair[1])

    def send_to_csv_file(self, csv_path, frame_num):
        """
        Function to store the data for each object in csv file
        :return: None
        """

        if frame_num == 1:  # overwrite if first frame
            with open(csv_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                for detected_object in self.detected:
                    if detected_object.show:
                        row = [frame_num, int(detected_object.name) + 1,
                               int(detected_object.location[0]),
                               int(detected_object.location[1]),
                               int(detected_object.location[2]),
                               int(detected_object.location[3]), 1, -1, -1, -1]
                        # write the header
                        writer.writerow(row)
        else:
            with open(csv_path, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)

                for detected_object in self.detected:
                    if detected_object.show:
                        row = [frame_num, int(detected_object.name) + 1,
                               int(detected_object.location[0]),
                               int(detected_object.location[1]),
                               int(detected_object.location[2]),
                               int(detected_object.location[3]), 1, -1, -1, -1]
                        # write the header
                        writer.writerow(row)


# used for YOLO
class YOLO:
    def __init__(self, src, args):
        """
        Input is a source image and arguments from terminal
        """
        self.image = src.copy()
        self.args = args

    @staticmethod
    def get_output_layers(net):
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
    def yolo_detector(self):
        """
        YOLO detector outputs a list of detected objects in the frame
        :return: detected_objects list of DetectedObject
        """

        image = self.image

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

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
        detected_objects = []

        for object_ID, i in enumerate(indices):
            if classes[class_ids[i]] != 'person':
                continue
            object_detected = DetectedObject(name=str(object_ID))
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            object_detected.location = box
            object_detected.class_id = classes[class_ids[i]]
            detected_objects.append(object_detected)
        return detected_objects

        # draw bounding boxes

    def draw_prediction(self, objects_detected, name_window="detected", file_dir="img1_output", frame_num=0, save=True):
        """
        Draw the predicted bounding boxes from objects_detected in image
        :param save: Boolean (save image or not)
        :param frame_num: Int
        :param file_dir: String
        :param objects_detected: Dict
        :param name_window: String
        :return: None
        """
        img = self.image.copy()
        for object_detected in objects_detected:
            curr_object = object_detected
            label = str(curr_object.class_id)
            show = curr_object.show
            if show and label == 'person':
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
        if save:
            self.store_frames(img, file_dir, frame_num)
        cv2.imshow(name_window, img)

    @staticmethod
    def store_frames(img, file_dir, frame_num):
        """
        Store image frames in a folder
        :param img: Numpy Array
        :param file_dir: String
        :param frame_num: Int
        :return: None
        """
        cv2.imwrite("{}/{:06d}.jpg".format(file_dir, frame_num), img)


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
            my_matrix = np.pad(my_matrix, ((0, pad_columns), (0, pad_rows)), 'constant', constant_values=0)

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
    def __init__(self, yolo, covariances, detected_objects, means=None):
        if means is None:
            means = defaultdict(lambda: [])
        self.yolo = yolo
        self.detected_objects = detected_objects
        self.means = means
        self.covariances = covariances
        # self.mean = mean # state vector [ cx, cy, w, h, vx, vy, vw, vh ]
        # self.covariance = covariance # uncertainty matrix - larger number larger uncertainty

    def predict(self):
        # populate mean
        for detected_object_id in self.detected_objects:
            mean = [0.0] * 8
            cx = self.detected_objects[detected_object_id].location[0] + \
                 (self.detected_objects[detected_object_id].location[2] / 2)
            cy = self.detected_objects[detected_object_id].location[1] + \
                 (self.detected_objects[detected_object_id].location[3] / 2)
            w = self.detected_objects[detected_object_id].location[2]
            h = self.detected_objects[detected_object_id].location[3]
            # all velocities set to 0

            mean[0] = cx
            mean[1] = cy
            mean[2] = w
            mean[3] = h

            self.means[detected_object_id] = mean

        pass

    def update(self):
        pass


def main():
    args = parser()

    if len(sys.argv) < 3:
        print(f'{sys.argv[0]} [groundtruth] [img dir]')
        sys.exit()

    dir = args.image
    parent_folder = args.parent_folder

    framenum = 1
    with os.scandir(dir) as it:

        # sort the files in alphabetical order so the video
        # appears in sequential order
        it = list(it)
        it.sort(key=lambda x: x.name)
        first_frame = cv2.imread(dir + "/" + it[0].name)

        first_frame_yolo = YOLO(first_frame, args)  # the first YOLO object for first frame
        detected_objects = DetectedObjects(
            yolo=first_frame_yolo)  # initialize list of detected objects based off of yolo detection

        first_frame_yolo.image = first_frame.copy()  # reset the image
        detected_objects.update_detected()
        detected_objects.update_tracking()

        # objects contains tracking list (t-1)

        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():

                # Read in the image
                src = cv2.imread(dir + "/" + entry.name)

                yolo = YOLO(src, args)
                detected_objects = DetectedObjects(yolo, tracking=detected_objects.tracking)

                detected_objects.update_detected()

                detected_objects.update_ious()
                ious = detected_objects.ious

                # call Hungarian algorithm here with IOUs
                hungarian = Hungarian(ious, is_profit_matrix=True)
                detected_objects.hungarian_update_detected(hungarian)

                # store detections in csv file

                yolo.draw_prediction(detected_objects.detected, frame_num=framenum, save=False)
                # yolo.draw_prediction(detected_objects.detected, "debug" + str(framenum), frame_num=framenum,
                # save=False)  # debug

                # detected_objects.send_to_csv_file(str(parent_folder) + "/My_detection.txt", framenum)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

                framenum += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
