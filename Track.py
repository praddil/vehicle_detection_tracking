'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan

    Updated           : For Project Traffic vehicle tracking
    Updated by        : Mandil Pradhan
    Updated Brief     : The original file is updated to include an additional
                        functionality to delete the tracks if it reaches out of bounds.
                        And some variables are changed to match the Traffic project variables
'''

import numpy as np
from kalman_filter import KalmanFilter
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Track(object):

    def __init__(self, detections, trackIdCount):
        """
        Args:
            detections: Detected centroids
            trackIdCount:   Id of the tracked object
        """
        self.track_id = trackIdCount                # Id for the detected trace
        self.KF = KalmanFilter()                    # KF instance for the object
        self.prediction = np.asarray(detections)
        self.skipped_frames = 0
        self.trace = []
        self.initial_frame_count = None
        self.passed_speed_checkpoint = False
        self.speed_count_started = False
        self.speed = 0


class Tracker(object):

    def __init__(self, dist_thresh, max_no_undetectedFrames, max_trace_length,
                 trackIdCount):

        self.dist_thresh = dist_thresh
        self.undetected_frames_thres = max_no_undetectedFrames
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.id = []
        self.trackIdCount = trackIdCount

    def Update(self, detections, area):
        """
        Args:
            detections: Detected centroids in a frame to corresponds with objects
            area: Polygon area of detection

        Returns:
            None
        """

        if not self.tracks:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.id.append(self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))
        for i in range(len(self.tracks)):

            # Loop thru all the detection to find the distance between each trace from last frame to all new detection
            for j in range(len(detections)):
                try:
                    # Distance calulation two centroids (x, y)
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    # Save distance of each combination in an array
                    cost[i][j] = distance

                except:
                    pass

        # Let's average the squared ERROR
        cost = 0.5 * cost
        assignment = []

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        un_assigned_tracks = []

        for i in range(len(assignment)):
            if len(self.tracks[i].trace) < 1:
                self.dist_thresh = 300
            elif len(self.tracks[i].trace) < 2:
                self.dist_thresh = 200
            elif len(self.tracks[i].trace) < 3:
                self.dist_thresh = 100
            else:
                self.dist_thresh = 60

            if assignment[i] != -1:
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.undetected_frames_thres:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for del_id in del_tracks:
                if del_id < len(self.tracks):
                    del self.tracks[del_id]
                    del self.id[del_id]
                    del assignment[del_id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.id.append(self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):

            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1)

            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)

            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

        # Delete id that has reached out of the detection zone
        anyDel = 0
        for o in range(len(self.tracks)-1):
            a = len(self.tracks[o-anyDel].trace)
            if a > 4:
                center = self.tracks[o-anyDel].trace[a - 1]
                initial = self.tracks[o-anyDel].trace[0]
                if initial[1] > center[1]:
                    point = Point(center[0], center[1] - 1)
                else:
                    point = Point(center[0], center[1] + 1)

                if area.contains(point):
                    pass
                else:
                    del self.tracks[o-anyDel]
                    del self.id[o-anyDel]
                    anyDel += 1
