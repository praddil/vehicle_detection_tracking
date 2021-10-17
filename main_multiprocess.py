import numpy as np
import time
import cv2
import copy
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from Detection import Detectors
from Track import Tracker

from multiprocessing import Process, Queue


def rescale_frame(input_frame):
    """
    Args:
        input_frame: Frames to be resized
    Returns: resized frames
    """
    width = 1280
    height = 720
    dim = (width, height)
    return cv2.resize(input_frame, dim, interpolation=cv2.INTER_AREA)


def videoCapture(captureQueue):
    # Open the input file for processing
    cap = cv2.VideoCapture('config/video/MVI_1991.MP4')
    cap.set(3, 1920)
    cap.set(4, 1080)
    # Condition if file not opened
    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        captureQueue.put(cap.read())

    cap.release()
    return True


def main():

    # Fixed width and height for the video frame
    width = 1280
    height = 720
    miles_in_pixel = 0.0052     # Approximate miles horizontally in detection zone
    frame_no = 0
    # font type
    font = cv2.FONT_HERSHEY_PLAIN
    trace_color = [(255, 0, 0)]

    # video output parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # From Darknet load YOLOweights for object recognition
    net = cv2.dnn.readNet("config/src/yolov3.weights", "config/src/yolov3.cfg")  # Original yolov3

    # From Darknet use the names of the object to recognize
    with open("config/data/coco.names", "r") as f:
        object_names = [line.strip() for line in f.readlines()]

    # Setting area for shaded polyygon area for detection zone and speed estimation
    upperBorder = int(height * 0.40)
    lowerBorder = int(height * 0.90)
    midBorder = int((upperBorder + lowerBorder) / 2)

    # Point of interest for speed calculation
    firstPoi = upperBorder + 20
    secondPoi = lowerBorder - 100
    midPoi = int((firstPoi + secondPoi)/2)

    # Coordinates of the detection zone
    first_cor = [540, upperBorder]
    sec_cor = [width-430, upperBorder]
    third_cor = [width-20, lowerBorder]
    fourth_cor = [220, lowerBorder]

    # Reshape Coordinates
    detectionZone = np.array([first_cor, sec_cor, third_cor, fourth_cor], np.int32)
    detectionZone = detectionZone.reshape((-1, 1, 2))

    # Detection zone
    zone = [tuple(first_cor), tuple(sec_cor), tuple(third_cor), tuple(fourth_cor)]

    # video file output
    videoOut = cv2.VideoWriter('output\\last_Attempt_MT.mp4', fourcc, 20.0, (width, height))

    # Create Detector Object
    detector = Detectors(net, object_names)

    # Create Tracker Object
    tracker = Tracker(20, 5, 15, 100)

    # Instantiate queues to hold video frames
    captureQueue = Queue()

    # Start Processes
    videoCapture_process = Process(target=videoCapture, args=(captureQueue,))
    videoCapture_process.start()

    # Not the best practice
    time.sleep(5)

    # Loop until frame Queue is empty
    while not captureQueue.empty():
        unused, frame = captureQueue.get()
        frame = rescale_frame(frame)

        # Draw horigontal line for speed estimation to see where the lines.
        # Displaying will cause issue with object detection
        # cv2.line(frame, (520, firstPoi), (width-400, firstPoi), (0, 0, 255), thickness=1)
        # cv2.line(frame, (390, midPoi), (width - 240, midPoi), (0, 255, 0), thickness=1)
        # cv2.line(frame, (260, secondPoi), (width-70, secondPoi), (0, 0, 255), thickness=1)

        # Capture centroids of any object within the zone
        centers = detector.Detect(frame, zone)

        # Process any centers detected
        if centers:

            # Update tracker with centroid's coordinate
            tracker.Update(centers, Polygon(zone))

            # Loop through all the detected tracks
            for i in range(len(tracker.tracks)):
                if len(tracker.tracks[i].trace) > 5:
                    x2, y2 = 0, 0

                    # Loop through object's centroids to draw trace line
                    for j in range(len(tracker.tracks[i].trace)-3):
                        x1 = tracker.tracks[i].trace[j+2][0][0]
                        y1 = tracker.tracks[i].trace[j+2][1][0]
                        x2 = tracker.tracks[i].trace[j+3][0][0]
                        y2 = tracker.tracks[i].trace[j+3][1][0]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), trace_color[0], 2)

                    # If the trace vector is within the zone, update the start_time
                    trace_i = len(tracker.tracks[i].trace) - 1
                    # trace_x = tracker.tracks[i].trace[trace_i][0][0]
                    trace_y = tracker.tracks[i].trace[trace_i][1][0]

                    if trace_y in range(firstPoi, secondPoi) and not tracker.tracks[i].speed_count_started:
                        tracker.tracks[i].initial_frame_count = frame_no
                        tracker.tracks[i].speed_count_started = True

                    # update as the object passes the mid of detection zone and calulcate speed
                    if midPoi + 5 >= trace_y >= midPoi - 5 and not tracker.tracks[i].passed_speed_checkpoint:
                        tracker.tracks[i].passed_speed_checkpoint = True

                        frame_length = frame_no - tracker.tracks[i].initial_frame_count

                        time_dur = (1/ 29) * frame_length
                        if time_dur > 0:
                            time_dur /= (60 * 60)       # Convert second to hour

                        try:
                            print(round(miles_in_pixel/time_dur))
                            tracker.tracks[i].speed = round(miles_in_pixel/time_dur)
                        except:
                            pass

                    # Optional condition to irradicate unusual readings
                    if 0 < tracker.tracks[i].speed < 100:
                        cv2.putText(frame, str(tracker.tracks[i].track_id) + ' mph:' + str(tracker.tracks[i].speed),
                                    (int(x2), int(y2)), font, 1, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame, str(tracker.tracks[i].track_id), (int(x2), int(y2)), font, 1, (255, 255, 255), 2)

            # Make copy of original frame
            orig_frame = frame.copy()
            cv2.fillPoly(orig_frame, [detectionZone], (0, 200, 0))

            # Blending the original frame with copy frame consisting of green shaded detection zone
            alpha = 0.3
            cv2.addWeighted(orig_frame, alpha, frame, 1 - alpha, 0, frame)

            # Display processed frame
            cv2.imshow('Processing', frame)

        # Write on the video file
        videoOut.write(frame)

        # Wait in Miliseconds for user prompt or Keyboard input
        cv2.waitKey(10)

        # wait for keyboard input to end the program
        user_key = cv2.waitKey(10) & 0xff
        # Esc key to exit
        if user_key == 27:
            captureQueue.close()
            videoCapture_process.terminate()
            break

        frame_no += 1

    videoOut.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
