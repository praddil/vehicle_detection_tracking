import numpy as np
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Detectors(object):
    """
    This class detects the object in the zone within the frame
    """
    def __init__(self, net, objects_name):
        """
        Declare and initialize variables
        Args:
            net: Cv2 dnn yolo weights used to identify objects
            objects_name: List of names of the identified objects
        """

        self.net = net
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)         # Intel GPU use (cv2.dnn.DNN_TARGET_OPENCL)
        self.layer_names = net.getLayerNames()
        self.outputlayers = [self.layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        self.names_list = objects_name
        self.frame_id = 0
        self.colors = np.random.uniform(0, 255, size=(len(self.names_list), 3))

    def Detect(self, frame, zone):

        self.frame_id += 1
        frame_height, frame_width, _ = frame.shape

        # Creates a blob image of each frame size 224×224, 416x416, 512x512
        # Higher Blob image size gives more detection but consume more resource
        # SwapRB ture because opencv uses BGR
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layers = self.net.forward(self.outputlayers)

        centroids = []      # Coordinates of objects center for each detection
        object_ids = []     # List of Id for each object in a frame
        confidences = []    # Confidence of detection for each object
        coordinates = []    # Box Coordinates for each detection

        for layer in output_layers:
            for detection in layer:

                scores = detection[5:]              # Get all possible scores for each detection
                object_id = np.argmax(scores)       # Retreive object id form the array with highest score
                confidence = scores[object_id]      # Get confidence score of the object

                # Condition for confidence and id of objects ( Car, Motorcycle, Bus, Truck)
                if confidence > 0.1 and object_id in {2, 3, 5, 7}:

                    # Calculate object  centroid's coordinate
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)

                    point = Point(center_x, center_y)       # Instantiate a class Point with the coordinates
                    area = Polygon(zone)                    # Detection Zone

                    # Check whether the object is within detection zone
                    if area.contains(point):
                        width = int(detection[2] * frame_width)     # Width of the detection box
                        height = int(detection[3] * frame_height)   # height of the detection box

                        start_x = int(center_x - width / 2)         # Start x-cor of the box
                        start_y = int(center_y - height / 2)        # Start y-cor of the box

                        # Append the object's rectangle box coordinates
                        coordinates.append([start_x, start_y, width, height])
                        # Confidence percentage of the detection
                        confidences.append(float(confidence))
                        # Append id of each object detected
                        object_ids.append(object_id)

        # Get rid of redundant overlapping coordinates with lower confidence
        indexes = cv2.dnn.NMSBoxes(coordinates, confidences, 0.25, 0.40)

        # Loop though all the rentangle coordinatesa and display a rectangular box shape
        for i in range(len(coordinates)):
            if i in indexes:
                start_x, start_y, width, height = coordinates[i]
                label = str(self.names_list[object_ids[i]])
                color = self.colors[object_ids[i]]

                # This block is used to save centroid information for tracking
                centroid_x = start_x + width / 2
                centroid_y = start_y + height / 2
                centroid_cor = np.array([[centroid_x], [centroid_y]])
                centroids.append(np.round(centroid_cor))

                # confidence = confidences[i]       # To get confidence of detection- Needs to added in cv2.putText

                # Output the rectangle box and label to the frame
                cv2.rectangle(frame, (start_x, start_y), (start_x + width, start_y + height), color, 1)
                cv2.putText(frame, str(label), (int(start_x), int(start_y)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        return centroids
