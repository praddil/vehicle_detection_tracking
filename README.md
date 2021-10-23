# Vehicle Detection and Tracking
----

### Description: 
This project aims to detect moving traffic vehicles. Each vehicles are assigned a unique ID number while tracking its path of travel.
I have used the Yolo weights V3 for the detection purpose. For tracking it utilizes Kalman Filter and Hungarian algorithm.

### Test Video:



### Required Files
Main -- main_multiprocessing.py <br />

Required “config” folder is provided to be placed in the content folder <br />
.<br/>
|-- config/data/coco.names <br />
|-- config/src/yolov3.weights <br />
|-- config/src/yolov3.cfg <br />
|-- config/video/MVI_1991.mp4<br />

https://drive.google.com/drive/folders/1CUhW7fkszUmnC7hIdWIQyRAnalaHCvjm?usp=sharing
<br/>

### Dependencies:
Python~=3.6 <br />
numpy~=1.17.3 <br />
opencv-python~=4.1.2.30 <br />
Shapely~=1.7.0 <br />
scipy~=1.4.1 <br />

### Note:
1. GPU preferred. Can run in CPU but it is utterly slow.
2. Multiprocessing starts off with a time sleep of 5 sec.
3. Adjust thresholds and parameters according to the video properties.

### Work in progress:
Speed Estimation for moving vehicles.

### Needs Improvement:
1. Detection of only selected objects instead of wide range of classes in yolo-weights can increase the runtime.
2. Multiprocessing can be better optimized- currently only one sub-process running.

### About Author:
@praddil <br />
This is my first project on Github. I am student in computer science major. I am not a professional by any means. I am learning by doing these side projects. <br />
There are two referenced files in this project that contains algorithm like Kalman Filter and Hungarian Algorithm. 
This project utilizes such algorithms to accomplish the task of tracking the objects. [Kalman.py]
<br />The author of the these algorithm files are mentioned in the referenced section.
<br />Please visit the link in the referenced section to see their work.

### References and Acknowledgements:

1. [Multiple object tracking using Kalman Filter and Hungarian Algorithm](https://github.com/srianant/kalman_filter_multi_object_tracking) -- Srini Ananthakrishnan
2. [Real Time Object Detection using YOLOv3 with OpenCV and Python](https://medium.com/analytics-vidhya/real-time-object-detection-using-yolov3-with-opencv-and-python-64c985e14786) -- Darshan Adakane




