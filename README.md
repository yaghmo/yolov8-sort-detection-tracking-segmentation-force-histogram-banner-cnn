# Abstract

This project focuses on the analysis of spatial and temporal relationships among objects in videos
by employing the Force Banner concept and integrating it into a learning process. The project
follows several key steps, including video reading and frame separation, object detection and seg-
mentation, object tracking over time, computation of Force Histograms between pairs of objects
(masks) that will represent a descriptor for each frame, and construction of the Force Banner using
the obtained descriptors. The project’s methodology will be tested and compared on various types
of videos, including synthetic sequences and real-world datasets comprising action scenarios and
drone footage. The progress will also involve integrating the developed approach into a learning
process, potentially utilizing convolutional neural networks (CNNs) for enhanced prediction of
complex spatiotemporal situations. By leveraging the proposed methodology, this project aims to
contribute to the analysis of object dynamics in videos by a prediction of their spatiotemporal pat-
terns, with potential applications in domains such as action recognition and video understanding.

# Dependencies

To install required dependencies run (python vesion 3.9 _ 3.10):

**`$ pip install -r requirements.txt`**

# Directories

`📂 Data` folder: is the directory that contains the raw video data and the associated trims manually captured from that video. The video is from YouTube, representing a sequence of the match 2021 FIVB Volleyball Men’s Nations League match between the Japanese and USA teams. [YouTube](https://www.youtube.com/watch?v=umM0XarqwYE&ab_channel=PowerVolleybal)

`📂 Model-Weights` folder: contains the yolov8 weights, if none is present in the folder, the program `📜 object_detection_to_masks.py` will download the weights version into that folder ( you can change the folder path on the code here : `L403. model = YOLO("Model-Weights/yolov8l-seg.pt")` ). The current available weight's versions are the following:

| Model name |     Version     |
| :---------: | :--------------: |
| YOLOv8n-seg |       nano       |
| YOLOv8s-seg |      small      |
| YOLOv8m-seg |      medium      |
| YOLOv8l-seg | large (used one) |
| YOLOv8x-seg |   extra-large   |

For more details please check [Yolov8](https://github.com/ultralytics/ultralytics).

`📂 utils` folder: contains the used algorithms that were needed for this project.

## Here's a scheme of the main repertory:

```markdown
📦MAIN
 ├─ 📂Data
 │ ├─ 📂trim_Frames
 │ ├─ 📂trim_Masks
 │ │ ├─ 📂A0
 │ │ └─ etc...
 │ ├─ trim.pm4
 │ └─ video.pm4 
 ├─ 📂utils
 │ ├─ 📜sort.py
 │ ├─ 📜FHistogramRaster.c
 │ └─ 📜fhistograms.py 
 ├─ 📂Model-Weights
 │ ├─ yolov8l-seg.pt
 │ └─ etc...
 ├─ 📜main.py
 ├─ 📜object_detection_to_masks.py
 ├─ 📜CNN.ipynb
 ├─ 📜banner.py
 ├─ 📜README.md
 └─ Requirements.txt 
```

# Demo

This project contains 4 main programs which some of them call other modules:

* `📜 main.py` -- is the program that is focused on frame extraction, you can call the next program ( not recommended ) directly by uncommenting the section present in the line: 65.
* `📜 object_detection_to_masks.py` -- is the next step program which can be lunched independently, it creates a folder that will contain saved images of pair of masks that had been selected to be tracked.
* `📜 banner.py` is the program that generate a banner matrix for every mask folder given in the path and regroupe them all in their corresponding class folder.
* `📜 CNN.ipynp` is the program computed on Google Colab under TPU runtime, it was used to train the model.

You can either pass through argumants or run the code without argument. If an argument is missing, you will need to change manually the variable inside the code. Now assuming that you dont use arguments:

1. Starting by running the code `📜 main.py` whithout the need of uncommenting the 65th line for every tram you want to extract its frame.
   manually giving the path and the video name LINE....
2. Then run `📜 object_detection_to_masks.py` by giving a new name to the folder class so it will create that folder inside the corresponding trim mask folder. manually change the variables LINE.....
3. After creating all the folders of pair masks wanted, run the program `📜 banner.py` which will arrange the masks files in case they are not ordered in their folder, then create a force banner matrix for every folder of pair masks, and then it will group them all by by their classes and sort them in three folders which will represent their class.
4. Lastely `📜 CNN.ipynb` comes to train a model on those last created class folders ( banners ).

Using flags:

* you can lunch the previous programs `📜 main.py` ,`📜 object_detection_to_masks.py` and `📜 banner.py` by passing them arguments without the need of changing the variables in the code:
  1. `📜 main.py` : run the command `python main.py "path to the video"`
  2. `📜 object_detection_to_masks.py` : run the command `python object_detection_to_masks.py -v "name of the video" -f "path to the folder that contains the video" -m "the name of the model" -t "the name of folder where to save the pair of masks"`
  3. d

Here's an example of execution according the repertory scheme :

```
$ python main.py Data/trim.mp4 
=> X frames extracted to Data/trim_frames/
$ python object_detection_to_masks.py -v [ trim , trim.pm4 ] -f Data -m [ yolov8l-seg , yolov8l-seg.pt ] -t A0
# you have the choice for whats in between [ ].
=> folder A0 created in side Data/trim_masks/ , masks generated and saved in the folder path Data/trim_masks/A0/
$ banner.py -f -d 
```

# Citations

sort alghorithm along with force-histogram ( both programs are found inside the folder '📂 utils' )

```
@ARTICLE{force_histograme,
  author={Matsakis, P. and Wendling, L.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A new way to represent the relative position between areal objects}, 
  year={1999},
  volume={21},
  number={7},
  pages={634-643},
  doi={10.1109/34.777374}}

@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}}
```
