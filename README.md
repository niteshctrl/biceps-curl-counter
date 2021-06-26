<!---
# Biceps Curl Counter
* Computer Vision based Biceps Curl Counter using Mediapipe's Pose Classification model([BlazePose](https://arxiv.org/abs/2006.10204)).
* Inaccurate range of motion(>40 degrees and <160 degrees) during the exercise will not be counted forcing the user to repeat with correct physical posture and full range of motion. This will also serve as a real-time accuracy feedback to the user.
* Angle is calculated using the coordinates of shoulder, elbow and wrist landmarks as predicted by the model.
* OpenCV is used for illustration and visualization.
--->
## Overview
A biceps curl counter based on human pose estimation. The idea is to make the user aware of inaccurate range of motion in real-time so that he/she can correct it immediately without the need of a human exercising partner. Often as a novice, we are dealing with incorrect exercising postures which eventually leads to no gain or in the worst conditions, unwanted damage and tear to the muscles.

The counter doesn't tick if the range of motion during a bicep curl is within 40 degrees and 160 degrees. Hence to get the counter going, user has to curl and uncurl his/her biceps in a full range of motion. No cheating now!


## Table of Contents
* [Installation and Demo](#Installation-and-Demo)
* [Description](#Description)
* [References and Credits](#References-and-Credits)


## Installation and Demo
* Installing the depedencies:
```
pip install requirements.txt
```
* Separate counts will be recoreded for left and right bicep curls. Press 'r' key to reset the counter anytime and 'Esc' to quit. To run the app, execute the following on the terminal:
```
python demo.py
```


## Description
The project is based on Mediapipe's Pose Estimation Model [Blazepose](https://arxiv.org/abs/2006.10204) at the backend which is responsible for calculating the 3D coordinates of the human body keypoints. We then extract the three: wrist, elbow and shoulder keypoints and form two straight lines joning shoulder and elbow(A) / elbow and wrist(B). Calculating the angle between the two straight lines A and B gives us the angle formed at elbow of the hand which is then used for counter increments. A binary variable 'flag' is responsible for maintaining the position of the hand during curls; UP or DOWN. A successful transition of the hand from DOWN to UP increases the counter by 1 each time. Separate counters are maintained for right and left hands.


## Future Improvements:
* Inclsion of custom range of motion which is currently fixed between between 40 degrees and 160 degrees so that user can use it for different forms of curls.


## References and Credits
1. [Guide to Human Pose Estimation with Deep Learning(Nanonets)](https://nanonets.com/blog/human-pose-estimation-2d-guide/)
2. [Mediapipe Pose Classification(Google's Github)](https://google.github.io/mediapipe/solutions/pose_classification.html)
3. [Real-time Human Pose Estimation in the Browser(TF Blog)](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
4. [MediaPipePoseEstimation(Nicknochnack's Github)](https://github.com/nicknochnack/MediaPipePoseEstimation)
