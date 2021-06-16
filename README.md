# Biceps Curl Counter
* Computer Vision based Biceps Curl Counter using Mediapipe's Pose Classification model([BlazePose](https://arxiv.org/abs/2006.10204)).
* Inaccurate range of motion(>40 degrees and <160 degrees) during the exercise will not be counted forcing the user to repeat with correct physical posture and full range of motion. This will also serve as a real-time accuracy feedback to the user.
* Angle is calculated using the coordinates of shoulder, elbow and wrist landmarks as predicted by the model.
* OpenCV is used for illustration and visualization.

### TBD:
* Currently only implemented for left hand curls. Can be extended to both hand in the future.
