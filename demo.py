# Importing the libraries
import cv2
import numpy as np
import mediapipe as mp


def calc_angle(a,b,c): # 3D points
    ''' Arguments:
        a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
        
        Returns:
        theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
    '''
    a = np.array([a.x, a.y])#, a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])#, b.z])    # Reduce 3D point to 2D
    c = np.array([c.x, c.y])#, c.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14    # Convert radians to degrees
    return np.round(theta, 2)


def infer():
    mp_drawing = mp.solutions.drawing_utils     # Connecting Keypoints Visuals
    mp_pose = mp.solutions.pose                 # Keypoint detection model
    flag = None     # Flag which stores hand position(Either UP or DOWN)
    count = 0       # Storage for count of bicep curls

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) # Lnadmark detection model instance
    while cap.isOpened():
        _, frame = cap.read()

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)                       # Get landmarks of the object in frame from the model

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Calculate angle
            angle = calc_angle(shoulder, elbow, wrist)      #  Get angle 

            # Visualize angle
            cv2.putText(image,\
                    str(angle), \
                        tuple(np.multiply([elbow.x, elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
        
            # Counter 
            if angle > 160:
                flag = 'down'
            if angle < 40 and flag=='down':
                count += 1
                flag = 'up'
            
        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(image, str(count), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    infer()