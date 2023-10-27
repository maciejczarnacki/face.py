# Faceanalyzer class has to initialize mediapipe face landmark detection tool 
# in result class has to return coordinates of all face features - eyes, irises, eyebrows and lips
# additional methods of this class could analyze if eyes and lips are open or close
# for clarity, the code does not meet the PEP 8 recommendation for line length. 

import itertools
import mediapipe as mp


class FaceAnalyzer:
    #mediapipe features initialization
    def __init__(self, max_num_faces = 3, static_image_mode = False, refine_landmarks = True):
        # face landmark detection settings   
        self.max_num_faces = max_num_faces
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius = 0)
        # eyes landmarks coordinates
        self.LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LEFT_EYE)))
        self.RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_RIGHT_EYE)))
        # eyes iris
        self.LEFT_IRIS_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LEFT_IRIS)))
        self.RIGHT_IRIS_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_RIGHT_IRIS)))
        # eyebrows landmarks coordinates
        self.LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LEFT_EYEBROW)))
        self.RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_RIGHT_EYEBROW)))
        # lips landmarks coordinates
        self.LIPS_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LIPS)))
        
        
    # face features extraction    
    def face_features(self, image: [], eyes = True, irises = True, eyebrows = True, lips_ = True) -> []:
        self.results = self.faceMesh.process(image)  
        if self.results.multi_face_landmarks and len(self.results.multi_face_landmarks) == 1:
            for faces, faceLms in enumerate(self.results.multi_face_landmarks):            
                left_eye = []
                right_eye = []
                lips = []
                left_iris = []
                right_iris = []
                left_eyebrow = []
                right_eyebrow = []
                ih, iw, ic = image.shape
                # eyes extraction
                if eyes:
                    for LEFT_EYE_INDEX in self.LEFT_EYE_INDEXES:
                        x_left, y_left = int(faceLms.landmark[LEFT_EYE_INDEX].x*iw), int(faceLms.landmark[LEFT_EYE_INDEX].y*ih)
                        left_eye.append([x_left, y_left])   
                    for RIGHT_EYE_INDEX in self.RIGHT_EYE_INDEXES:
                        x_right, y_right = int(faceLms.landmark[RIGHT_EYE_INDEX].x*iw), int(faceLms.landmark[RIGHT_EYE_INDEX].y*ih)
                        right_eye.append([x_right, y_right])
                else:
                    left_eye, right_eye = None, None
                # irises extraction    
                if irises:                       
                    for LEFT_IRIS_INDEX in self.LEFT_IRIS_INDEXES:
                        x_left, y_left = int(faceLms.landmark[LEFT_IRIS_INDEX].x*iw), int(faceLms.landmark[LEFT_IRIS_INDEX].y*ih)
                        left_iris.append([x_left, y_left])   
                    for RIGHT_IRIS_INDEX in self.RIGHT_IRIS_INDEXES:
                        x_right, y_right = int(faceLms.landmark[RIGHT_IRIS_INDEX].x*iw), int(faceLms.landmark[RIGHT_IRIS_INDEX].y*ih)
                        right_iris.append([x_right, y_right])
                else:
                    left_iris, right_iris = None, None 
                # eyebrows extraction
                if eyebrows:        
                    for LEFT_EYEBROW_INDEX in self.LEFT_EYEBROW_INDEXES:
                        x_left, y_left = int(faceLms.landmark[LEFT_EYEBROW_INDEX].x*iw), int(faceLms.landmark[LEFT_EYEBROW_INDEX].y*ih)
                        left_eyebrow.append([x_left, y_left])   
                    for RIGHT_EYEBROW_INDEX in self.RIGHT_EYEBROW_INDEXES:
                        x_right, y_right = int(faceLms.landmark[RIGHT_EYEBROW_INDEX].x*iw), int(faceLms.landmark[RIGHT_EYEBROW_INDEX].y*ih)
                        right_eyebrow.append([x_right, y_right])
                else:
                    left_eyebrow, right_eyebrow = None, None
                # lips extraction
                if lips_:                    
                    for LIPS_INDEX in self.LIPS_INDEXES:
                        x, y = int(faceLms.landmark[LIPS_INDEX].x*iw), int(faceLms.landmark[LIPS_INDEX].y*ih)
                        lips.append([x, y])     
                else:
                    lips = None            
            return left_eye, right_eye, left_iris, right_iris, left_eyebrow, right_eyebrow, lips
        else:
            return None, None, None, None, None, None, None
    
    # calculation if eyes are open or closed. Points for calculations:
    # Left eye horizontally 6, 7, left eye vertically 2, 11
    # Right eye horizontally 1, 4 or 1, 12, left eye vertically 8, 15
    # Lips horizontally 17, 30, lips vertically 4, 5
    # ear - eye aspect ratio
    def eyes_status(self, left_eye: [], right_eye: []) -> (bool, bool):
        left_horizontal = (left_eye[7][0] - left_eye[6][0])**2 + (left_eye[7][1] - left_eye[6][1])**2
        left_vertical = (left_eye[2][0] - left_eye[11][0])**2 + (left_eye[2][1] - left_eye[11][1])**2
        right_horizontal = (right_eye[1][0] - right_eye[4][0])**2 + (right_eye[1][1] - right_eye[4][1])**2
        right_vertical = (right_eye[8][0] - right_eye[15][0])**2 + (right_eye[8][1] - right_eye[15][1])**2
        left_ear = left_vertical** 0.5 / left_horizontal** 0.5
        right_ear = right_vertical** 0.5 / right_horizontal** 0.5
        if left_ear < 0.18:
            left_eye_opened = False
        else:
            left_eye_opened = True
        if right_ear < 0.18:
            right_eye_opened = False
        else:
            right_eye_opened = True
        return (left_eye_opened, right_eye_opened)
      
    # calculation if lips are open or closed
    # ler - lips aspect ratio
    def lips_status(self, lips: []) -> bool:
        lips_horizontal = (lips[30][0] - lips[17][0])**2 + (lips[30][1] - lips[17][1])**2
        lips_vertical = (lips[4][0] - lips[5][0])**2 + (lips[4][1] - lips[5][1])**2 
        ler = lips_vertical ** 0.5 / lips_horizontal ** 0.5
        if ler < 0.05:
            lips_opened = False
        else:
            lips_opened = True
        return lips_opened
            
            