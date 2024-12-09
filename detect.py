# Required libraries ko import kar rahe hain
import cv2
import argparse

# Function to detect faces and highlight them with a rectangle
def highlightFace(net, frame, conf_threshold=0.7):
    # Frame ka ek copy banate hain taaki original image modify na ho
    frameOpencvDnn = frame.copy()
    
    # Image/frame ki height aur width ko store karte hain
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Image ko model ke liye processable format (blob) me convert karte hain
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Processed image (blob) ko face detection model me input karte hain
    net.setInput(blob)
    # Face detection results ko fetch karte hain
    detections = net.forward()
    
    # Face boxes ke liye ek empty list banate hain
    faceBoxes = []
    
    # Har detection ko loop se process karte hain
    for i in range(detections.shape[2]):
        # Har detection ke liye confidence score nikalte hain
        confidence = detections[0, 0, i, 2]
        
        # Agar confidence threshold se zyada hai, to usse face consider karte hain
        if confidence > conf_threshold:
            # Detected face ka bounding box ke coordinates nikalte hain (x1, y1, x2, y2)
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            # Detected face ko faceBoxes list me store karte hain
            faceBoxes.append([x1, y1, x2, y2])
            # Face ke around ek rectangle draw karte hain
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    # Processed image (jisme face ke rectangles draw hue hain) aur faceBoxes list return karte hain
    return frameOpencvDnn, faceBoxes

# Argument parser banate hain jo command-line argument se image ka path lega
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Models ke configuration aur trained weights files ke paths define karte hain
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# RGB normalization ke liye model mean values define karte hain
MODEL_MEAN_VALUES = (103.939, 116.779, 123.68)

# Age aur gender prediction ke categories list me define karte hain
ageList = ['(0-5)', '(6-10)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-45)', 
           '(46-50)', '(51-55)', '(56-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', 
           '(86-90)', '(91-95)', '(96-100)']
genderList = ['Male', 'Female']

# Models ko load karte hain
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Webcam se video capture karte hain
video = cv2.VideoCapture(0)
padding = 20  # Face ke aas paas extra padding add karne ke liye

# Ek frame capture karte hain webcam se
ret, frame = video.read()

# Agar frame capture nahi hota, to program ko exit kar dete hain
if not ret:
    print("Unable to capture image")
    exit()

# Face detection function call karte hain aur result aur faceBoxes ko fetch karte hain
resultImg, faceBoxes = highlightFace(faceNet, frame)

# Agar koi face detect nahi hota, to program ko exit kar dete hain
if not faceBoxes:
    print("No face detected")
    exit()

# Har detected face ke liye processing karte hain
for faceBox in faceBoxes:
    # Detected face ko thoda padding ke saath image se extract karte hain
    face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                 max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

    # Extracted face ko model ke liye blob format me convert karte hain
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Gender prediction ke liye face blob ko gender model me input karte hain
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    # Sabse high probability wala gender result fetch karte hain
    gender = genderList[genderPreds[0].argmax()]

    # Age prediction ke liye face blob ko age model me input karte hain
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    # Sabse high probability wala age range result fetch karte hain
    age = ageList[agePreds[0].argmax()]

    # Gender aur age ko console me print karte hain
    print(f'Gender: {gender}')
    print(f'Age: {age[1:-1]} years')  # Parentheses ko hata kar clean output de rahe hain

# Webcam ko release karte hain aur sab windows ko destroy karte hain
video.release()
cv2.destroyAllWindows()
