from imutils import face_utils
import dlib
import cv2
 
# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# cap = cv2.VideoCapture('Ryan Reynolds Hilariously Crashes Original X-Men Reunion With Hugh Jackman Patrick Stewart.mp4')
cap = cv2.VideoCapture(0)
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (X, Y, W, H) = face_utils.rect_to_bb(rect)
        cv2.putText(image, f"Faces: {i+1}", (X-10, Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw on our image, all the finded cordinate points (x,y) 
        for (x, y) in shape:
            cv2.rectangle(image, (X, Y, W, H), (0, 255, 0), 1)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    
    # Show the image
    cv2.imshow("Output", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

