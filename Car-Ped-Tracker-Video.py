import cv2

# our video
video = cv2.VideoCapture('video1.mp4')

# this pre-trained classifier has been made long time back so old cars images must have been used to train
classifier_file = 'cars.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Run forever untill video stops
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # convert to gray frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Car detector
    cars = car_tracker.detectMultiScale(frame_gray)

    # Pedestrian detector
    pedestrians = pedestrian_tracker.detectMultiScale(frame_gray)

    # Draw rectangle on cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw rectangle on pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # show the image
    cv2.imshow('Car Detector >>', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

print('code completed!')
