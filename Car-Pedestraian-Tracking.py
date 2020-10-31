import cv2

# Our image
img_file = 'car4.jpg'


# Our pre-trained car classifier
classifier_file = 'cars.xml'

# create opencv image
img = cv2.imread(img_file)

# convert into grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars --> returns array of coordinates
cars = car_tracker.detectMultiScale(img_gray)

print(cars)

# Draw rectangles around the car
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# show the image
cv2.imshow('Car Detector >>', img)
cv2.waitKey()

print('Code Completed!')
