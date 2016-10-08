import cv2
import sys
import numpy as np

# Get user supplied values
imagePath = sys.argv[1]
paintingPath = sys.argv[2]
cascPath = sys.argv[3]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
painting = cv2.imread(paintingPath)

#image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Init photo", image)
cv2.waitKey(3000)
cv2.imshow("Init painting", painting)
cv2.waitKey(3000)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50),
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

img_height, img_width = image.shape[:2]

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    extend_height_top = (h / 2)
    extend_height_bottom = h / 4
    extend_width = w / 4
    y_from = max(0, y - extend_height_top)
    y_to = min(img_height, y + h + extend_height_bottom)
    x_from = max(0, x - extend_width)
    x_to = min(img_width, x + w + extend_width)

    crop_img = image[y_from:y_to, x_from:x_to]

    cv2.imshow("Faces found", crop_img)
    cv2.waitKey(3000)

    mask = np.zeros(crop_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)

    crop_img_height, crop_img_width = crop_img.shape[:2]
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (20, 20, crop_img_width, crop_img_height)
    cv2.grabCut(crop_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    crop_img = crop_img * mask2[:, :, np.newaxis]
    cv2.imshow("Faces found", crop_img)
    cv2.waitKey(2000)

    # find where to place
    gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
    faces2 = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    (x, y, w, h) = faces2[0]
    painting_face = painting[y:y+h, x:x+w]
    cv2.imshow("Painting face found", painting_face)
    cv2.waitKey(200)

    #where to place
    crop_img = cv2.resize(crop_img, (0, 0), fx=1.0, fy=1.0)
    crop_img_height, crop_img_width = crop_img.shape[:2]
    x_place = (x + (w / 2)) - (crop_img_width / 2)
    y_place = (y + (h / 2)) - (crop_img_height / 2)
    print(x_place)
    print(y_place)

    # take only foreground
    print("take only foreground")
    rows, cols, channels = crop_img.shape
    img2gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Img gray", img2gray)
    cv2.waitKey(200)

    print (crop_img_width)
    print (crop_img_height)
    roi = painting[y_place:y_place+rows, x_place:x_place+cols]
    cv2.imshow("ROI", roi)
    cv2.waitKey(2000)

    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    print("foreground taken")
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)

    painting[y_place:y_place+rows, x_place:x_place+cols] = dst
    cv2.imshow("Painting to train", painting)
    cv2.waitKey(5000)

    cv2.imwrite('restult.png', painting)

cv2.imshow("Faces found", image)
