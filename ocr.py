import ocr_engine as engine
import cv2


# load the classifier
clf = engine.load('digit_classifier_knn.model')

name = "images/1.jpg"
img = cv2.imread(name)
digits = engine.perform_ocr(img, clf)

cv2.imshow("Digits", img)
cv2.waitKey(0)
