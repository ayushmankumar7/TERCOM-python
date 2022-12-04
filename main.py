import cv2 
import numpy as np 

# Read image
scene = cv2.imread("data/scene.png", 0)
object = cv2.imread("data/object.png", 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(scene, None)
kp2, des2 = sift.detectAndCompute(object, None)

scene = cv2.drawKeypoints(scene, kp1, None)
object = cv2.drawKeypoints(object, kp2, None)

cv2.imshow("scene", scene)
cv2.imshow("object", object)

cv2.waitKey(0)
cv2.destroyAllWindows()