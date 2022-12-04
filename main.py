import cv2 
import numpy as np 

# Read image
scene = cv2.imread("data/scene.png", 0)
object = cv2.imread("data/object.png", 0)

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

kp1, des1 = sift.detectAndCompute(scene, None)
kp2, des2 = sift.detectAndCompute(object, None)

scene = cv2.drawKeypoints(scene, kp1, None)
object = cv2.drawKeypoints(object, kp2, None)

matches = matcher.knnMatch(des1, des2, 2)

good_matches = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good_matches.append(m)

# scene = cv2.drawMatches(scene, kp1, object, kp2, good_matches, None, flags=4)
img_matches = cv2.drawMatches(scene, kp1, object, kp2, good_matches, None, flags=4)

objs, scenes = [], []

objs  = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
scenes = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)

H = cv2.findHomography(objs, scenes, cv2.RANSAC)[0]

h, w, _ = object.shape

obj_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
scene_corners = cv2.perspectiveTransform(obj_corners.reshape(1, -1, 2), H).reshape(-1, 2)

out = cv2.polylines(img_matches, [np.int32(scene_corners)], True, (0, 255, 0), 3)

cv2.imshow("scene", out)

cv2.waitKey(0)
cv2.destroyAllWindows()