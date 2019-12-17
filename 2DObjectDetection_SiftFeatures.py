import cv2 as cv
import numpy as np



scene1 = cv.imread("ImageT0.jpg")
scene2 = cv.imread("ImageT2.jpg")

cv.imshow("Scene", scene2)
roi = cv.selectROI('Object of interest selection', scene1, False)

u0 = roi[0]
v0 = roi[1]
object_width = roi[2]
object_height = roi[3]

object_image = scene1[v0:v0+object_height, u0:u0+object_width]

cv.destroyAllWindows()

sift = cv.xfeatures2d.SIFT_create()

object_keypoints, object_descriptors = sift.detectAndCompute(object_image, None)
print(object_keypoints)
scene_keypoints, scene_descriptors = sift.detectAndCompute(scene2, None)

object_keypoints_drawed = object_image.copy()
scene_keypoints_drawed = scene2.copy()

cv.drawKeypoints(object_image, object_keypoints, object_keypoints_drawed)
cv.drawKeypoints(scene2, scene_keypoints, scene_keypoints_drawed)

object_kp_drawed_resized = cv.copyMakeBorder(object_keypoints_drawed, scene2.shape[0]-object_keypoints_drawed.shape[0], 0, scene2.shape[1]-object_keypoints_drawed.shape[1], 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
keypoints_image = np.hstack((object_kp_drawed_resized, scene_keypoints_drawed))

cv.imshow("keypoints", keypoints_image)
cv.waitKey()
cv.destroyAllWindows()

descriptor_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = descriptor_matcher.knnMatch(object_descriptors, scene_descriptors, 2)

matches = []

for x, y in knn_matches:
    if x.distance < 0.75 * y.distance:
        matches.append(x)




matches_img = np.empty((max(object_image.shape[0], scene2.shape[0]), object_image.shape[1]+scene2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(object_image, object_keypoints, scene2, scene_keypoints, matches, matches_img, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("good matches", matches_img)
cv.waitKey()
cv.destroyAllWindows()

object_ = np.empty((len(matches), 2), np.float32)
scene = np.empty((len(matches), 2), np.float32)

for t in range(len(matches)):
    object_[t, 0] = object_keypoints[matches[t].queryIdx].pt[0]
    object_[t, 1] = object_keypoints[matches[t].queryIdx].pt[1]
    scene[t, 0] = scene_keypoints[matches[t].trainIdx].pt[0]
    scene[t, 1] = scene_keypoints[matches[t].trainIdx].pt[1]

homography, r = cv.findHomography(object_, scene, cv.RANSAC)

object_corners = np.zeros((4, 1, 2))
object_corners[1, 0, 0] = object_image.shape[1]
object_corners[2, 0, 0] = object_image.shape[1]
object_corners[2, 0, 1] = object_image.shape[0]
object_corners[3, 0, 1] = object_image.shape[0]

scene_corners = cv.perspectiveTransform(object_corners, homography)

cv.line(matches_img, (int(scene_corners[0, 0, 0] + object_image.shape[1]), int(scene_corners[0, 0, 1])), (int(scene_corners[1, 0, 0] + object_image.shape[1]), int(scene_corners[1, 0, 1])), (0, 0, 255), 4)
cv.line(matches_img, (int(scene_corners[1, 0, 0] + object_image.shape[1]), int(scene_corners[1, 0, 1])), (int(scene_corners[2, 0, 0] + object_image.shape[1]), int(scene_corners[2, 0, 1])), (0, 0, 255), 4)
cv.line(matches_img, (int(scene_corners[2, 0, 0] + object_image.shape[1]), int(scene_corners[2, 0, 1])), (int(scene_corners[3, 0, 0] + object_image.shape[1]), int(scene_corners[3, 0, 1])), (0, 0, 255), 4)
cv.line(matches_img, (int(scene_corners[3, 0, 0] + object_image.shape[1]), int(scene_corners[3, 0, 1])), (int(scene_corners[0, 0, 0] + object_image.shape[1]), int(scene_corners[0, 0, 1])), (0, 0, 255), 4)

cv.imshow("Detetected object", matches_img)
cv.waitKey()
cv.destroyAllWindows()








