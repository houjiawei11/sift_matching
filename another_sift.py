#https://answers.opencv.org/question/199318/how-to-use-sift-in-python/
import numpy as np
import cv2
img1=cv2.imread("/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/freiburg_building101/freiburg_building101.jpg")
img2=cv2.imread("/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/freiburg_building101/freiburg_building101_Freiburg101_scan_-145_.jpg")
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
sift=cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)
matches=sorted(matches,key= lambda x:x.distance)
matching_result=cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=2)
cv2.imshow("image",matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()