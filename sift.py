from __future__ import print_function
import cv2
import numpy as np
 
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
import math
import sys
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def eulerAnglesToRotationMatrix(theta) :
  R = np.array([[math.cos(theta),    math.sin(theta),    0],
                [-math.sin(theta),    math.cos(theta),     0],
                [0,                     0,                      1]
                ])
  return R
def getT(theta,translation) :
  R = np.array([[math.cos(theta),    math.sin(theta),    translation[0]],
                [-math.sin(theta),    math.cos(theta),     translation[1]],
                [0,                     0,                      1]
                ])
  return R
def _extract_target_file_name(img_src, img_dst, method=None):
    '''
    '''
    spl_src = img_src.split('/')
    spl_dst = img_dst.split('/')
    if len(spl_src)>1 and len(spl_dst)>1:
        # include the current directories name in the target file's name
        tmp = spl_src[-2]+'_'+spl_src[-1][:-4] + '__' + spl_dst[-2]+'_'+spl_dst[-1][:-4]
    else:
        # only include the input files' name in the target file's name
        tmp = spl_src[-1][:-4] + '__' + spl_dst[-1][:-4]

    return tmp if method is None else method+'_'+ tmp
def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  src = np.array(points1)
  dst = np.array(points2)
   
  
  # estimate affine transform model using all coordinates
  model = AffineTransform()
  model.estimate(src, dst)

  # robustly estimate affine transform model with RANSAC
  model_robust, inliers = ransac((dst, src), AffineTransform, min_samples=3,
                                residual_threshold=2, max_trials=100)
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
 
  # Print estimated homography
  print("Estimated homography : \n",  h)
  #print("Affine transform:")
  #print(f"Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), "
      #f"Translation: ({model.translation[0]:.4f}, "
      #f"{model.translation[1]:.4f}), "
      #f"Rotation: {model.rotation:.4f}")
  #print("RANSAC:")
  print(f"Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), "
      f"Translation: ({model_robust.translation[0]:.4f}, "
      f"{model_robust.translation[1]:.4f}), "
      f"Rotation: {model_robust.rotation:.4f}")
  R=eulerAnglesToRotationMatrix(model_robust.rotation)
  #print("R=")
  #print(R)
  T=getT(model_robust.rotation,model_robust.translation)
  print("T = ")
  print(f"{T[0][0]:.4f}  {T[0][1]:.4f}  {T[0][2]:.4f} \n"
      f"{T[1][0]:.4f}  {T[1][1]:.4f}  {T[1][2]:.4f} \n"
      f"{T[2][0]:.4f}  {T[2][1]:.4f}  {T[2][2]:.4f} \n")
  return im1Reg, h
 
 
if __name__ == '__main__':
  args = sys.argv
  ###### fetching options from input arguments
  # options are marked with single dash
  options = []
  for arg in args[1:]:
      if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
          options += [arg[1:]]

  ###### fetching parameters from input arguments
  # parameters are marked with double dash,
  # the value of a parameter is the next argument
  listiterator = args[1:].__iter__()
  while 1:
      try:
          item = next( listiterator )
          if item[:2] == '--':
              exec(item[2:] + ' = next( listiterator )')
      except:
          break
  
  out_file_name = _extract_target_file_name(img_src, img_dst)
  #out_file_name = _extract_target_file_name(img_dst, img_src)
  # Read reference image
  refFilename=img_dst
  #refFilename=img_src
  #refFilename = "../lab_c/lab_c_scan.png"
  print("Reading reference image : ", refFilename)
  #imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  imReference = np.flipud( cv2.imread( refFilename, cv2.IMREAD_COLOR) )
 
  # Read image to be aligned
  imFilename=img_src
  #imFilename=img_dst
  #imFilename = "../lab_c/lab_c_scan_lab_c_15.png"
  #print("Reading image to align : ", imFilename);  
  #im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  im = np.flipud( cv2.imread( imFilename, cv2.IMREAD_COLOR) )
   
  #print("Aligning images ...")
  ## Registered image will be resotred in imReg. 
  ## The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
  #imReg, h = alignImages(imReference, im)
   
  # Write aligned image to disk. 
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
