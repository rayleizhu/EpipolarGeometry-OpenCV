import numpy as np
import cv2
from scipy import stats

def get_sift_matches(img1, img2):
    '''
    compute good SIFT matches in img1 and img2
    img1: query image /  left image
    img1: train image / right image
    '''
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    return pts1, pts2, good


def get_fundamental_mat(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2, cv2.FM_LMEDS)
#     F, mask = cv2.findFundamentalMat(pts1,pts2, cv2.FM_RANSAC)
   
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    return F, pts1, pts2


def draw_epilines(img1, img2, pts1, pts2, F12):
    # get epilines in the img1 w.r.t points in the img2
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F12)
    lines1 = lines1.reshape(-1,3)
    # get epilines in the img2 w.r.t points in the img1
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F12)
    lines2 = lines2.reshape(-1,3)
    
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    
    num_pts = len(pts1)
    colors = np.random.randint(0,255, (num_pts, 3))
    for r1, r2, pt1, pt2, color in zip(lines1, lines2, pts1, pts2, colors):
        color = tuple(color.tolist())
        # two end points of epiline in img1
        x1_l, y1_l = map(int, [ 0, -r1[2]/r1[1] ])
        x1_r, y1_r = map(int, [ w1, -(r1[2]+r1[0]*w1)/r1[1] ])
        # two end points of epiline in img2
        x2_l, y2_l = map(int, [ 0, -r2[2]/r2[1] ])
        x2_r, y2_r = map(int, [ w2, -(r2[2]+r2[0]*w2)/r2[1] ])
        # draw line
        img1 = cv2.line(img1, (x1_l, y1_l), (x1_r, y1_r), color, 1)
        img2 = cv2.line(img2, (x2_l, y2_l), (x2_r, y2_r), color, 1)
        # draw point
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2
        

def drawlines(img1,img2,lines,pts1,pts2):
    ''' 
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    '''
    r,c, _ = img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def get_rectified_stereo(im_left, im_right):
    imsize = im_left.shape[1], im_right.shape[0]
    
    pts1, pts2, _ = get_sift_matches(im_left, im_right)
    F, pts1_inliers, pts2_inliers = get_fundamental_mat(pts1, pts2)


    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_inliers,
                                                   pts2_inliers,
                                                   F, imsize)

#     retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1.astype(int32),
#                                                    pts2.astype(int32),
#                                                    F, imsize)
    
    assert retval, 'failed to estimate homographies for stereo rectification1'
    im_left_rect = cv2.warpPerspective(im_left, H1, imsize)
    im_right_rect = cv2.warpPerspective(im_right, H2, imsize)
    
    return im_left_rect, im_right_rect

def get_disp_map(im_left_rect, im_right_rect, num_disp=96, method='sgbm', filtering=True):
    # wsize default 3; 5; 7 for SGBM reduced size image;
    # 15 for SGBM full size image (1300px and above); 5 Works nicely
    assert method in ['sgbm', 'bm']
    if method == 'sgbm':
        window_size = 7  
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2, 
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM)
    else:
        left_matcher = cv2.StereoSGBM_create(numDisparities=num_disp,
                                            blockSize=7)
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    displ = left_matcher.compute(im_left_rect, im_right_rect).astype(np.int16)
    dispr = right_matcher.compute(im_right_rect, im_left_rect).astype(np.int16)
   
    if filtering:
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0
        wls_filter_l = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter_l.setLambda(lmbda)
        wls_filter_l.setSigmaColor(sigma)

        wls_filter_r = cv2.ximgproc.createDisparityWLSFilter(matcher_left=right_matcher)
        wls_filter_r.setLambda(lmbda)
        wls_filter_r.setSigmaColor(sigma)


        displ = wls_filter_l.filter(displ, im_left_rect, None, dispr)
        dispr = wls_filter_r.filter(dispr, im_right_rect, None, displ)
    
    displ_n = cv2.normalize(src=displ,
                              dst=displ,
                              beta=0, alpha=255,
                              norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    dispr_n = cv2.normalize(src=dispr,
                              dst=dispr,
                              beta=0, alpha=255,
                              norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    
    return displ_n, dispr_n



def draw_points_and_line(pts, img):
    color = tuple(np.random.randint(0,255,3).tolist())
    assert len(pts) == 2
    img = cv2.circle(img, tuple(pts[0].tolist()), 5, color, -1)
    img = cv2.circle(img, tuple(pts[1].tolist()), 5, color, -1)
    img = img1 = cv2.line(img,
                          tuple(pts[0].tolist()), 
                          tuple(pts[1].tolist()),
                          color,3)
    return img    


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def cal_normalized_distance(disp_map, pt1, pt2):
    disps = []
    for y in range(pt1[1], pt2[1]):
        x = pt1[0] + ((y - pt1[1]) / (pt2[1] - pt1[1])) * (pt2[0] - pt1[0])
        x = int(x)
        disps.append(disp_map[y, x])
    disps = reject_outliers(np.array(disps))
    min_disp = np.min(disps)
    max_disp = np.max(disps)
    disp_range = np.max(disps) - min_disp
    # in the following case we don't trust the mean disparity
    if disp_range / (min_disp + 1e-3) > 5 or max_disp < 5:
        disp = np.max(disps)
    else:
        disp = np.mean(disps)
    dist = np.linalg.norm(pt2 - pt1)
    return dist / disp


    