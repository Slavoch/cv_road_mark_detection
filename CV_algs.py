import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import collections 
from scipy.optimize import fsolve
import math

from Configs.configs import PARAMS,IMG_SHAPE,WINDOW_HEITH,WINDOW_PARAMS,CLASSES,ROADBED_PARAMS,SIDES, HIST_PARAMS


def plot_images(data, layout='row', cols=2, figsize=(20, 12)):
    '''
    Utility function for plotting images
    :param data [(ndarray, string)]: List of data to display, [(image, title)]
    :param layout (string): Layout, row-wise or column-wise
    :param cols (number): Number of columns per row
    :param figsize (number, number): Tuple indicating figure size
    '''
    rows = math.ceil(len(data) / cols)
    f, ax = plt.subplots(figsize=figsize)
    if layout == 'row':
        for idx, d in enumerate(data):
            img, title = d

            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            #plt.axis('off')
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
                #print(img.max())
            elif len(img.shape) == 3:
                plt.imshow(img)
                
    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = data[r + rows*c]
                nb_channels = len(img.shape)
                
                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                #plt.axis('off')
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                
                elif len(img.shape) == 3:
                    plt.imshow(img)
              
                counter += 1
    plt.show()

    return ax


def warp_image(img, warp_shape, src, dst):
    '''
    Performs perspective transformation (PT)
    :param img (ndarray): Image
    :param warp_shape: Shape of the warped image
    :param src (ndarray): Source points
    :param dst (ndarray): Destination points
    :return : Tuple (Transformed image, PT matrix, PT inverse matrix)
    '''
    
    # Get the perspective transformation matrix and its inverse
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image
    warped = cv2.warpPerspective(img, M, warp_shape, flags=cv2.INTER_LINEAR)
    return warped, M, invM


def preprocess_image(img, visualise=False):
    '''
    Pre-processes an image. Steps include:
    1. Distortion correction FIX
    2. Perspective Transformation
    3. ROI crop FIX
    
    :param img (ndarray): Original Image
    :param visualise (boolean): Boolean flag for visualisation
    :return : Pre-processed image, (PT matrix, PT inverse matrix)
    '''
    
    k = 209/251
    b = 37958/251
    f = lambda x:k*x + b
    int(f(400))
    
    #k2 = -207/251 
    k2 = -k
    b2 = 162409/251
    f2 = lambda x:k2*x + b2
    f2(400)
    
    ysize = img.shape[0]
    xsize = img.shape[1]
    
    # 1. Distortion correction
    undist = img
    
    # 2. Perspective transformation
    range_vision = PARAMS["RANGE_OF_VISION"]
    src = np.float32([
        (f(range_vision),range_vision), #(436,342),    
        (f2(range_vision),range_vision),#(365,342), 
        (f2(600),600), #(158,593),  
        (f(600),600), #(645,593),
    ])

    dx = 300
    dst = np.float32([
        (xsize - dx, 0),
        (dx, 0),
        (dx, ysize),
        (xsize - dx, ysize)
    ])

    warped, M, invM = warp_image(undist, (xsize, ysize), src, dst)

    # 4. Visualise the transformation
    if visualise:
        img_copy = np.copy(img)
        
        cv2.polylines(img_copy, [np.int32(src)], True, 1, 3)
        
        plot_images([
            (img_copy, 'Original Image'),
            (warped, "warped")
        ])

    return warped, (M, invM)


def get_poly_points(fit):
    '''
    Get the points for the left lane/ right lane defined by the polynomial coeff's 'left_fit'
    and 'right_fit'
    :param fit (ndarray): Coefficients for the polynomial that defines the lane line
   : return (Tuple(ndarray, ndarray)): x-y coordinates for the lane line
    '''
    ysize, xsize = IMG_SHAPE
    
    # Get the points for the entire height of the image
    plot_y = np.linspace(0, ysize-1, ysize)
    #plot_x = fit[0] * plot_y**2 + fit[1] * plot_y + fit[2]
    polynom = np.poly1d(fit)
    plot_x = polynom(plot_y)

    # But keep only those points that lie within the image
    plot_x[plot_x < 0] = 0
    plot_x[plot_x > xsize] = xsize
    plot_y = np.linspace(ysize - len(plot_x), ysize - 1, len(plot_x))
    
    return plot_x.astype(np.int64), plot_y.astype(np.int64)


def check_validity(left_fit, right_fit, diagnostics=False):
    '''
    Determine the validity of lane lines represented by a set of second order polynomial coefficients 
    :param left_fit (ndarray): Coefficients for the 2nd order polynomial that defines the left lane line
    :param right_fit (ndarray): Coefficients for the 2nd order polynomial that defines the right lane line
    :param diagnostics (boolean): Boolean flag for logging
    : return (boolean)
    '''
    
    if left_fit is None or right_fit is None:
        if(diagnostics):
            print(f"Exception: left_fit is {left_fit}, right_fit is {right_fit}")
        return False
    
    plot_xleft, plot_yleft= get_poly_points(left_fit)
    plot_xright, plot_yright =get_poly_points(right_fit)
    # Check whether the two lines lie within a plausible distance from one another for three distinct y-values

    y1 = IMG_SHAPE[0] - 1 # Bottom
    y2 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * PARAMS['VALIDATE_Y2']) # For the 2nd and 3rd, take values between y1 and the top-most available value.
    y3 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * PARAMS['VALIDATE_Y3'])

    # Compute the respective x-values for both lines
    x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
    x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
    x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]

    x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
    x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
    x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]

    # Compute the L1 norms
    x1_diff = abs(x1l - x1r)
    x2_diff = abs(x2l - x2r)
    x3_diff = abs(x3l - x3r)

    # Define the threshold values for each of the three points #FIT
    
    
    min_dist_y1 = PARAMS["min_dist_y1"] # 510 # 530 
    max_dist_y1 = PARAMS["max_dist_y1"] # 750 # 660
    min_dist_y2 = PARAMS["min_dist_y2"]
    max_dist_y2 = PARAMS["max_dist_y2"] # 660
    min_dist_y3 = PARAMS["min_dist_y3"]
    max_dist_y3 = PARAMS["max_dist_y3"] # 660
    
    if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
        (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
        (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
        if diagnostics:
            print("Violated distance criterion: " +
                  "x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f}".format(x1_diff, x2_diff, x3_diff))
        return False

    # Check whether the line slopes are similar for two distinct y-values
    # x = Ay**2 + By + C
    # dx/dy = 2Ay + B
    
    y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
    y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
    y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
    y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]

    # Compute the L1-norm
    norm1 = abs(y1left_dx - y1right_dx)
    norm2 = abs(y3left_dx - y3right_dx)
    
#     if diagnostics: print( norm1, norm2)

    # Define the L1 norm threshold
    thresh = PARAMS['TANGENT']
    if (norm1 >= thresh) | (norm2 >= thresh):
        if diagnostics:
            print("Violated tangent criterion: " +
                  "norm1 == {:.3f}, norm2 == {:.3f} (thresh == {}).".format(norm1, norm2, thresh))
        return False
    
    return True


def polyfit_sliding_window(binary, cache, roadbed_fit, out = None,visualise=False, diagnostics=False):
    '''
    Detect lane lines in a thresholded binary image using the sliding window technique
    :param binary (ndarray): Thresholded binary image
    :cahe = [deque(maxlen = ..) *2] contains the last several linefits
    :param visualise (boolean): Boolean flag for visualisation
    :param diagnositics (boolean): Boolean flag for logging
    '''
    # Step 1: Compute the histogram along all the columns in the lower half of the image. 
    # The two most prominent peaks in this histogram will be good indicators of the
    # x-position of the base of the lane lines
    if visualise and out is None:
        out = np.dstack((binary, binary, binary)) * 255
    
    histogram = None
    cutoffs = [int(binary.shape[0] //3)]
    
    for cutoff in cutoffs:
        histogram = np.sum(binary[cutoff:,:], axis=0)
        
        if histogram.max() > 0:
            break

    if histogram.max() == 0:
        print('Unable to detect lane lines in this frame. Trying another frame!')
        return False,out, np.array([None,None]),(CLASSES.NONE,CLASSES.NONE)
    
    # Find the peak of the left and right halves of the histogram
    # They will be the starting points for the left and right lines
    
    hist_roi = np.zeros_like(histogram)
    hist_roi[HIST_PARAMS['roi_x_start']:HIST_PARAMS['roi_x_end']] = histogram[HIST_PARAMS['roi_x_start']:HIST_PARAMS['roi_x_end']]
    
    midpoint = hist_roi.shape[0] // 2
    first_max = np.argmax(hist_roi)
    leftx_base = None
    rightx_base = None
    if(first_max < midpoint):
        leftx_base = first_max
        #new_midpoint = max(leftx_base + PARAMS['min_dist_y1'],midpoint)
        start = leftx_base + PARAMS['min_dist_y1']
        end = leftx_base + PARAMS['max_dist_y1']
        rightx_base = np.argmax(hist_roi[start:end]) + start
    else:
        rightx_base = first_max
        #new_midpoint = min(rightx_base - PARAMS['min_dist_y1'],midpoint)
        start = rightx_base  - PARAMS['max_dist_y1']
        end = rightx_base  - PARAMS['min_dist_y1']
        leftx_base = np.argmax(hist_roi[start:end]) + start
    
    if diagnostics:
        plt.imshow(binary, cmap="gray")
        plt.plot(histogram, 'm', linewidth=4.0)

        plt.plot((midpoint, midpoint), (0, IMG_SHAPE[0]), 'c')
        plt.plot((0, IMG_SHAPE[1]), (cutoff, cutoff), 'c')
        
        plt.plot((HIST_PARAMS['roi_x_start'],HIST_PARAMS['roi_x_start']),(0, IMG_SHAPE[0]),'b')
        plt.plot((HIST_PARAMS['roi_x_end'],HIST_PARAMS['roi_x_end']),(0, IMG_SHAPE[0]),'b')
        
        plt.plot((leftx_base,leftx_base),(0, IMG_SHAPE[0]),'g')
        plt.plot((rightx_base,rightx_base),(0, IMG_SHAPE[0]),'g')
    
    
    
    #####
    # Perform line search and fit through the base points
    left_fit, left_class, right_fit, right_class = None,None,None,None
    if leftx_base is not None:
        left_fit, left_class,left_pCounts = get_Class_LaneInds(binary,leftx_base,out)
    if rightx_base is not None:
        right_fit, right_class,right_pCounts = get_Class_LaneInds(binary,rightx_base,out)

    # close scope with roadbed if needs
    if left_fit is None:
        left_fit = roadbed_fit[0]
        left_class = CLASSES.SOLID
    if right_fit is None:
        right_fit = roadbed_fit[1]
        right_class = CLASSES.SOLID
        
    ### check line collision
    if (left_fit is not None) and (right_fit is not None):
        diff = np.poly1d(left_fit) - np.poly1d(right_fit)
        solution,_,collision,_ = fsolve(diff,IMG_SHAPE[0]//2,full_output=True)
        if collision == 1:
            if  0 < solution and solution < IMG_SHAPE[1]:
                if(left_pCounts < right_pCounts):
                    left_fit = None
                    left_class = CLASSES.NONE
                else:
                    right_fit = None
                    right_class = CLASSES.NONE
    
    #####
    # Validate detected lane lines    
    valid = check_validity(left_fit, right_fit, diagnostics=diagnostics)
    ret = False
    if not valid:
        # If the detected lane lines are NOT valid:
        # 1. Compute the lane lines as an average of the previously detected lines
        # from the cache and flag this detection cycle as a failure by setting ret=False
        # 2. Else, if cache is empty, return 
        
        if len(cache[0]) == 0 or len(cache[1]) == 0:
            if diagnostics: print('WARNING: Unable to detect lane lines in this frame.')
            return ret, out, [left_fit, right_fit] , (left_class,right_class)
        if diagnostics:
            print(f"Compute the lane lines as an average of the previously detected lines, Cache len :{len(cache[0])}")
        avg_params = np.mean(cache, axis = 1)
        left_fit, right_fit = avg_params[0], avg_params[1]
        cache[0].popleft()
        cache[1].popleft()
    else:
        cache[0].append(left_fit)
        cache[1].append(right_fit)
        ret = True
    
    # Plot the fitted polynomial
    if visualise:
        plot_xleft, plot_yleft = get_poly_points(left_fit)
        plot_xright, plot_yright = get_poly_points(right_fit)
        
        left_poly_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
        right_poly_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])

        cv2.polylines(out, np.int32([left_poly_pts]), isClosed=False, color=(200,255,0), thickness=4)
        cv2.polylines(out, np.int32([right_poly_pts]), isClosed=False, color=(200,255,0), thickness=4)
    return ret, out, np.array([left_fit, right_fit]) , (left_class,right_class)


def isBaseValid(new_window_base):
    return (WINDOW_PARAMS.margin.value<= new_window_base) and (new_window_base <= IMG_SHAPE[1] - WINDOW_PARAMS.margin.value)


def get_Class_LaneInds(binary,x_base, out = None):
    
    nonzero = binary.nonzero()
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0] # np.array is mondatory
    
    # Current positions to be updated for each window
    x_current = x_base
    
    lane_inds = []
    classes = []
    dx = collections.deque([0, 0, 0],  maxlen=3)
    isWindow_movedOut = False
    for window in range(WINDOW_PARAMS.nb_windows.value):
        
        win_x_low = x_current - WINDOW_PARAMS.margin.value
        win_x_high = x_current + WINDOW_PARAMS.margin.value
        
        win_y_low = IMG_SHAPE[0] - (1 + window) * WINDOW_HEITH
        win_y_high = IMG_SHAPE[0] - window * WINDOW_HEITH

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                         & (nonzerox >= win_x_low) & (nonzerox <= win_x_high)).nonzero()[0]
        
        classes.append(classify(binary,win_x_high, win_y_high, win_x_low, win_y_low))
        # print(classes[window])
        if(classes[window] != CLASSES.NONE) and (classes[window] != CLASSES.EMPY):
            if len(good_inds) >  WINDOW_PARAMS.minpix.value:
                #Fit ax +b line in window
                x = nonzerox[good_inds]
                y = nonzeroy[good_inds]        
                
                fit, residual, *_ = np.polyfit(y, x, 1,full=True)
                fit = np.poly1d(fit)
                residual = max(residual,default=0)
                
                #If residual too high then:                
                max_res =  WINDOW_PARAMS.MAX_RESIDUAL_FITLINE_SOLID_SOLID.value if \
                    classes[window] == CLASSES.SOLID_SOLID  or\
                    classes[window] == CLASSES.DASH_SOLID or \
                    classes[window] == CLASSES.SOLID_DASH\
                    else WINDOW_PARAMS.MAX_RESIDUAL_FITLINE.value
                # print(residual)
                if(residual > max_res):
                    classes[window] = CLASSES.NONE
                    # FIX FIX FIX add  diagnostics
                    #print(f"resid warning{residual}")
                    x_new = x_current + int(np.mean(dx))
                    if isBaseValid(x_new):
                        x_current = x_new
                    else:
                        isWindow_movedOut = True
                else:    
                    lane_inds.append(good_inds)
                    #next base is meadian of Y on the next window
                    x_new = int(fit(win_y_low  - WINDOW_HEITH / 2))
                    if isBaseValid(x_new):
                        dx.append(x_new - x_current)
                        x_current = x_new
                    else:
                        isWindow_movedOut = True
        else:
            x_new = x_current + int(np.mean(dx))
            if isBaseValid(x_new):
                x_current = x_new
            else:    
               isWindow_movedOut = True      
        
        if (out is not None):
            # Draw windows for visualisation
            cv2.rectangle(out, (win_x_low, win_y_low), (win_x_high, win_y_high),\
                          WINDOW_PARAMS.win_colors.value[classes[window]], 2)
            
        if (isWindow_movedOut):
            break
        
    if (out is not None):
        x = 20
        y = 20
        dy = 30
        for CLASS in CLASSES:
            out = cv2.putText(out,CLASS.name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,WINDOW_PARAMS.win_colors.value[CLASS])
            y +=dy
    
    classes = np.array(classes)
    count_of_nonNone = np.sum(np.logical_and((np.array(classes) !=CLASSES.EMPY), (np.array(classes) !=CLASSES.NONE)))
    if count_of_nonNone < PARAMS["MIN_WINDOWS_TO_DETECT"]:        
        return None, CLASSES.NONE, 0
    
    final_class = classify_by_seq(classes)
    
    if (len(lane_inds) == 0):
        return None, final_class, 0
    
    lane_inds = np.concatenate(lane_inds)
    # Extract pixel positions for the left and right lane lines
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    if (out is not None):
         # Color the detected pixels for each lane line
        out[y, x] = [255, 0, 0]
    
    if len(x) >= WINDOW_PARAMS.min_lane_pts.value:
        return np.polyfit(y, x, 2), final_class,len(x)
    else:
        return None, final_class, len(x)


def classify(binary,x_high, y_high, x_low, y_low):
    
    new_binary = np.array([[binary[i,j] ^ binary[i,j + 1] for j in range(x_low,x_high - 1)] for i in range(y_low,y_high)])
    
    summs = new_binary.sum(axis=1)    
    
    MAX = max(summs,default=0)
    MIN = min(summs,default=0)
    #MEAN = np.mean(summs) if MAX != 0 else 0
    
    MaxSums = summs.copy()
    #FIX FIX FIX
    while (np.sum(MaxSums == MAX) <  len(summs) // WINDOW_PARAMS.min_share_toClassify.value) and MAX > 0:
        MaxSums[MaxSums == MAX] = 0
        MAX = max(MaxSums,default=0)
    
    MinSums = summs.copy()
    while (np.sum(MinSums == MIN) <  len(summs) // WINDOW_PARAMS.min_share_toClassify.value) and MIN < MAX:
        MinSums[MinSums == MIN] = MAX
        MIN = min(MinSums,default = MAX)
        
    MEAN = int((MAX + MIN) //2)
    #print("MINMAX ",MIN,MAX,MEAN)
    #TODO #######################################
    #add sigma
    carrent_class = CLASSES.EMPY
    #if(SHIFT > 80):
    #    current_class = CLASSES["NONE"]
    if(MAX == 0):
        current_class =  CLASSES.EMPY
    elif(MAX == 2) and (MIN == 0):
        current_class =  CLASSES.DASH
    elif(MEAN == 2):
        current_class =  CLASSES.SOLID
    elif(MEAN == 4):
        current_class = CLASSES.SOLID_SOLID
    elif(MAX == 4) and (MIN == 2):
        current_class = CLASSES.DASH_SOLID
    else:
        current_class =  CLASSES.NONE

    return current_class


def classify_by_seq(windows):
    arr = np.insert(windows,[0,len(windows)],CLASSES.NONE)
    Count_N = np.count_nonzero(arr == CLASSES.NONE)
    Count_E = np.count_nonzero(arr == CLASSES.EMPY)
    # DASH if prev EMPTY next SOLID and vice versa
    length = len(arr)
    for i in range(1,length):
        prev_class = arr[i-1]
        current_class = arr[i]
        
        if (current_class == CLASSES.SOLID and prev_class == CLASSES.EMPY):
            arr[i] = CLASSES.DASH 
        
        elif(current_class == CLASSES.EMPY and prev_class == CLASSES.SOLID):
            arr[i - 1] = CLASSES.DASH
    # for i in range(1,length-1):
    #     prev_class = arr[i-1]
    #     current_class = arr[i]
    #     if (current_class == CLASSES.DASH):
            
    # Strong line if it is n times in a row
    Count_of_SS_in_row = max(np.diff(np.where(arr !=  CLASSES.SOLID_SOLID)[0]) - 1,default=0)
    if(Count_of_SS_in_row > WINDOW_PARAMS.MAX_SS_WIN_TO_DETECT_SS.value):
        return  CLASSES.SOLID_SOLID
    
    Count_of_S_in_row = max(np.diff(np.where(arr !=  CLASSES.SOLID)[0]) - 1,default=0)
    if(Count_of_S_in_row > WINDOW_PARAMS.MAX_SOLID_WIN_TO_DETECT_SOLID.value):
        return  CLASSES.SOLID 
    
    # dash if Empty > n but not more then k
    Count_of_E_in_row = max(np.diff(np.where(arr !=  CLASSES.EMPY)[0]) - 1,default=0)
    if(Count_E >= WINDOW_PARAMS.MIN_E_WIN_TO_DETECT_D.value) and \
        len(arr) - Count_E -Count_N >= WINDOW_PARAMS.MAX_NO_E_WIN_TO_DETECT_D.value:
        return  CLASSES.DASH 
    
    #set MAX priority class if it is here at least n times
    MAX = CLASSES.EMPY
    for CLASS in CLASSES:
        counts = np.count_nonzero(arr == CLASS)
        #FIX
        if counts >= 2:
            if CLASS.value > MAX.value:
                MAX = CLASS
    
    # counts = set()
    # MAX = CLASSES.EMPY
    
    # for element in arr:
    #     if(element in counts):
    #         if (element.value > MAX.value):
    #             MAX = element
    #     else:
    #         counts.add(element)
    # print(MAX)
    # print("NONONONONE",Count_of_S_in_row)
    return  MAX


def roadbed_line_fit(binary,side):
    
    binary = binary[ROADBED_PARAMS.slice_of_Y_min.value:ROADBED_PARAMS.slice_of_Y_max.value,:]

    target = None
    if(side == SIDES.left):
        f = lambda y: 0.71 * y - 175
        fun = lambda x,y: x > f(y)
        target = 0 
    elif side == SIDES.right: 
        f = lambda y: -0.71 * y + 800 + 175
        fun = lambda x,y: x < f(y)
        target =-1
    
    y_points = []
    x_points = []
    
    for i,row in enumerate(binary):
        nonzero = np.nonzero(row)
        if(len(nonzero[0]) > 0):
            
            y = i + ROADBED_PARAMS.slice_of_Y_min.value
            x = nonzero[0][target] #+ adjustment(y,side)
            y_points.append(y)
            x_points.append(x)
              
#             if(side == SIDES.right):
#                 print(i,x,y,fun(x,y))
#             if(fun(x,y)):
#                 y_points.append(y)
#                 x_points.append(x)
                
#     for point in zip(x_points,y_points):
#         print(side,point)
        
    polyfit = None
    if (len(y_points) >= ROADBED_PARAMS.min_lane_pts.value):
        polyfit = np.polyfit(y_points, x_points, 4)    
    return polyfit


def adjustment(y,side):
    y_max = 600
    y_min = ROADBED_PARAMS.adjustment_y_min.value
    if(y<=y_max and y>= y_min):
        a = ROADBED_PARAMS.adjustment_width.value
        k = a/(y_max - y_min)
        b = a - k* y_max
        res = k* y + b
        if(side == SIDES.right):
            return res
        elif(side == SIDES.left):
            return -res
    return 0


def roadbed_processing(bin_roadbed, side,out = None, visualise = False):
    #line fit
    polyfit = roadbed_line_fit(bin_roadbed,side)
    
    #plotting
    if visualise:
        if out is None:
            out = np.dstack((bin_roadbed,bin_roadbed,bin_roadbed)) * 255
        
        if polyfit is not None:
            plot_x, plot_y = get_poly_points(polyfit)
            poly_pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))])    
            cv2.polylines(out, np.int32([poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
    success = True
    if(polyfit is None):
        success = False
    return success, out, polyfit


def draw(img, warped, invM, poly_param, color= (0,220, 110),lineColor =(255, 255, 255)):
    '''
    Utility function to draw the lane boundaries and numerical estimation of lane curvature and vehicle position.
    :param img (ndarray): Original image
    :param warped (ndarray): Warped image
    :param invM (ndarray): Inverse Perpsective Transformation matrix
    :param poly_param (ndarray): Set of 2nd order polynomial coefficients that represent the detected lane lines

    :return (ndarray): Image with visual display
    '''
    undist = img
    warp_zero = np.zeros_like(warped[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fit = poly_param[0]
    right_fit = poly_param[1]
    plot_xleft, plot_yleft = get_poly_points(left_fit) 
    plot_xright, plot_yright = get_poly_points(right_fit)
    
    pts_left = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_xright, plot_yright])))])
    
    pts = np.hstack((pts_left, pts_right))
    
    # Color the road
    cv2.fillPoly(color_warp, np.int_([pts]), color)
                    
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False,
                color=lineColor, thickness=10)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False,
                color=lineColor, thickness= 10)

    # Unwarp and merge with undistorted original image
    unwarped = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    out = cv2.addWeighted(undist, 1, unwarped, 0.4, 0)

    return out


def smalldeleteArreas(img,diagnostics = False):
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img)
    
    for i in range(1, numLabels):
        if stats[i, cv2.CC_STAT_AREA] < PARAMS["MIN_ROADBED_AREA"]:
            labels[labels == i] = 0
    result = (labels > 0).astype("uint8")
    
    if(diagnostics):
        print("after smalldeleteArreas:")
        plt.imshow(labels)
        plt.show()
    
    return result

#TO DO FIX cache
def simple_test(mask_mark_bin, mask_roadbed_bin ,visualisation = False,diagnostics = False):
    binary_roadbed = smalldeleteArreas(mask_roadbed_bin,diagnostics= diagnostics)
    cache = [collections.deque(maxlen = PARAMS['MAX_CACHE_LEN']),
         collections.deque(maxlen = PARAMS['MAX_CACHE_LEN'])]
    l_suc,out,l_roadbed_fit = roadbed_processing(binary_roadbed,SIDES.left,visualise=visualisation)
    r_suc,out,r_roadbed_fit = roadbed_processing(binary_roadbed,SIDES.right,out = out,visualise=visualisation)
    roadbed_fit = (l_roadbed_fit, r_roadbed_fit)
    suc = min([l_suc,r_suc])
    
    ret, img_poly, poly_param, (lc,rc) = polyfit_sliding_window(mask_mark_bin,cache, roadbed_fit, visualise= visualisation, diagnostics=diagnostics)
    if diagnostics:
        plot_images([(img_poly, 'Polyfit'), (out, 'Out')])
    return [
        [ret, poly_param, (lc,rc)],
        [suc,roadbed_fit],
        [img_poly,out]
    ]


def addLine(Mat,init_size,Minv,poly_fit,order):
    y_size = Mat.shape[0]
    x_size = Mat.shape[1]
    
    y_init_size, x_init_size = init_size
    
    Y = np.arange(0,y_init_size)
    polynom = np.poly1d(poly_fit)
    X = polynom(Y)
    points = np.vstack([X,Y,np.ones_like(X)])
    newPoints = (Minv @ points)
    newPoints = ((newPoints / newPoints[2])[0:2])
    
    #point_x = np.clip(newPoints[0] * x_size/x_init_size,0,x_size -1).astype("int")
    
    point_x = newPoints[0] * x_size/x_init_size
    point_y = newPoints[1] * y_size/y_init_size
    
    condition = (point_x > 0) & (point_x < x_size) & (point_y > 0) & (point_y < y_size)
    
    point_x = point_x[condition].astype("int")
    point_y = point_y[condition].astype("int")
    points = np.array([point_x,point_y]).T
    thikness = 5
    Mat = cv2.polylines(Mat, [points],False, order, thikness)
    Mat[:,:thikness] = 0
    Mat[:,-thikness:] = 0
    
    ##Mat[point_y,point_x] = order
    return Mat




