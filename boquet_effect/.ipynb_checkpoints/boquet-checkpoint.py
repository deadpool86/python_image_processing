import cv2
import numpy as np
import Tracker_HSV as tr
from PIL import Image
import os

########## all the files to import ############


##############################################
##############################################

###### read the file ##########
image = cv2.imread("./flower.jpeg")
######## apply GaussianBlur #####
def find_best_contour(contours):
    maxi = cv2.contourArea(contours[0])
    best_contour = []

    ######### find the best contour in the list of contours #######
    for i in range(len(contours)):
        contour = contours[i]
        contour_area = cv2.contourArea(contour)

        if contour_area > maxi:
            maxi = contour_area
            best_contour = contour
    return best_contour

#################### getTransparentMask function ########
def get_transparent_mask(datas):
    new_datas = []
    for data in datas:
        if data[0] in range(0, 10) and data[1] in range(0, 10) and data[2] in range(0, 10):
            new_datas.append((255, 255, 255, 0))
        else:
            new_datas.append(data)
    return new_datas







##################### boquet function #################
def boquet(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 3)
    ######## Convert to HSV ###########
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ############ save a super blurred image in this folder #########
    image_very_blur = cv2.GaussianBlur(image_blur, (81, 81), 3)

    cv2.imwrite('image_blurred.jpg', image_very_blur)


    ############################################################
    ############## select roi #########################
    # helps us to select the required region of interest using just our mouse
    xs, ys, w, h = cv2.selectROI('mask', image_blur)

    crop_img = crop_img_true = crop_img_contour = image_blur[ys:ys+h, xs:xs+w]

    ### if user didn't select any ROI, then we select whole image as our ROI
    if crop_img.shape[0] <= 1:
        crop_img_true = image_blur


    #########################################################################
    #########################################################################
    ## use tr.tracker to get hsv range
    ## use inRange function to create a mask in given range 
    ## save the mask
    lh, ls, lv, uh, us, uv = tr.tracker(crop_img_true)

    crop_img_true = cv2.cvtColor(crop_img_true, cv2.COLOR_BGR2HSV)
    mask_inRange = cv2.inRange(crop_img_true, (lh, ls, lv), (uh, us, uv))
    # _, mask_inRange = cv2.threshold(mask_inRange, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite('mask_inRange.jpg', mask_inRange)

    ############## we have our mask ###############
    ###############################################
    ## create a threshhold of mask
    ## create a GaussianAdaptiveThreshhold of the earlier threshHold
    _, threshhold = cv2.threshold(mask_inRange, 200, 255, cv2.THRESH_BINARY)
    Guassian_thresh = cv2.adaptiveThreshold(threshhold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)

    ############## we have a mask with lesser noise #################

    ###########################################################################
    ############################################################################
    ######## draw contours on the image ######################################
    _, contours, _ = cv2.findContours(Guassian_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ########### take the contour with max area and draw that on the image ####
    best_contour = find_best_contour(contours)
    
    
    ################## draw the best contour on the crop_img_contour and save the image #######
    cv2.drawContours(crop_img_contour, best_contour, -1, (0, 255, 0), 5)
    # cv2.imwrite('image_contour.jpg', crop_img_contour)


    #####################################################################
    #######################################################################
    ## make a blank space of 1 channel and blank color only for the mask
    black_background = np.zeros((crop_img_true.shape[0], crop_img_true.shape[1]), np.int8)
    black_background[:] = (0)

    ########## take the polygon made by the contour and place it on the black image
    # mask_poly = cv2.fillConvexPoly()
    mask_poly = cv2.fillConvexPoly(black_background, best_contour, (255, 255, 255))
    crop_img_true = cv2.cvtColor(crop_img_true, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('check_if_image_is_RGB.jpg', crop_img_true)

    temp_image = cv2.bitwise_and(crop_img_true, crop_img_true, mask=mask_poly)
    cv2.imwrite('image_mask.jpg', temp_image)
    img = Image.open('image_mask.jpg').convert('RGBA')
    datas = img.getdata()
    new_datas = get_transparent_mask(datas)
    
    ########### convert the mask into a transparent mask #######
    img.putdata(new_datas)
    img.save('this_is_a_transparent.png')


    ###################################################################
    ###################### paste this image on xs, ys ################
    blurred_image = Image.open('image_blurred.jpg')
    
    ########### image is not rgba type image ##############
    blurred_image.paste(img, (xs, ys), img)
    blurred_image.save('flower_boquet.png')

    blurred_image.show('Final result')
    
    
#########################################
#########################################
#### call the function ##################
boquet(image)



cv2.waitKey(0)
cv2.destroyAllWindows()