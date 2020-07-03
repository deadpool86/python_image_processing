import cv2
import numpy as np

image = cv2.imread("Nadia_Murad.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
########## convert image to rgb ########

########################################################
###### general program to apply any filter on the image 
def apply_filter(image, color_rgb, intensity=0.5):
    
    overlay = np.full(image.shape, color_rgb, dtype='uint8')
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    ################ blend two images ###############
    final_image = cv2.addWeighted(overlay, intensity, image, 1.0, 0)
    return final_image
########################################################################
###################### functions to apply filters on any image ###########
########################################################################

######################################################################
########## 1. sepia filter ############################################
def apply_sepia(image, intensity=0.5):
    sepia_r, sepia_g, sepia_b = (112, 66, 20)
    sepia_rgb = (sepia_r, sepia_g, sepia_b)
    sepia_image = apply_filter(image, sepia_rgb)
    return sepia_image

############# 2. black and white filter ################################
############ check if image is grayScale or not #######################
def validate_grayscale(image):
    if image.shape[2]:
        #not grayScale image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def apply_black_and_white(image, threshhold=127, maxValue=255):
    gray_image = validate_grayscale(image)
    
    rows, cols = gray_image.shape
    
    for row in range(rows):
        for col in range(cols):
            if gray_image[row, col] < threshhold:
                gray_image[row, col] = 0
            else:
                gray_image[row, col] = maxValue
    return gray_image

###################################################################
######################################################################
##### 3. Cartoon filter on the image ######################
def apply_cartoon(image):
    ############################################
    ########## make edges homogeneous
    bw_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ######### blur the image first ##################
    bw_image = cv2.medianBlur(bw_image, 5)
    ############ apply an adaptive threshhold on the image ###########
    edges = cv2.adaptiveThreshold(bw_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    
    #####################################################################
    ########### convert image to cartoonish using bilateral filter ########
    image = cv2.bilateralFilter(image, 7, 300, 300)
    
    ############ get final image by taking bitwise and of edges and image######
    final_image = cv2.bitwise_and(image, image, mask=edges)
    return final_image
    
################################################################
########## negative filter ####################################
def apply_negative(image):
    rows, cols, channels = image.shape
    new_image = image.copy()
    for channel in range(channels):
        for row in range(rows):
            for col in range(cols):
                new_image[row, col, channel] = ~image[row, col, channel]
    return new_image


######################################################################
########################## clay effect #############################
#################### apply filters by calling functions ##################

sepia_image = apply_sepia(image, 0.8)

black_white_image = apply_black_and_white(image, 100, 255)

cartoon_image = apply_cartoon(image)

negative_image = apply_negative(image)

clay_image = apply_clay(image)
while True:
    ######### display normal image ###########
#     cv2.resizeWindow('normal_image', 1000, 1000)

    cv2.imshow('normal_image', image)
    
    ########## display filter image #########
#     cv2.resizeWindow('sepia_image', 1000, 1000)
#     cv2.imshow('cartoon_image', cartoon_image)
#     cv2.imshow('sepia_image', sepia_image)
#     cv2.imshow('bw_image', black_white_image)
#     cv2.imshow('negative', negative_image)
    cv2.imshow('clay_image', clay_image)
    if cv2.waitKey(3) & 0xff == 27:
        break
        
cv2.destroyAllWindows()

