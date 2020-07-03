import cv2
import numpy as np

image = cv2.imread("Nadia_Murad.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
########## convert image to rgb ########

########################################################
###### general program to apply any filter on the image 
def apply_filter(image, color_rgb, intensity=0.5, ):
    
    overlay = np.full(image.shape, color_rgb, dtype='uint8')
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    ################ blend two images ###############
    final_image = cv2.addWeighted(overlay, intensity, image, 1.0, 0)
    return final_image

###################### fun to apply sepia filter on an image ###########
def apply_sepia(image, intensity=0.5):
    sepia_r, sepia_g, sepia_b = (112, 66, 20)
    sepia_rgb = (sepia_r, sepia_g, sepia_b)
    sepia_image = apply_filter(image, sepia_rgb)
    return sepia_image
sepia_image = apply_sepia(image)

#################### fun to apply 

while True:
    ######### display normal image ###########
#     cv2.resizeWindow('normal_image', 1000, 1000)

    cv2.imshow('normal_image', image)
    
    ########## display filter image #########
#     cv2.resizeWindow('sepia_image', 1000, 1000)

    cv2.imshow('sepia_image', sepia_image)
    if cv2.waitKey(3) & 0xff == 27:
        break
        
cv2.destroyAllWindows()

