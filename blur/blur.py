import numpy as np
import cv2

image = cv2.imread('flower.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


####### global variables ##############
blurring = False
height, width = 60, 60
# prev_x, prev_y = 0, 0


def blur_area(event, x, y, flags, param):
    
    global blurring, height, width
#     , prev_x, prev_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        blurring = True
    if event == cv2.EVENT_MOUSEMOVE:
        top_left_x, top_left_y = (x-height//2, y-width//2)
        
        if blurring == True:
            new_image = image[top_left_y: top_left_y+height, top_left_x:top_left_x+width]
            new_image = cv2.blur(new_image, (5, 5))
            image[top_left_y: top_left_y+height, top_left_x:top_left_x+width] = new_image
            
        
        
    if event == cv2.EVENT_LBUTTONUP:
        blurring = False
        
    

#######################################
##### initiate named window ###########
cv2.namedWindow('image')

cv2.setMouseCallback('image', blur_area)
############## show the image ############
while True:
    
    cv2.imshow('image', image)
    
    if cv2.waitKey(3) & 0xff == 27:
        cv2.imwrite('flower_blurred.jpg', image)

        break

cv2.destroyAllWindows()