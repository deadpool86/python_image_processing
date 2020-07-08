import os
import numpy as np
import cv2
import face_recognition

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR): #loads the dir in the known_faces_folder
#     print(name)
    
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
#         print(filename)
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
    
#       learn how to recognize a face
        encoding = face_recognition.face_encodings(image)
        if len(encoding) == 0:
            continue
            
        known_faces.append(encoding[0])
        known_names.append(name)
print(len(known_faces))





#unknown faces has some image 
UNKNOWN_FACES_DIR = 'unknown_faces_test'

# take each image in unknown_faces
for filename in os.listdir(UNKNOWN_FACES_DIR):
    
    if filename == '.ipynb_checkpoints/':
        continue
    # print the name of the file
    print(f'Filename {filename}', end='')
    
    #load image file and store in image
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}', mode="RGB")
    
    #get all the faces in the image
    locations = face_recognition.face_locations(image)
    
    #know how to recognize the image based on locations
    encodings = face_recognition.face_encodings(image, locations)
    
    #convert color channel from bgr2rgb
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #print the number of faces found 
    print(f', found {len(encodings)} face(s)')
    
    if(len(encodings) == 0):
        continue
    
    names_in_the_image = []
    #make pair of the face_location with corresponding encoding
    for (top, right, bottom, left), face_encoding in zip(locations, encodings):
        
        ## compare the cur_face encoding and check if it matches to a known face
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        
        ######### initialize the name to None #########
        name = None
        
        ########## find the best match index using the encondings ##########
        face_distances  = face_recognition.face_distance(known_faces, face_encoding)
        
        ######## find the index of the minimum face distance 
        ####### as this is the best match
        best_match_index = np.argmin(face_distances) # returns index of min Value
        
        ######### check if there is a match,  get the name at best match index
        if matches[best_match_index]:
            name = known_names[best_match_index]
        
        if name:
            names_in_the_image.append(name)
            
        ####### show a rectangle on the face
        top_left = (left, top)
        bottom_right = (right, bottom)
        
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 5)
        
        ###### show the name of the person below the rectangle
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, name, (left + 10, bottom + 15), font, 0.5, (200, 200, 200), 10)
        
        
    ######### get the name from the names_in_the_image and push the image to that folder ########
    for name in names_in_the_image:
        
        ########## get folder by name #######
#         path = os.path.join(KNOWN_FACES_DIR, name)
        
#         ###### write the image in this path ###########
#         cv2.imwrite(f'{KNOWN_FACES_DIR}/{name}/{filename}', image)
        print(f'{name} ')
    print("\n")
        
#         shutil.copy(image, path)
        
        
# now what do we do