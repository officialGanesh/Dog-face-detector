# import required modules
import cv2 as cv
from random import randrange

# Grabbing the dataset
dog_training_data_set = cv.CascadeClassifier('dog_dataSet.xml')

def Detect_dog_images():
    """This function is responsible for converting an image in gray scale"""

    img = cv.imread('Images/dog3.jpg')
    cv.imshow("Dog3",img)
    
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    # cv.imshow('Gray-Dog-1',gray)

    def rect_face():
        """This function can draw rectangles across face coordinates"""
        dog_face_coordinates = dog_training_data_set.detectMultiScale(gray)

        for (x,y,w,h) in dog_face_coordinates:
            rect_dog = cv.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

        cv.imshow("Dog Detected",rect_dog)
    rect_face()
# main function
if __name__ == "__main__":

    Detect_dog_images()
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Code Completed 🔥")
