# import required modules
import cv2 as cv
from random import randrange

# Grabbing the dataset
dog_training_data_set = cv.CascadeClassifier('dog_dataSet.xml')

def Detect_dog_images():
    """This function is responsible for converting an image in gray scale"""

    img = cv.imread('Images/dog1.jpg')
    cv.imshow("Dog1",img)
    
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    cv.imshow('Gray-Dog-1',gray)


# main function
if __name__ == "__main__":

    Detect_dog_images()

    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Code Completed 🔥")
