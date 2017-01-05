#import cv2
import tensorflow as tf
from random import shuffle
import numpy as np
import os

def read_dataset(dataset_location):
    #Reading the dataset
    number_of_images = 0
    number_of_training_images=0
    number_of_validation_images=0
    available_classes = os.listdir(dataset_location)
    number_of_classes = len(available_classes)
    print(number_of_classes, "classes found:")
    
    one_hot_matrix = np.identity(number_of_classes)
    # Returns a 2-D tensor with dimension equal to number_of_classes
    # [[1, 0, 0],
    #  [0, 1, 0],
    #  [0, 0, 1]] for 3 classes
    
    dataset = {}
    index = 0

    for i in available_classes:
        image_count = len(os.listdir(dataset_location + "/" + i))
        # the number of items in path
        print(index, "\t", i, ":", image_count)
        dataset[i] = [image_count, index]
        index += 1
    
    classified_input_list = [] #The list which will be used for classification input
    classified_validation_list = [] #The list which will be used for validation

    #selected 20 classes from caltech101 dataset, name, number of images and its label for each class
    #the dictionary consists of number of images in each class and its ONE-HOT-list label

    for i in dataset.items():
        number_of_images+=i[1][0]
        for j in range(1,i[1][0]+1):

            #obtain file name
            if(j<10):
                filename = dataset_location + "/" + i[0] +"/image_000" + str(j) + ".jpg"
            elif(j>=10) and (j<100):
                filename = dataset_location + "/" + i[0] + "/image_00" + str(j) + ".jpg"
            elif(j>=100):
                filename = dataset_location + "/" + i[0] + "/image_0" + str(j) + ".jpg"
            else:
                filename = dataset_location + "/" + i[0] + "/image_" + str(j) + ".jpg"

            if(j%5==0):
                classified_validation_list.append([i[1][1],filename])#add 1/5 of the dataset to validation set
                number_of_validation_images+=1
            else:
                classified_input_list.append([i[1][1],filename])#add 4/5 of the dataset to input set
                number_of_training_images+=1

            #THIS SEGMENT IS FOR PREPROCESSING THE INPUT IMAGES
            #image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            #dest=cv2.resize(image,(100,100),interpolation = cv2.INTER_CUBIC)
            #cv2.imwrite(filename,dest)

    shuffle(classified_input_list)
    shuffle(classified_validation_list)

    """
    for i in range(0, len(classified_input_list)):
        print(classified_input_list[i])
    return number_of_images,classified_input_list,classified_validation_list, dataset, number_of_classes
    """

    print("===TRAINING===")
    for i in range(0, len(classified_input_list)):
        print(classified_input_list[i][0])
        print(classified_input_list[i][1])

    print("Total number of training images: ", number_of_training_images)
    print("===VALIDATION===")
    for i in range(0, len(classified_validation_list)):
        print(classified_validation_list[i][0])
        print(classified_validation_list[i][1])
    print("Total number of validation images: ", number_of_validation_images)
    print("Total number of images:", number_of_images)

    return number_of_images, number_of_training_images, number_of_validation_images, number_of_classes, classified_input_list, classified_validation_list, dataset


#read_dataset("./firstDataSet")
#print("Success!")
