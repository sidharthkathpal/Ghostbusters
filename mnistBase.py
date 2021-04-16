import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import cv2
import tensorflow_datasets as tfds

class mnistBase:
    
    def __init__(self):
        self.numOfObjects = 3
        self.imageSize = 128
        self.testOrTrain = 1
        self.printVal = 0

    def mnistBaseDef(self, testOrTrain, shuffleFiles):
        ds = tfds.load('mnist', split=testOrTrain, shuffle_files=shuffleFiles)

        imagesVal = []
        labelsVal = []

        for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
            labelsVal.append(np.array(example["label"]))
            imagesVal.append(np.array(example["image"]).reshape((28,28)))
        
        return imagesVal, labelsVal

    def convertToList(self, imagesList, labelsList):
        image_lists = [ [] for _ in range(10) ]
        label_lists = [ [] for _ in range(10) ]
        for i in range(len(labelsList)):
            image_lists[labelsList[i]].append(imagesList[i])
            label_lists[labelsList[i]].append(labelsList[i])
        return image_lists, label_lists
        
    """
    def datasetCreationCall(numOfImages, datasetDef, imagesVal, labelsVal):
        for i in range(numOfImages):
            for j in range(numOfObjects):
    """

    def randomLocationGen(self, listOfLabels, images_list, labels_list):
        images = []
        temp = ''
        for i in range(len(listOfLabels)):
            labelVal = listOfLabels[i]
            imageIndex = random.randint(0, (len(labels_list[labelVal]) - 1))
            if self.printVal == 1:
                print(labelVal, len(images_list[labelVal]), len(labels_list[labelVal]) , imageIndex)
            images.append(images_list[labelVal][imageIndex])
            temp = temp + str(labelVal) + '_'
        temp = '../pics/' + temp + str(self.testOrTrain)+'.jpg'
        final_image = self.createImage(images)
        cv2.imwrite('{}'.format(str(temp)), final_image)
        return final_image, listOfLabels, temp

    def createImage(self, image_val_list):
        finalImage = np.zeros((self.imageSize, self.imageSize))
        borderVal = int(self.imageSize/2)
        valList = [[0,0],[0,borderVal],[borderVal,0],[borderVal,borderVal]]
        random.shuffle(valList)
        for j in range(len(image_val_list)):
            resizeImage = self.resizeImageForFour(image_val_list[j])
            finalImage[valList[j][0] : valList[j][0] + borderVal,valList[j][1] : valList[j][1] + borderVal] = resizeImage
        return finalImage

    def resizeImageForFour(self, image):
        borderVal = int(self.imageSize/2)
        compiledImage = np.zeros((borderVal, borderVal))
        lenNewBorder = len(image)
        x_axis = random.randint(0, (borderVal - lenNewBorder))
        y_axis = random.randint(0, (borderVal - lenNewBorder))
        compiledImage[x_axis:x_axis + lenNewBorder, y_axis:y_axis + lenNewBorder] = image
        return compiledImage
    
    def generateCompleteDataset(self, img_lists, lbl_lists):
        image_final = []
        list_of_labels = []
        list_of_images = []
        for i in range(10):
            for j in range(i+1, 10):
                for k in range(j+1, 10):
                    final_image, listOfLabels, img_name = self.randomLocationGen([i,j,k], img_lists, lbl_lists)
                    image_final.append(final_image)
                    list_of_labels.append(listOfLabels)
                    list_of_images.append(img_name)
        return image_final, list_of_labels, list_of_images


"""
first_image = np.array(imagesVal[1], dtype='uint8')
pixels = first_image.reshape((28, 28))
print(first_image.shape)
print(np.array(labelsVal[1]))
plt.imshow(pixels, cmap='gray')


images, labels = mnistBaseDef('train', False)
imagesList, labelsList = convertToList(images, labels)
#print(labelsList[0])
pixels, _ = randomLocationGen([0,2,3], imagesList, labelsList) 
plt.figure()
print(pixels)
plt.imshow(pixels, cmap='gray')
"""