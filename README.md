# Ghostbusters: How the Absence of Class pairs in Multi Class Multi Label Datasets Impacts Classifier Accuracy

## Abstract

> Compositional  bias  is  common  in  Multi-Class  Multi-Label datasets where certain classes frequently co-occur to-gether.  Classification performance due to non-iid behaviorof Multi-Class Multi-Label datasets is largely unexplored.We evaluate the potential impact of this compositional biason Multi-Class Multi-Label classifiers, and, demonstrate aremedy - with novel framework of representing bias through“connectedness”.   Our  work  effectively  strives  to  answerquestions along the lines of “Is the classification accuracyof cat impacted by the presence of the person in the image?”and  more  importantly,“Is  the  classification  accuracy  ofcat impacted by the absence of the person in the test im-age?”.  Interestingly our experiments show that class pairsthat are present in a test image, but do not appear togetherelsewhere in the training dataset do impact the recognitionaccuracy  and  thus  we  refer  to  them  as  ghost  class  pairs.We make a surprising discovery: higher the connectednessof classes (based on class pairs appearing within the sameimages), higher the classification accuracy.  Based on thisobservation, we develop a greedy data augmentation strat-egy that recommends which missing pairs need to be addedto the training dataset in order to improve F1 score withminimal data addition.  This ultimately is able to improveclassification accuracy by 25-30% in several scenarios.

## Hypothesis depiction

![](/Images/cat-person.png) 

## Uniqueness of MSCOCO dataset

![](/Images/pullfig.png)

## Setup python libraries

- TensorFlow
- Keras
- sklearn
- openCV
- Random
- NumPy

## Dataset Images
- MNIST-MCML Dataset

![](/Images/0_2_4_4.jpg) ![](/Images/0_2_6_5.jpg)
![](/Images/1_3_5_4.jpg) ![](/Images/1_3_7_5.jpg)

- Alphabets-MCML Dataset

![](/Images/A_C_E_0_1.jpg) ![](/Images/D_B_F_0_1.jpg)
![](/Images/H_D_B_0_1.jpg) ![](/Images/G_A_C_0_1.jpg)

## Connectivity Calculation

```
def connectednessMeasure(trainLabels, testLabelsVal):
    arrayVal = correlationmt(trainLabels)
    connectedness = []
    y_test_num = convertToNum(testLabelsVal)
    for w in range(len(testLabelsVal)):
        count = 0
        for m in range(len(testLabelsVal[w])):
            counter = 0
            for i in range(len(y_test_num[w])):
                if arrayVal[y_test_num[w][i]][m] == 1:
                    counter = counter + 1
            count = count + counter/len(y_test_num[w])
        connectedness.append([y_test_num[w],round(count/10, 2)])
    return connectedness
```

## Team
- Sidharth Kathpal
- Siddha Ganju
- Anirudh Koul
