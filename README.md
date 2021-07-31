# Ghostbusters: How the Absence of Class pairs in Multi Class Multi Label Datasets Impacts Classifier Accuracy

## Abstract

> Machine learning classifiers like Naive Bayes, Convolutional Neural Networks assume independent and identically distribution (iid), yet most real-world datasets exhibit non independent and identically distributed (non-iid) properties, like class imbalance. For multiclass multilabel datasets where certain classes frequently co-occur together, compositional bias is common. Instead of all potential class pairs occurring equally (pure iid), most pairs don’t occur, some pairs occur, and few pairs occur frequently (i.e. long tailed distribution of pairs). Classification performance of non-iid behavior on multiclass multilabel datasets is largely unexplored. Motivated by the potential effect of this compositional bias on most real-world large-scale datasets, we evaluate the potential impact of this bias on multiclass multilabel classifiers to ascertain whether or not it is a significant issue, and can we demonstrate a remedy. Our work effectively strives to answer questions along the lines of ''Is the accuracy of the class person impacted by the presence of the class cat in the image?'' and more importantly, ''Is the accuracy of the class person impacted by the absence of the class cat in the image?''. Interestingly our experiments show that the absent class pairs i.e. classes that are absent in the query image, but do appear elsewhere in the dataset impact recognition accuracy of the query image and thus we refer to them as ghost class pairs. Given the greenfield nature of this domain, we exhaustively experiment with a MNIST-like synthesized dataset that possesses class distributions similar to MSCOCO. We make a surprising discovery: higher the degree of separation of classes (based on class pairs appearing within the same images), lower the recognition accuracy. Based on this observation, we develop a greedy data augmentation strategy that recommends which missing pairs need to be added to the training dataset in order to improve F1 score with minimal data addition. This ultimately is able to improve classification accuracy by 35-40\% in several scenarios. Our augmentation strategy is automated, generalizable, and scalable and thus impacts production real-world classifiers where test-time examples can’t always be anticipated. Our analysis can also be utilized as a reference to compare and contrast a practitioners dataset with our benchmarked scenarios and figure out how much the accuracy can be improved further.

![](/Images/cat-person.png) 

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
