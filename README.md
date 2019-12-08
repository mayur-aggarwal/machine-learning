# machine-learning
## Getting Started with Machine Learning Algos
## *_Implementation from scratch_*

This repository contains basic ML algos implementation from scratch, no lib is used like sklearn or tf.
Following ML algos has been implemented.
1. EM & MLE
2. K-Means
3. Parzen window
4. K-NN
5. PCA

### _**ML algos implemented for below mentioned problem statements**_

* #### *Problem statement 1:*
  * #### _**Fashion-MNISTDataset**_
Fashion-MNIST is a dataset consisting of 60,000 training examples and 10,000 test examples.
Each example is a 28x28 pixels gray-scale image. Each image is labeled with 10 class categories.
Figure 1 shows example images in this dataset.
Here, each image is considered to be 784 dimensional data sample. So, there is a need to reduce the dimension of the data. Principal component analysis can be used for selecting the important features and create a lower dimensional feature vector for classification task. So, implement a module or a function to get the principal components for each of the data sample. Again, implement your own code for implementing PCA. Also, evaluate the variation of classification performance with variation in number of principal components that are included as part of feature vectors.
You can retrieve this dataset from https://github.com/zalandoresearch/fashion-mnist
<p align="center">
  <img src="https://github.com/mayur-aggarwal/machine-learning/blob/master/fashion_mnist_sample_dataset.png">
</p>
<p align="center">Figure 1: Fashion-MNIST Dataset.</p>

* #### *Problem statement 2:*
  * #### _**Blood Test**_
This dataset consist of outcomes of three Blood Tests (Test1, Test2 and Test3) for analyzing
the condition of Heart of a patient. Doctors in a hospital is analyzing these outcomes and
are providing the report for patient indicating whether the Heart is Healthy or the patient
needs medication or there is a need of any kind of Surgery. This dataset is also containing
the doctor’s advice for whether the Heart is HEALTHY, MEDICATION and SURGERY based on the
outcomes of the three tests.
You are required to create various kinds classifiers classifying patients in categories of
Healthy, need medication or undergo surgery.
The snapshot of the dataset is shown in Figure 2
<p align="center">
  <img src="https://github.com/mayur-aggarwal/machine-learning/blob/master/medical_table_sample_dataset.png">
</p>
<p align="center">Figure 2: Heart Health Test Dataset
</p>

* #### *Problem statement 3:*
  * #### _**Train Selection**_
Indian Railways has introduced a new luxury train from Mumbai to New Delhi. This train has
all facilities like WiFi, Club, Lounge, Playing, SPA etc. Each of the facilities are chargeable
along
with the travel fare. To analyze the interest shown by public, they floated a form with
information such as Age, Sex, fare paid, number of members traveling with, Travel class etc.
This form was filled by the person while booking the ticket for the train. After the first day
launch of the train, the department analyzed whether the person has boarded the train or
not.
The dataset that is provided contains all the information about the person along with
whether the person as boarded the train or not. You need to create classifiers to classifying
whether a person will board the train or not if provided with information such as age, fare
paid, number of members traveling with etc.
The snapshot of the dataset is shown in Figure 3.
<p align="center">
  <img src="https://github.com/mayur-aggarwal/machine-learning/blob/master/train_sample_dataset.png">
</p>
<p align="center">Figure 3: Train Selection Dataset
</p>
