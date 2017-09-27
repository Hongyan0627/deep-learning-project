# Recurrent Network For Lung Cancer Classification -2017 Kaggle Data Science Competition
## Abstract
We participated in the Kaggle lung cancer detection challenge. The aim is to use chest CT scans to predict whether or not a patient has lung cancer. We tested a simple tree network that takes advantage of transfer learning, and tried to improve upon this by using RNN across the scan slices to allow the network to learn 3D information and to manage the multiple instance nature of detecting any nodules in hundreds of scans. Our results demonstrate some effectiveness of the simple tree network, but our RNN strategies offered no improvement.

## Approaches
Two approaches were applied in our project. One approach is CNN plus LightGBM, another approach is CNN plus LSTM layer.
### CNN + LightGBM
  In our project, we use two state-of-the-art pre-trained CNN models, ResNet50 and VGG19 from keras with Tensor-flow backend are used as featurizers. For image classification tasks, the first few layers of a CNN represent lower level features and later layers represent high level features which is specific for image classification. In our model, we extract features from penultimate layer outputs using Keras predict function.
  LightGBM is a gradient boosting framework which uses
tree based learning algorithms. Features generated from the pre-trained models are fed into the LightGBM classifier.
Model structure will be in this link https://user-images.githubusercontent.com/16090170/30889877-512d470c-a2f7-11e7-9cbf-29fb27098104.png

### CNN + LSTM
CNN pretrained model same as first approach. The idea was that using recurrence with RNN LSTM layers would allow the network to learn the 3D dependencies between slices, and also manage the fact that cancerous nodules only appear on some slices, but we want an output per patient. The use of RNN also allows for a dynamic ‘depth’, so we will not need to down or upsample the data into a consistent size. This is especially useful since the number of slices per patient varies substantially.
Model structure is in this link https://user-images.githubusercontent.com/16090170/30889938-b154e176-a2f7-11e7-8846-85e914ed6852.png
## Experiments
### Dataset
In this study, we used the stage 1 CT imaging data provided by the National Cancer Institute of Kaggle 2017 Lung Cancer competition.
Training data is large, about 140G. In this dataset, it includes 285380 CT images from 1595 patients, every patient has an patient id and every id includes a set of 2D slices
### Evaluation metrics
Evaluation using Log Loss and ROC curve
### Results
transfer learning worked up to a point. Using the boosted tree on top of features generated on pretrained networks, we were able to create a working network, but performance was far from state of the art.We added the RNN blocks with the aim of having the
network learn spatial knowledge of cancerous nodes, and also to learn that there only needs to be some slices with cancer to predict that the entire patient has cancer, but was unsuccessful in this project.

## Code 
Code was run in Jupyter Notebook on tufts AWS machine.
### ResNet_Vgg19.ipynb
Features used is based on both ResNet50 and VGG19, which mean the dataset is big.
### RestNet-label_image.ipynb
Features used is extracted based on every image of every patient via pre-trained ResNet50 model. So in data preprocessing, we label every image first.
### RestNet_final.ipynb
Features used is extracted based on every patient via pre-trained ResNet50 model. So the dataset is smaller than label every image.
### vgg19.ipynb
Features used is extracted based on every patient via pre-trained VGG19 model. So the dataset is smaller than label every image.
### vgg19_label_image.ipynb
Features used is extracted based on every patient via pre-trained VGG19 model. So the dataset is larger than label every image.
### RNN.ipynb
Add LSTM layer to do classification. All others used LightGBM to do classification.
