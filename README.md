# Oil Palm Disease Detection

![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

This repo is the source code of my research regarding disease detection from oil palm images. Try out the prediction at [oil-palm-stem-prediction.fly.dev](https://oil-palm-stem-prediction.fly.dev).

The prediction is performed using a machine learning model, SVM, with linear kernel. The feature used by the model is extracted from images using the [ResNet50 pre-trained model](https://keras.io/api/applications/resnet/#resnet50-function). Unfortunately, I can't give the dataset used for training the model as it is private data. But you can use your own dataset to train the model. Put the dataset in the `dataset/selected_dataset` folder. Inside it, make sure it has both `infected` and `normal` folder.

Check how the feature extraction is performed in `deep_feature_extractor.ipynb` notebook. The model training is performed in `skripsi.ipynb` notebook.