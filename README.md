# CIFAR-10-

### Team members

Kaoutar Benazzou | Aya Lyousfi | El Mehdi Ziate


## Introduction

Computer vision is a field of artificial intelligence that has various interesting applications. This project specifically focuses on using image classification using machine learning (ML). The principal goal of image classification is to train a computer to recognize and categorize objects or certain chosen patterns within images, as if replicating human visual perception. This requires the use of  ML algorithms to analyze the features and characteristics of images, thus the system becomes capable of making predictions about the content or the class to which an image is part of. Image classification using ML has seen important advancements during the past years with the improvement of deep learning, especially Convolutional Neural Network (CNNs). These models excel at capturing patterns and hierarchical representations within images, making them practical and convenient for image classification tasks. The continuous evolution of ML techniques ensures that image classification remains a dynamic and rapidly evolving field with numerous practical applications across industries such in automotive, healthcare, finance, retail, etc.

## Project Description

For the final project this semester, we will be working on building the model using different layers of connected CNNs. We will be using a dataset from Kaggle (CIFAR-10) to train our model. Then, we will be classifying images using the model and evaluating its performance. The steps followed during this project are as described in the instructions sheet:

1. Get familiar with topic and resources.
2. Watch the video given in the instructions and add the source code to our Google Colab file.
3. Download the dataset from Kaggle and use it in our Google Colab code and experiement with the code and hyperparameters.
4. Assess, discuss, understand, and analyze observed results.

## Image Classification Steps

Image classification is a set of steps and a workflow that relies on a previous step to complete the next. First, data collection is necessary where the labelled images are gathered and prepared (by image resizing, normalization, and data augmentation). The data collected can be divided into: 

→ Training Set: which is approximately 70% to 80% of the total dataset to train the ML model, allowing it to learn patterns and relationships within the data.

→ Validation Set: which is typically around 10% to 15% of the total dataset to fine-tune the model during training to adjust hyperparameters and preventing overfitting.

→ Test Set: which is the remaining portion, usually 10% to 20% of the total dataset to evaluate the model's performance on unseen data and assess how well the model generalizes to new, previously unseen examples.

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/9f6d4a16-21d3-4dcb-8c25-5587014c7d76)


 Then, using techniques like CNNs we can extract relevant features from the input data to train the model. During the training, the algorithm adjusts its parameters based on the input data given to it and learns how to differentiate between different classes. The model's performance is then evaluated on a new set of images to test and verify its accuracy and get insights into its ability to correctly classify objects. Then fine-tuning is often a necessary step to enhance the model’s performance by adjusting hyperparameters, experimenting with different architectures, or incorporating regularization techniques to prevent overfitting. Finally, the model can be deployed to classify new images.

## Convolutional Neural Network (CNNs)

Neural networks, a subset of machine learning and central to deep learning, consist of node layers, including an input layer, hidden layers, and an output layer. Nodes connect with associated weights and thresholds. If a node's output exceeds the threshold, it activates and transmits data to the next layer; otherwise, no data passes through. There are various types of neural nets, which are used for different use cases and data types. Recurrent neural networks excel in natural language processing and speech recognition, while CNNs are prevalent in classification and computer vision. Unlike traditional methods, CNNs streamline image classification and object recognition by leveraging matrix multiplication principles. Despite their computational demands, often necessitating GPUs for training (even in our case, which made the training relatively faster), CNNs offer a scalable approach to visual tasks. CNNs processes input images by assigning importance, represented by learnable weights and biases, to different aspects or objects within the image, allowing for effective differentiation. Unlike many classification algorithms, CNNs require less pre-processing, as they can learn filters and characteristics through training, eliminating the need for hand-engineered filters. The architecture of CNNs mirrors the connectivity pattern of neurons in the human brain and draws inspiration from the organization of the Visual Cortex. This neural network design is such that individual neurons respond to stimuli within specific regions known as Receptive Fields. The combination of these fields creates overlapping coverage across the entire visual area, enhancing the network's ability to analyze and understand complex visual information.

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/f616e945-4d64-4f75-b257-46a7bcf2c2b6)

## Kaggle

Kaggle is a popular online platform for data science competitions, collaborative data science projects, and learning. It provides a platform where data scientists, researchers, and enthusiasts can access datasets, participate in competitions, and share insights. Kaggle's competitions often involve real-world problems and large datasets, making it a valuable resource for honing data science and machine learning skills. Some of its features are:

- Diverse range of datasets covering various domains.
- Competitions with real-world challenges hosted by companies and organizations.
- Collaboration features for team-based projects.
- Notebooks for coding and analysis, allowing users to share code and findings.
- Educational resources, forums, and discussions to facilitate learning and knowledge exchange.

In this project we will focus on CIFAR-10 dataset.

## CIFAR-10 dataset

 

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/a51ced18-dc9f-46d8-a065-ccedf1d5670a)


The CIFAR-10 dataset comprises 60,000 32x32 color images distributed across 10 classes, each containing 6,000 images. The dataset is divided into 50,000 training images and 10,000 test images, maintaining the original train/test split. The provided files include:

- **train.7z:** A folder containing training images in PNG format.
- **test.7z:** A folder containing test images in PNG format.
- **trainLabels.csv:** The file containing labels for the training images.

It's essential to note that 290,000 additional images, referred to as "junk images," have been introduced in the test set to discourage certain forms of cheating, such as hand labeling. These junk images are disregarded during scoring. Trivial modifications have also been made to the official 10,000 test images to prevent identification by file hash, but these modifications are not expected to significantly impact scoring.

The label classes in the dataset are as follows:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each class is mutually exclusive, with no overlap between automobile and truck categories. Specifically, "Automobile" encompasses sedans and SUVs, while "Truck" includes only large trucks, excluding pickup trucks. The objective is to predict labels for all 300,000 images in the dataset.

## Tenserflow
![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/baea1aa4-1071-4822-aba1-34a6fb2da44b)


TensorFlow is an open-source machine learning framework developed by Google, renowned for its versatility in building and training diverse types of machine learning models, particularly neural networks. It operates on a foundational concept of constructing computational graphs, where nodes represent mathematical operations and edges depict the flow of data through these operations. TensorFlow empowers users with both high-level APIs, like Keras, for simplified model creation, and lower-level APIs for fine-grained control over model architectures and training processes. Its flexibility, efficiency in handling large-scale computations across multiple devices, extensive documentation, and vibrant community support make it a leading choice for developing and deploying machine learning models across various domains.

## Keras

Keras is an open-source high-level neural networks API written in Python. It acts as an interface for building, training, and deploying deep learning models. Keras is designed to be user-friendly, modular, and extensible, allowing both beginners and experienced researchers to work with complex neural network architectures effortlessly. Some of its best features are:

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/5915decb-3e6c-4892-b266-beee396c343d)


- Simple and intuitive interface.
- Compatibility with various backends, including TensorFlow and Microsoft Cognitive Toolkit (CNTK).
- Support for both convolutional and recurrent neural networks.
- Extensibility for advanced users to create custom layers and models.
- Widely used for applications like image and speech recognition, natural language processing, and more.

## Hyperparameters

- **Learning Rate:** The learning rate is a crucial hyperparameter determining the step size taken during the optimization process. A higher learning rate might speed up convergence but risks overshooting the optimal solution, while a lower rate might lead to slower convergence but potentially better precision in finding the minimum. It significantly impacts model training speed, stability, and convergence.
- **Batch Size:** Batch size determines the number of samples processed before updating the model's parameters. Smaller batch sizes lead to more frequent updates but might increase training time due to inefficiency in utilizing computational resources. Conversely, larger batch sizes might speed up training but could hinder generalization or convergence to the optimal solution.
- **Epochs:** Epochs signify the number of complete iterations over the entire dataset during training. Fewer epochs might lead to underfitting, where the model fails to capture complex patterns in the data, while too many epochs could cause overfitting, where the model learns noise instead of generalizable patterns.
- **Optimizer:** The optimizer governs the method by which the model's weights are updated based on gradients computed from the loss function. Various optimizers, like Adam or RMSprop, have different strategies for adjusting weights, impacting the speed and quality of convergence. Some optimizers offer adaptive learning rates or momentum to navigate complex loss surfaces efficiently.

## Data Visualization

### Loading the data

- As the first step that we need to take before training our model, we need to understand and discover the data we are working with , therefore we had to load the data into our shared notebook shared in GoogleCollab.
- We had the choice to download the whole data from the Kaggle website into our local machines and upload it to the could to us it in GoogleCollab, but we figured out that we can load the dataset using the following code:
    
    ```python
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    ```
    

### Printing the shapes:

- Forwarding, we needed to check the format/shape of both the test and training datasets with their associated labels, which is a common practice in machine learning and deep learning to inspect the dimensions or shapes of the datasets before feeding them into models. Since, understanding the shapes of the datasets is essential to confirm that they align with the expected structure, ensuring compatibility when building and training machine learning models. It's also a way to catch potential issues, such as mismatched dimensions, before feeding the data into models. Also, we were not sure that importing the data using tensorflow will give us results matching the same description of the CIFAR-10 data (Being sure).
    
    ```python
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape)
    ```
    
- As a result we have the following shapes:
    
    ```python
    x_train shape: (50000, 32, 32, 3) y_train shape: (50000, 1)
    x_test shape: (10000, 32, 32, 3) y_test shape: (10000, 1)
    ```
    
- **`x_train shape: (50000, 32, 32, 3)`**:
    - There are 50,000 samples in the training set.
    - Each sample is an image represented by a 32x32 grid.
    - The '3' at the end signifies that each pixel in the image has three values, likely representing the RGB (Red, Green, Blue) color channels.
- **`y_train shape: (50000, 1)`**:
    - Corresponding to the 50,000 samples in the training set, there is one label per sample.
    - The '1' indicates that each sample is associated with a single label.
- **`x_test shape: (10000, 32, 32, 3)`**:
    - There are 10,000 samples in the testing set.
    - Each sample is also a 32x32 image with three color channels.
- **`y_test shape: (10000, 1)`**:
    - Corresponding to the 10,000 samples in the testing set, there is one label per sample.
- What we can conclude is that the dataset is divided into a training and validating sets with 17% for validating and 83% to train, which is very close to the standard (80% for training and 20% for validating).

### Printing some samples:

```python
data_dict = {0 : "Airplane",
             1 : "Car",
             2 : "Bird",
             3 : "Cat",
             4 : "Deer",
             5 : "Dog",
             6 : "Frog",
             7 : "Horse",
             8 : "Ship",
             9 : "Truck",
            }

def get_10_random_images():
  d = {}
  while len(d.keys()) < 10:
        idx = random.randint(0, len(y_train)-1)
        if y_train[idx][0] not in d.keys():
            d[y_train[idx][0]] = idx
  return d

d = get_10_random_images()
rows= 2
cols=5
fig,axes = plt.subplots(rows, cols, figsize = (15,6))
axes = axes.ravel()
for i in range(1, rows*cols+1):
    axes[i-1].imshow(x_train[d[i-1]])
    axes[i-1].axis('off')
    axes[i-1].set_title(data_dict[i-1], fontsize=15)
fig.suptitle('10 Random Images from Dataset of each class', fontsize = 20)
plt.show()
```

- The code above defines a function **`get_10_random_images()`** that aims to retrieve ten random images from the dataset, each belonging to a different class. The **`data_dict`** maps the class indices to their respective labels. The function initiates an empty dictionary **`d`**, then iterates to randomly select indices from the training dataset (**`x_train`** and **`y_train`**). For each randomly chosen index, if its associated label hasn't been encountered before, it adds the index to the dictionary, associating it with its corresponding label. Once it collects ten unique indices for each class, it displays a grid of ten images (two rows and five columns) using Matplotlib. Each image is plotted alongside its associated class label from the **`data_dict`**. The resulting visual representation presents a selection of ten random images, one from each class, providing a snapshot of the diversity within the dataset.
- Here is the result

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/ade87bbb-3911-4373-b84b-ef816728b285)


## Models

- To decide on the architecture of our model and tune it, we had to go through a various trial phases. Therefore we had to manually tune the architecture by trying 3 different architectures to achieve a better result.

### Manual Tuning

Manual hyperparameter tuning is an approach to fine-tune the hyperparameters of a machine learning model through hands-on experimentation. This method requires iteratively adjusting the hyperparameters and assessing the model's performance until desired outcomes are obtained. While it may be time-intensive, manual tuning offers the flexibility to explore different hyperparameter combinations and customize them according to specific datasets and objectives.

### Model 1

```python
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
```

- **Architecture**:
    - **Convolutional layers**
        - The model starts with a convolutional layer (**`Conv2D`**) with 32 filters, a kernel size of (3, 3), and ReLU activation. This layer processes the input images, which have a shape of (32, 32, 3) representing a 32x32 RGB image.
        - A max-pooling layer (**`MaxPooling2D`**) follows with a pool size of (2, 2), reducing the spatial dimensions.
    - **Additional Convolutional layers**
        - Two convolutional layers followed by a max-pooling layers were added. The second convolutional layer has 64 filters, and the third convolutional layer also has 64 filters. Both use a (3, 3) kernel size and ReLU activation.
    - **Flatten Layer:**
        - After the convolutional layers, a flatten layer is added to convert the 3D output to a 1D vector. This is necessary before feeding the data into fully connected layers.
    - **Dense Layers:**
        - Two dense (fully connected) layers follow the flatten layer. The first dense layer has 64 units with ReLU activation, and the second dense layer has 10 units (output layer) without an activation function, indicating a multi-class classification problem with 10 classes (assuming a softmax activation is used during training).
- **Summary**
    - **Total Parameters:**
        - The model has a total of 122,570 parameters.
    - **Trainable Parameters:**
        - All parameters in the model are trainable, indicating that the model is designed to learn and adjust these parameters during training.
    - **Non-trainable Parameters:**
        - There are no non-trainable parameters in this model, meaning that all parameters will be updated during the training process.
- **Results**
    - After fitting the model using the following code that initializes the model using TensorFlow and Keras, configuring it for training with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric. The model is then trained on the specified training data for 25 epochs, and its performance is evaluated on a separate validation dataset. The sparse categorical crossentropy loss is chosen for multi-class classification, with the model outputting logits. Throughout training, the model adapts its parameters to minimize the loss and improve accuracy on the training and validation sets. Once trained, the model is ready for making predictions on new data, having learned patterns and relationships from the training dataset.
    
    ```python
    model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=25,validation_data=(x_test, y_test))
    ```
    
        We got the following results:
    
![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/5fec2046-a62e-4809-9544-9602d646a795)
    
    - However, by looking at each step of the epoch, in the training history provided, a noticeable pattern of overfitting emerges around epoch 15. While the training loss continues to decrease, and training accuracy improves, the validation loss starts to rise, and the validation accuracy either plateaus or slightly decreases. This divergence indicates that the model is becoming too specific to the training data and is failing to generalize well to unseen data, a classic symptom of overfitting.
    - Therefore we decided to improve this architecture and use dropout or L2 regularization to prevent the model from relying too heavily on specific features in the training set.

### Model 2:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Added another convolutional layer
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()
```

- **Architecture**:
    
    The architecture of this is similar to the one before we only added a new convolutional layer and a dropout layer applied after the first Dense layer with a dropout rate of 0.5. Dropout helps prevent overfitting by randomly setting a fraction of input units to zero during training. 
    
- **Summary**
    - **Total Parameters:**
        - The total number of parameters used in the provided model is 167,818.
    - **Trainable Parameters:**
        - All parameters in the model are trainable, indicating that the model is designed to learn and adjust these parameters during training.
    - **Non-trainable Parameters:**
        - There are no non-trainable parameters in this model, meaning that all parameters will be updated during the training process.
- **Results**
    - We used the same code in the previous model to compile and fit the model but we reduced the number of epochs to 15 and we got the following results:
    
    ![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/041b6677-4b4d-4a94-bd51-ba23a86cd8db)

    
    - As we can see the model shows decent performance on the training set, but the gap between training and validation accuracy suggests overfitting. The model might be too complex or not getting enough diverse examples to generalize well.
    - Therefore we decided to refine our model more and look for a more suitable approach especially the architecture we are using.

### Model 3

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras
from keras import regularizers
y_train_cat = keras.utils.to_categorical(y_train,10)
y_test_cat = keras.utils.to_categorical(y_test,10)
weight_decay = 1e-4
input_shape = x_train.shape[1:]

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='same', input_shape = input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units = 10, activation = 'softmax'))
```

- **Categorical Encoding**
    - The **`y_train`** and **`y_test`** label vectors are converted to binary class matrices for use with categorical crossentropy during training. This is necessary for multi-class classification when the labels are integers.
- **Weight Decay**:
    - A weight regularization parameter is defined to prevent overfitting. Regularization techniques such as L2 regularization (weight decay) penalize large weights in the model.
- **Input Shape**:
    - The shape of the input data (**`x_train`**) is used to specify the input shape of the first layer in the neural network.
- **Convolutional Layers**:
    - Several **`Conv2D`** layers are added, specifying the number of filters, kernel size, padding type, and kernel regularizer. These layers will extract features from the input image by applying filters.
        - The first **`Conv2D`** layer with 32 filters.
        - A second **`Conv2D`** layer with 32 filters immediately after the first layer.
        - The third **`Conv2D`** layer with 64 filters following the first dropout layer.
        - A fourth **`Conv2D`** layer with 64 filters right after the third layer.
        - The fifth **`Conv2D`** layer with 128 filters after the second dropout layer.
        - A sixth **`Conv2D`** layer with 128 filters immediately following the fifth layer.
- **Batch Normalization**:
    - This normalizes the activations of the previous layer at each batch, i.e., it applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
- **Pooling Layers**:
    - **`MaxPooling2D`** layers are added to reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer. This is done by selecting the maximum value in each window of a predefined size (2x2).
- **Dropout**:
    - Dropout layers are added to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
- **Flatten Layer**:
    - The **`Flatten`** layer is used to flatten the input and does not affect the batch size. This is necessary to transition from the 2D feature maps to 1D feature vectors for the dense layers that follow.
- **Dense Layer**:
    - A **`Dense`** layer with 10 units is added to output the final predictions. This layer is the output layer and uses the 'softmax' activation function to output probabilities for the 10 classes.
- **Summary:**
    
   ![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/7412f8e2-d56f-48bb-ba63-efd49bc09dee)

    
   ![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/9f321457-b52d-423a-9f6c-c50e11f1cc3d)

    
    - **Total Parameters:**
        - The total number of parameters used in the provided model is 309,290.
    - **Trainable Parameters:**
        - The number of trainable parameters is 308,394
    - **Non-trainable Parameters:**
        - The number of non-trainable parameters is 896 which is due to the referring to these parameters associated with BatchNormalization, Dropout, and Activation layers. While they play a crucial role during training, they are not part of the primary weight updates and are not fine-tuned during the optimization process.
- To compile, run and save the model we use the following code:

```python
from tensorflow.keras.optimizers import legacy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_delta=0.0001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=legacy.RMSprop(learning_rate=0.001, decay=1e-6),
    metrics=['accuracy']
)

epochs_hist = model.fit(x_train, y_train_cat, batch_size=64, epochs=50, validation_split=0.2, shuffle=True, callbacks=[early_stopping, reduce_lr])
model.save("/content/drive/My Drive/Model.h6")
```

- The model is trained with a batch size of 64, which means that 64 examples are used to estimate the error gradient before the model weights are updated.
- The training set is split such that 80% is used for training and 20% is used for validation, as indicated by **`validation_split=0.2`**. The validation set is not used for updating model weights but to evaluate the model's performance on unseen data after each epoch.
- **`shuffle=True`** indicates that the training data is shuffled before each epoch, which helps to prevent the model from learning any order-dependent patterns in the training dataset.
- The loss and accuracy are reported for both the training set (**`loss`** and **`accuracy`**) and the validation set (**`val_loss`** and **`val_accuracy`**) after each epoch.
- We also defined and early stopping and reduce_lr so we can stop the model when it will start overfitting.

## Results of Model 3:

### Plotting the accuracy and loss (train and validation)

```python
from tensorflow.keras.models import load_model

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(epochs_hist.history['val_accuracy'], label = 'CV Acc')
plt.plot(epochs_hist.history['accuracy'], label = 'Training Acc')
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_hist.history['val_loss'], label = 'CV Loss')
plt.plot(epochs_hist.history['loss'], label = 'Training Loss')
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
```

- The code snippet above is used for plotting the accuracy and loss of a machine learning model during training and validation across epochs. The plotted graphs are a common way to visualize the model's learning progression over time.

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/94200595-921b-40b6-96d3-1f19b59e0b6a)

- **Learning Rate Adjustments**: The model's learning rate was initially set to 0.001 and was reduced by a factor of 0.2 whenever the validation loss stopped improving for 2 epochs (as seen by the 'ReduceLROnPlateau' callback activations). This helped the model to fine-tune its weights and potentially avoid overshooting the minima in the loss landscape.
- **Early Stopping**: The training was stopped early at epoch 32 instead of running all 50 epochs because the validation loss did not improve for 5 consecutive epochs. This is a mechanism to prevent overfitting and to save computational resources. The 'EarlyStopping' callback restores the weights from the epoch with the best validation loss, which was epoch 27 in your case.
- **Accuracy Graph**: The accuracy graph shows the training and cross-validation (CV) accuracy over epochs. Both accuracies increase over time and begin to plateau, indicating that the model is converging. The cross-validation accuracy closely follows the training accuracy, which suggests that the model is generalizing well and not overfitting significantly.
- **Loss Graph**: The loss graph shows the training and cross-validation loss over epochs. Both losses decrease over time and begin to flatten out, which also indicates convergence. The cross-validation loss is slightly higher than the training loss, which is typical in model training.
- **Performance**: By the end of training, the model achieved a cross-validation accuracy of approximately 84.64% (epoch 31), which is quite high. The training accuracy at this point was slightly higher, at around 89.40%, which shows the model might be starting to overfit since the training accuracy is significantly higher than the validation accuracy.
- **Optimization**: The 'ReduceLROnPlateau' callback was triggered several times (epochs 6, 11, 18, 27, 29, 31), each time reducing the learning rate to help the model to continue learning and improving.
- **What is next?**: As we mentioned before, for the choose of the hyperparameters, it was done manually (manual tuning), and we magically got some decent results, but we want to fine tune the hyperparameters using more advanced techniques (random search, grid …) and maybe we may get better results.

## Fine Tuning Hyperparameters

- As mentioned previously we want to increase the accuracy of the model so we will try ti fine tune 4 hyperparameters as shown in this code:

```python
from keras.optimizers import Adam
from sklearn.model_selection import ParameterGrid
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras import regularizers
import matplotlib.pyplot as plt

# Define hyperparameter grid with reduced variations
hyperparameters = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
    'epochs': [10, 20],
    'optimizer': ['adam', 'rmsprop']
}

# Randomly select combinations
random.seed(42)  # Set seed for reproducibility
num_combinations = 5  # Choose the number of combinations to try

selected_combinations = random.sample(list(ParameterGrid(hyperparameters)), num_combinations)

results = {}

for params in selected_combinations:
    print("Training with hyperparameters:", params)
    # the model code here:
		# ......
    # Compile the model with the given hyperparameters
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    else:
        optimizer = legacy.RMSprop(learning_rate=params['learning_rate'], decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train_cat,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_split=0.2,
                        shuffle=True,
                        verbose=0)  # Set verbose to 0 for less output

    # Store performance metrics
    results[str(params)] = history.history

# Plot performance as a function of the chosen hyperparameters
plt.figure(figsize=(10, 6))
for params, result in results.items():
    plt.plot(result['val_accuracy'], label=str(params))
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Fine-tuning Hyperparameters')
plt.show()
```

- **Hyperparameter Grid**:
    - **`learning_rate`**: Determines how much to change the model in response to the estimated error each time the model weights are updated. Choices are 0.001 and 0.01.
    - **`batch_size`**: The number of training samples to work through before the model's internal parameters are updated. The options are 32 and 64.
    - **`epochs`**: The number of complete passes through the training dataset. The options are 10 and 20.
    - **`optimizer`**: The mechanism for updating the weights. The options are 'adam' and 'rmsprop'.
- **Random Selection of Combinations**:
    - A random seed is set for reproducibility so that the same random combinations can be generated in future runs.
    - A subset of 5 combinations of hyperparameters is selected from the complete grid to reduce computation time compared to testing all possible combinations.
- **Regularization**:
    - **`regularizers.l2`** is used to apply L2 regularization, which penalizes large weights and helps prevent overfitting. The **`weight_decay`** parameter controls the strength of this regularization but is undefined in the provided code snippet and should be set before running the code.
- **Model Compilation and Training**:
    - For each set of hyperparameters, the model is compiled with the specified optimizer and learning rate.
    - The model is trained with the given batch size and number of epochs, and with 20% of the training data held out for validation (**`validation_split=0.2`**).
- **Performance Tracking and Visualization**:
    - The training process is run quietly with **`verbose=0`**, meaning no output is shown during training.
    - The training histories are stored in the **`results`** dictionary.
    - After all combinations are evaluated, a plot is created showing the validation accuracy for each set of hyperparameters across epochs.
- **Outcome**:
    - The visualization at the end will help identify which hyperparameter combinations lead to the best performance on the validation set.

### Graph:

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/5782e882-4a28-4197-8050-b3ed66bc3ce5)


- **Red Line (Batch size: 32, Epochs: 20, Learning rate: 0.01, Optimizer: 'rmsprop')**:
    - This combination exhibits the highest validation accuracy after 20 epochs. It appears to be the best performing set of hyperparameters among those presented.
- **Orange Line (Batch size: 32, Epochs: 10, Learning rate: 0.001, Optimizer: 'adam')**:
    - This line shows a rapid increase in accuracy but seems to plateau or slightly decrease after about 6 epochs, which might indicate that the learning rate could be too low to continue making significant progress or that the epochs are not sufficient.
- **Green Line (Batch size: 64, Epochs: 10, Learning rate: 0.01, Optimizer: 'rmsprop')**:
    - This set starts off well but does not maintain the same level of performance as the red line. This could suggest that a smaller batch size (such as 32) is more beneficial for this specific task or that more epochs are needed for a larger batch size to converge.
- **Blue Line (Batch size: 32, Epochs: 20, Learning rate: 0.001, Optimizer: 'adam')**:
    - This curve is quite erratic, showing substantial fluctuations in validation accuracy, which might suggest that the model could benefit from a more stable learning rate or an adjusted optimization strategy.Purple Line (Batch size: 64, Epochs: 20, Learning rate: 0.01, Optimizer: 'rmsprop'):
- **Purple Line (Batch size: 64, Epochs: 20, Learning rate: 0.01, Optimizer: 'rmsprop')**:
    - Despite having the same number of epochs and learning rate as the best performing model (red line), this combination with a larger batch size performs worse, suggesting that the model with a batch size of 64 might require a different learning rate or more epochs.
- In conclusion, based on the information available from the graph, the hyperparameters used for the red line (batch size: 32, epochs: 20, learning rate: 0.01, optimizer: 'rmsprop') would be the recommended choice as they yield the highest validation accuracy after 20 epochs.
- After Finetuning the hyperparameters we decided on training using the suggested ones:

```python
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
# Adjusted factor and patience for a more conservative learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_delta=0.0001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.01),  # Updated learning rate
    metrics=['accuracy']
)

# Keep the batch size and epochs as previously set
epochs_hist = model.fit(x_train, y_train_cat, batch_size=32, epochs=40, validation_split=0.2, shuffle=True, callbacks=[early_stopping, reduce_lr])

# Ensure you have the right path for saving the model
model.save("/content/drive/My Drive/Model.h5")
```

- We got the following accuracy:

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/9b3bd31b-70cf-4f36-9fd2-e533e97b3b77)


- The accuracy and loss for both training and cross-validation are relatively close, which is a good sign. It implies there's no significant overfitting.
- The spikes in loss, particularly in the cross-validation loss, are somewhat unusual. This could be due to:
    - The model experiencing difficulty with certain batches of data.
    - High learning rate causing instability in the training process.
    - A small validation set that does not represent the overall data distribution well, leading to high variance in validation results.
- In order to handle those spikes we have decided to implement a learning rate scheduling to decrease the learning rate as training progresses.

```python
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import numpy as np

# Function for learning rate scheduling
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

# Learning Rate Scheduler callback
lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Adjusted ReduceLROnPlateau callback for a more conservative learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_delta=0.0001)

# Compile the model with initial learning rate
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.01),  # Initial learning rate
    metrics=['accuracy']
)

# Train the model with the learning rate scheduler, early stopping, and reduce learning rate on plateau callbacks
epochs_hist = model.fit(
    x_train, y_train_cat,
    batch_size=32, epochs=40,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stopping, reduce_lr, lr_scheduler]  # Add lr_scheduler to the callbacks
)
# Save the model
model.save("/content/drive/My Drive/Model.h5")
```

- and we got the following final accuracy graph:

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/8f6e49ae-1f99-474b-9d43-d8db696b2b21)


- The implementation of the learning rate scheduler has led to a more stable training process, as evidenced by the smoother decline in validation loss.
- The absence of significant spikes in the later epochs of training suggests that the learning rate adjustments were well-timed and effective at maintaining a steady progression of the model's learning.
- The model appears to have a good fit, with neither overfitting nor underfitting, as the training and validation lines closely follow each other.

## Comparison between model 3 and fine tuned model:

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/30df681d-2f18-4c2d-9f5a-015e83696bf7)


First set of graphs (Fine tuned model)

![image](https://github.com/elmehdiziate/CIFAR-10-/assets/109172506/fb6ff805-804c-44ed-bccc-9fdfdb1fdc9a)


Second set of graphs (Model 3)

- **Model Accuracy Graphs**:
    - The first set of graphs shows that both the training and cross-validation (CV) accuracy steadily increase as the number of epochs increases, with CV accuracy slightly higher than training accuracy for most of the training process. This suggests a well-fitting model that generalizes well to unseen data.
    - The second set of accuracy graphs also shows an increase in accuracy with the number of epochs, but in this case, the training accuracy surpasses the CV accuracy early on and remains higher throughout. This could indicate a slight overfitting as the model may be learning the training data too well and thus might not generalize as effectively to new, unseen data.
- **Model Loss Graphs**:
    - The first set of loss graphs shows that both training and CV loss decrease sharply at the beginning and then level off, with the CV loss slightly higher than the training loss. The leveling off at a higher loss value could suggest that the model has reached its capacity in terms of learning from the data provided.
    - In the second set of loss graphs, there's a sharp decrease in both training and CV loss initially, and both losses level off similarly to the first set. However, the training loss becomes lower than the CV loss earlier and remains lower, which, as with accuracy, suggests overfitting.
- **Key Observations**:
    - The first model (first row of graphs) seems to be more balanced in terms of accuracy and loss for training and CV, suggesting good generalization.
    - The second model (second row of graphs) seems to have better performance on the training set but potentially worse generalization as indicated by the higher CV loss and lower CV accuracy compared to the training metrics.
    - The number of epochs for the second model is fewer (30) compared to the first one (40), which might suggest that the second model could potentially improve further with more epochs, or it might start to overfit even more.
    - The scales of the graphs are similar but not identical, making direct numerical comparison challenging without the exact values.
- Accuracy comparaison table:

|  | Training accuracy | Testing Accuracy |
| --- | --- | --- |
| Model 3 | 0.8941 | 0.8462 |
| Fine tuned | 0.8473 | 0.8261 |

## Demo

[Project 4](https://www.youtube.com/watch?v=buV2AnmLuWw)

## Resources

- [https://www.tensorflow.org/tutorials/images/classification](https://www.tensorflow.org/tutorials/images/classification)
- [https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjsl96T0_-CAxXgTqQEHWi7CzcQFnoECAsQAQ&url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2021%2F07%2Fstep-by-step-guide-for-image-classification-on-custom-datasets%2F&usg=AOvVaw2GEAl2MObrhpdqXYm2DVg3&opi=89978449](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjsl96T0_-CAxXgTqQEHWi7CzcQFnoECAsQAQ&url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2021%2F07%2Fstep-by-step-guide-for-image-classification-on-custom-datasets%2F&usg=AOvVaw2GEAl2MObrhpdqXYm2DVg3&opi=89978449)
- [https://www.ibm.com/topics/convolutional-neural-networks](https://www.ibm.com/topics/convolutional-neural-networks)
- [https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [https://keras.io/about/](https://keras.io/about/)
- [https://www.coursera.org/articles/kaggle](https://www.coursera.org/articles/kaggle)
- @misc{cifar-10,
author = {Will Cukierski},
title = {CIFAR-10 - Object Recognition in Images},
publisher = {Kaggle},
year = {2013},
url = {[https://kaggle.com/competitions/cifar-10](https://kaggle.com/competitions/cifar-10)}
}
- [https://blog.roboflow.com/what-is-hyperparameter-tuning/#:~:text=Manual hyperparameter tuning is a,until satisfactory results are achieved](https://blog.roboflow.com/what-is-hyperparameter-tuning/#:~:text=Manual%20hyperparameter%20tuning%20is%20a,until%20satisfactory%20results%20are%20achieved).
