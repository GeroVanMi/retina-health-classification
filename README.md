# Retinal Imaging
## Medical Image Classification
### Data Science in Health
#### Gérome Laurin Meyer | Rebekka von Wartburg
#### May 24, 2024
![Titelbild](img/title_img.jpeg) [1]

---

## Introduction
In this project, the application of a Convolutional Neural Networks (CNNs) classification of three major eye diseases - **cataract**, **retinopathy** and **glaucoma** - is investigated. These diseases are among the most common causes of visual impairment and blindness. Early detection and diagnosis can slow down the progression of the disease, preserve vision and generally improve the quality of life of those affected.


* **Cataract:**
A cataract is where the lens of the eye becomes opaque, clouding and losing the ability to focus. It is often associated with ageing and can also be caused by factors such as diabetes, smoking and prolonged exposure to sunlight.

* **Glaucoma:**
Glaucoma occurs when the fluid pressure inside the eye rises above normal levels. This increased pressure can damage the optic nerve, which is responsible for transmitting visual information from the eye to the brain. If left untreated, glaucoma can lead to loss of vision or even blindness.

* **Retinopathy:**
Retinopathy is a disease of the retina, the light-sensitive tissue at the back of the eye. It is often associated with diabetes and can damage the blood vessels in the retina, which can result in vision loss or even blindness if left untreated.






These diseases can have serious effects on vision and early detection is crucial for effective treatment.


| Normal Eye  | Cataract    | Retinopathy | Glaucoma    |
|-------------|-------------|-------------|-------------|
| ![Normal Eye](img/normal_eye.png) | ![Cataract](img/cataract.png) | ![Retinopathy](img/retinopathy.png) | ![Glaucoma](img/glaucoma.png) | 
[2]



## Install Guide
TODO

## Data Description & Structure Analysis

### Content
* Cataract 1038 Files​

* Diabetic Retinopathy 1098 Files​

* Glaukoma 1007 Files​

* Normal 1074 Files

### Structure

## Data Preprocessing
Um die Bilder für das Training des CNN vorzubereiten, werden verschiedene Preprocessing-Schritte durchgeführt:
1. **Resizing the images to uniform dimensions:**

*  The images are loaded first. Most of them are available with a resolution of 512 x 512 pixels. Those that are larger will be resized to 512 x 512. This ensures that no important information is lost and that the images are not distorted despite the reduction in resolution.

1. **Splitting data:**
* The data is split into training and validation data with a split of 80% training data (3374 images) and 20% validation data (843 images).

## Model Architecture
Das CNN-Modell besteht aus mehreren Schichten:
1. **Convolutional Layer:** 
* 6 layers with 2 convolutions, resulting in a total of 12 convolutions
* This increases the number of channels after each convolution layer.
* The increase in the number of channels corresponds to x2: 3 (RGB) => 16 => 32 => 64 => 128 => 256 => 512

2. **Pooling Layer:**
* Max pooling is carried out after each convolution layer. This means that deeper layers have a larger receptive field and therefore contain more of the original image.
* After the first two convolution layers, max pooling is performed with kernel size 4. This reduces the size of the image so that the training phase does not take too long and the images fit into memory.
* After the other layers, max pooling is performed with kernel size 2 so that not too much information is lost.

3. **Fully Connected Layer:**
* Finally, the input is passed to a Fully Connected Neural Network with 8192 Input Neurons => 4064 Hidden => 256 Hidden => 4 Output

## Model Training
* NUMBER_OF_EPOCHS = 20
* BATCH_SIZE = 64
* LEARNING_RATE = 1e-5

## Model Performance
* The performance of the model is evaluated with Multiclass Accuracy (from Torchmetrics), as the four classes are sufficiently balanced.
* Finally, the model is saved on Weights & Biases so that it can be used for predictions of unlabeled images.

## Results
TODO

## Discussion
TODO



## Sources
[1] https://scitechdaily.com/unlocking-the-future-of-health-predicting-disease-with-retinal-imaging-and-genetics/

[2] https://atlanticeyeinstitute.com/diabetic-eye-issues-5-ways-diabetes-impacts-vision/

