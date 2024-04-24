# Retinal Imaging
## Medical Image Classification
### Data Science in Health
#### Gérome Laurin Meyer | Rebekka von Wartburg
#### May 24, 2024
![Titelbild](img/title_img.jpeg) [1]

---

## Introduction
In this project, a Convolutional Neural Network (CNN) is being developed with the aim of recognizing and classifying three important eye diseases: Cataract, Retinopathy and Glaucoma. These diseases can have serious effects on vision and early detection is crucial for effective treatment.


| Normal Eye | Cataract | Retinopathy | Glaucoma |
|------------|----------|-------------|----------|
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
1. Größenanpassung der Bilder auf einheitliche Dimensionen.
2. Normalisierung der Pixelwerte.
3. Data Augmentation, um die Robustheit des Modells zu verbessern.

*  D'Bilder werded glade und uf 512x512 pixel resized (die meiste sind scho uf dere Uflösig, aber es hett e paar wo no grösser sind). Ich duen sie uf die 512x512 resize, well denn bi de meiste Bilder kei Informatione verlore göhnd, ohni eppis z'verzerre.
* Nachher werded d'Bilder in Training und Validation Split (80% => 3374 Train, 20% => 843 Validation) uufteilt

## Model Architecture
Das CNN-Modell besteht aus mehreren Schichten:
1. **Convolutional Layer:** 
2. **Pooling Layer:**
3. **Fully Connected Layer:**

* Für d'Predictions han ich jetzt 6 layers à 2 Convolutions gno = 12 total convolutions 
Nach jedem Layer wird d'Azahl Channels vergrösseret und es Max-Pooling gmacht, d.h. die tüüfere Layers hend es grösseres receptive field (= sie gshend meh vom Originale Bild) 
D'Azahl Channels isch fascht immer x2: 3 (RGB) => 16 => 32 => 64 => 128 => 256 => 512
Mir mached nach de erschte zwei Layer es MaxPooling mit Kernel 4 (das macht s'Bild schnell chliiner, süsch würdeds nöd in Memory passe / langsam trainiere). Nachher sind d'MaxPooling nur no mit Kernel 2 damit nöd zvill Information verlore gaht
* Am Schluss hetts no es Fully Connected Neural Network mit 8192 Input Neurone => 4064 Hidden => 256 Hidden => 4 Output

## Model Training

## Model Performance
Die Leistung des Modells wird anhand der Accuracy und weiterer Metriken wie Precision, Recall und F1-Score evaluiert. Diese Metriken helfen uns zu verstehen, wie gut unser Modell die verschiedenen Krankheiten identifizieren kann.

5. Evaluiere düend mir mit Multiclass Accuracy (vo Torchmetrics), well d'Klasse relativ balanced sind (ich han au de F1-Score no agluegt, aber er isch eigentlich immer identisch zu de Accuracy) 
6. Schlussendlich speicherts s'Modell uf W&B damit mer es chönnted abelade & wiederverwende (z.B. zum neue Predictions z'mache wo mer no kei Labels hett)

## Results

## Discussion
TODO



## Sources
[1] https://scitechdaily.com/unlocking-the-future-of-health-predicting-disease-with-retinal-imaging-and-genetics/

[2] https://atlanticeyeinstitute.com/diabetic-eye-issues-5-ways-diabetes-impacts-vision/

