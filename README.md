# 🐱 Cat Breed Classifier (12 Breeds, ResNet18 + Transfer Learning) 🐱

A deep-learning project for classifying **12 cat breeds** using **ResNet18** model trained on the **Oxford-IIT Pet Dataset**.  
The system includes **complete preprocessing**, **training scripts**, **evaluation plots**, **confusion matrix**, and a **Gradio web interface** for easy real-world testing.

---

## Features:

- Classification of **12 cat breeds**  
- Transfer Learning using **ResNet18 (ImageNet pretrained)**  
- Full training pipeline: preprocessing → augmentations → loaders → training → evaluation  
- Model achieves **92.4% test accuracy**  
- Visual evaluation: accuracy + loss graphs, confusion matrix  
- Gradio-based interface for interactive testing  
- Ready-to-run notebook + full source code  

---

## Dataset

- **Oxford-IIT Pet Dataset**
- Contains:  
  - 12 cat breeds  
  - ~200 images per breed  
  - Large variation in lighting, angles, and quality  
- Metadata converted into a compact file: `cats_labels.csv`

---

## Model Architecture

- Backbone: **ResNet18**
- Pretrained on **ImageNet**
- Replaced the final FC layer with **12-neuron classifier**
- Trained using:
  - Optimizer: **Adam**
  - Loss: **CrossEntropyLoss**
  - Augmentations: random crop, horizontal flip, normalization

---

## Training Results

### **Accuracy & Loss**
![accuracy](images/accuracy_curve_beige.png)
![loss](images/loss_curve_beige.png)

### **Confusion Matrix**
![confusion_matrix](images/confusion_matrix_bej.png)

---

## Example Predictions

Real test images passed through the Gradio interface.

| Maine Coon Prediction | British Shorthair Prediction |
|----------------------|------------------------------|
| ![mc](images/pred_maine_coon.png) | ![bs](images/pred_british.png) |

---

## How to Run

### **1. Install dependencies**
```bash
pip install -r requirements.txt
```
### **2. Train the model**
```bash
python train.py
```
### **3. Run the Gradio app**
```bash
python app.py
```
## Technologies Used
- Python, PyTorch
- ResNet18
- Scikit-learn
- Matlplotlib & Seaborn
- Gradio
- Pillow
- NumPy

## Future Improvements

- Add more cat breeds
- Predict cat age from image
- Add genetic disease risk predictions
- Automatic cat detection (face/full-body)

## References

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,“Cats and Dogs,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012. Oxford-IIT Pet Dataset. (https://www.robots.ox.ac.uk/~vgg/data/pets)

K. He, X. Zhang, S. Ren, J. Sun,“Deep Residual Learning for Image Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. ResNet. (https://arxiv.org/abs/1512.03385)
