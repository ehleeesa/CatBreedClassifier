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

## Dataset:

- **Oxford-IIT Pet Dataset**
- Contains:  
  - 12 cat breeds  
  - ~200 images per breed  
  - Large variation in lighting, angles, and quality  
- Metadata converted into a compact file: `cats_labels.csv`

---

## Model architecture:

- Backbone: **ResNet18**
- Pretrained on **ImageNet**
- Replaced the final FC layer with **12-neuron classifier**
- Trained using:
  - Optimizer: **Adam**
  - Loss: **CrossEntropyLoss**
  - Augmentations: random crop, horizontal flip, normalization

---

## Training results:

<h3><b>Accuracy & Loss</b></h3>

<div align="center">
    <img src="https://github.com/user-attachments/assets/0822ceb7-7e79-4ad2-90fa-877c48b5a98d"
         alt="accuracy" width="450">
    <img src="https://github.com/user-attachments/assets/408bf52e-3d6f-4bbf-9311-6a69b6c94d3c"
         alt="loss" width="450">
</div>


<h3><b>Confusion Matrix</b></h3>

<div align="center">
    <img src="https://github.com/user-attachments/assets/c33dd786-2bbd-40ae-8bd2-c2cd3dc106f7"
         alt="confusion matrix" width="600">
</div>


---

## Example predictions:

Real test images passed through the Gradio interface, as seen below:


<div align="center"> <table> <tr> <th>Maine Coon Prediction</th> <th>British Shorthair Prediction</th> </tr> <tr> <td> <img src="https://github.com/user-attachments/assets/6c51ea3d-a1d4-4579-96bb-13530eb0e723" width="350"/> </td> <td> <img src="https://github.com/user-attachments/assets/5eb0acb7-e76d-4261-96b1-dc399b1218f3" width="350"/> </td> </tr> </table> </div>


---

## How to run:

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

## Future improvements:

- Add more cat breeds
- Predict cat age from image
- Add genetic disease risk predictions
- Automatic cat detection (face/full-body)

## References:

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,“Cats and Dogs,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012. Oxford-IIT Pet Dataset. (https://www.robots.ox.ac.uk/~vgg/data/pets)

K. He, X. Zhang, S. Ren, J. Sun,“Deep Residual Learning for Image Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. ResNet. (https://arxiv.org/abs/1512.03385)
