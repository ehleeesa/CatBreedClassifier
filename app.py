import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as T

from PIL import Image
import pandas as pd
import gradio as gr


# ------------------------------
# Config & device
# ------------------------------

MODEL_PATH = "best_cat_breeds_resnet18.pth"   
CSV_PATH = "cats_labels.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Label mapping
# ------------------------------

df = pd.read_csv(CSV_PATH)

breed_names = {}
for cid in sorted(df["class_id"].unique()):
    row = df[df["class_id"] == cid].iloc[0]
    file_name = row["file"]
    breed_name = file_name.rsplit("_", 1)[0]  # ex: "Abyssinian_1" -> "Abyssinian"
    breed_names[cid] = breed_name

class_ids = sorted(df["class_id"].unique())
class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}
idx_to_class_id = {i: cid for cid, i in class_id_to_idx.items()}

label_to_breed = {
    class_id_to_idx[cid]: breed_names[cid]
    for cid in class_ids
}

num_classes = len(class_ids)


# ------------------------------
# Transformari
# ------------------------------

val_test_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------
# Model loading
# ------------------------------

def load_model():
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


# ------------------------------
# Functie de predictie (Gradio)
# ------------------------------

def predict_image(img: Image.Image):
    img = img.convert("RGB")
    x = val_test_transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, 3)  # top 3 rase
    result = {}

    for p, idx in zip(top_probs, top_idxs):
        breed = label_to_breed[idx.item()]
        result[breed] = float(p)

    return result


# ------------------------------
# Interfața Gradio
# ------------------------------

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Sistem de recunoaștere a rasei unei pisici",
    description="Încărcați o fotografie cu o pisică pentru prezicerea rasei acesteia (top 3 rase):"
)

if __name__ == "__main__":
    demo.launch()
