import os
import torch
from io import BytesIO
from flask import Flask, jsonify
from PIL import Image
from torchvision import transforms

from catfish_model import CatfishModel, CatfishClasses

def load_model():
    m = CatfishModel
    location = os.environ["CATFISH_MODEL_LOCATION"]
    m.load_state_dict(torch.load(location))
    return m

model = load_model()

app = Flask(__name__)

img_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
])

@app.route("/")
def status():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_url = request.form.image_url
    else:
        img_url = request.args.get('image_url', '')

    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img_tensor = img_transforms(img).unsqueeze(0)
    prediction =  model(img_tensor)
    predicted_class = CatfishClasses[torch.argmax(prediction)]
    return jsonify({"image": img_url, "prediction": predicted_class})

if __name__ == '__main__':
    app.run(host=5000, port=5000)
    #app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])