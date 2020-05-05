import os
import requests
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from torchvision import transforms

from catfish_model import load_catfish_model, CatfishClasses # from . import CatfishModel


# curl http://127.0.0.1:8080/predict?image_url=https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/A_domestic_shorthair_tortie-tabby_cat.jpg/412px-A_domestic_shorthair_tortie-tabby_cat.jpg

model = load_catfish_model()

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

  response = requests.get(img_url) # maybe also possible with flask.request
  img = Image.open(BytesIO(response.content)) # instead of open_image()
  img_tensor = img_transforms(img).unsqueeze(0)
  prediction =  model(img_tensor) # F.softmax(model(img_tensor))
  predicted_class = CatfishClasses[torch.argmax(prediction)] # different to chap 2: prediction.argmax()
  return jsonify({"image": img_url, "prediction": predicted_class})

if __name__ == '__main__':
  app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])