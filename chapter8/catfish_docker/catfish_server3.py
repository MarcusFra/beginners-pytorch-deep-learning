import os
import requests
import torch
from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from torchvision import transforms

from catfish_model import CatfishModel, CatfishClasses

from shutil import copyfileobj ###
from tempfile import NamedTemporaryFile ###
from urllib.request import urlopen ###

def load_model():
	m = CatfishModel
	if "CATFISH_MODEL_LOCATION" in os.environ:
		parameter_url = 'https://drive.google.com/u/0/uc?id=1nXrujwT1NitPyOHc67fJjuBJ5BrrzGYE&export=download' #  os.environ["CATFISH_MODEL_LOCATION"]
		with urlopen(parameter_url) as fsrc, NamedTemporaryFile() as fdst: ### url https://drive.google.com/file/d/1nXrujwT1NitPyOHc67fJjuBJ5BrrzGYE/view?usp=sharing
			copyfileobj(fsrc, fdst)
			m.load_state_dict(torch.load(fdst))
	return m

model = load_model()

img_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
])

#def create_app():
app = Flask(__name__)

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

    #return app
if  __name__ == '__main__':
	app.run(host='5000', port='5000')
# app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])