import torch
from catfish_model import CatfishModel, CatfishClasses
import os
from shutil import copyfileobj ###
from tempfile import NamedTemporaryFile ###
from urllib.request import urlopen ###

m = CatfishModel
if "CATFISH_MODEL_LOCATION" in os.environ:
	parameter_url = 'https://drive.google.com/u/0/uc?id=1nXrujwT1NitPyOHc67fJjuBJ5BrrzGYE&export=download'  # os.environ["CATFISH_MODEL_LOCATION"]
	with urlopen(
			parameter_url) as fsrc, NamedTemporaryFile() as fdst:  ### url https://drive.google.com/file/d/1nXrujwT1NitPyOHc67fJjuBJ5BrrzGYE/view?usp=sharing
		copyfileobj(fsrc, fdst)
		m.load_state_dict(torch.load(fdst))

print(m)