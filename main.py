from PIL import Image
from clip_interrogator import Config, Interrogator
import requests
import os

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

for file in os.listdir("./images"):
  image = Image.open("./images/" + file, mode='r')
  tags = ci.interrogate_fast(image)
  print("%s: %s" % (file, tags))
