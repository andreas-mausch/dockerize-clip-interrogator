from PIL import Image
from clip_interrogator import Config, Interrogator
from glob import glob
import requests
import sys

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

print(sys.argv)

for argument in sys.argv[1:]:
  for filename in glob(argument, recursive=True):
    image = Image.open(filename, mode='r')
    tags = ci.interrogate_fast(image)
    print("%s: %s" % (filename, tags))
