from PIL import Image
from clip_interrogator import Config, Interrogator
from glob import glob
import pyexiv2
import requests
import sys

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

for argument in sys.argv[1:]:
  for filename in glob(argument, recursive=True):
    image = Image.open(filename, mode='r')
    tags = ci.interrogate_fast(image)
    print("%s: %s" % (filename, tags))

'''
WIP: Save tags as metadata on the image

    with open(filename, 'rb') as file:
      with pyexiv2.ImageData(file.read()) as image:
        key = 'Xmp.xmp.ClipInterrogatorDescription'

        image.modify_xmp({ key: tags })

        with open('./images/result.jpg', 'xb') as result:
          result.write(image.get_bytes())
'''
