from PIL import Image
from clip_interrogator import Config, Interrogator
from glob import glob
import click
import pyexiv2
import requests
import sys

@click.command()
@click.argument('files', nargs=-1)
def clip(files):
  """Generate descriptions for the given images by using clip-interrogator.
  The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.

  FILES are the filenames of the images to generate the descriptions for. They can include wildcards / glob patterns.
  """
  ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

  for argument in files:
    for filename in glob(argument, recursive=True):
      image = Image.open(filename, mode='r')
      tags = ci.interrogate_fast(image)
      print("%s: %s" % (filename, tags))

if __name__ == '__main__':
  clip()

'''
WIP: Save tags as metadata on the image

    with open(filename, 'rb') as file:
      with pyexiv2.ImageData(file.read()) as image:
        key = 'Xmp.xmp.ClipInterrogatorDescription'

        image.modify_xmp({ key: tags })

        with open('./images/result.jpg', 'xb') as result:
          result.write(image.get_bytes())
'''
