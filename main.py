from PIL import Image
from clip_interrogator import Config, Interrogator
from pathlib import Path
import click
import pyexiv2
import requests
import sys

@click.command()
@click.option('--save-to-file',
              type=click.Choice(['none', 'new'], case_sensitive=False),
              default='none')
@click.argument('files', nargs=-1)
def clip(files, save_to_file):
  """Generate descriptions for the given images by using clip-interrogator.
  The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.

  FILES are the filenames of the images to generate the descriptions for. They can include wildcards / glob patterns.
  """
  ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

  for argument in files:
    for path in Path.cwd().glob(argument):
      image = Image.open(path)
      tags = ci.interrogate_fast(image)
      print("%s: %s" % (path, tags))

      if save_to_file == 'new':
        with path.open(mode='rb') as file:
          with pyexiv2.ImageData(file.read()) as image:
            key = 'Xmp.xmp.ClipInterrogatorDescription'

            image.modify_xmp({ key: tags })

            with open('./images/result.jpg', 'xb') as result:
              result.write(image.get_bytes())

if __name__ == '__main__':
  clip()
