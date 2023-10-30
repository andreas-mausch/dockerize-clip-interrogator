from PIL import Image
from clip_interrogator import Config, Interrogator
from pathlib import Path
import click
import pyexiv2
import requests
import sys

def modify_metadata(path, description, metadata_key):
  with path.open(mode='rb') as file:
    with pyexiv2.ImageData(file.read()) as image:
      image.modify_xmp({ metadata_key: description })
      return image.get_bytes()

@click.command()
@click.option('--save-to-file',
              type=click.Choice(['none', 'new'], case_sensitive=False),
              default='none')
@click.option('--metadata-key', default='Xmp.xmp.ClipInterrogatorDescription')
@click.argument('files', nargs=-1)
def clip(files, save_to_file, metadata_key):
  """Generate descriptions for the given images by using clip-interrogator.
  The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.

  FILES are the filenames of the images to generate the descriptions for. They can include wildcards / glob patterns.
  """
  ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", quiet=True))

  for argument in files:
    for path in Path.cwd().glob(argument):
      image = Image.open(path)
      description = ci.interrogate_fast(image)
      print("%s: %s" % (path, description))

      if save_to_file == 'new':
        image_bytes = modify_metadata(path, description, metadata_key)
        with path.with_stem(path.stem + '.clip').open(mode='xb') as result:
          result.write(image_bytes)

if __name__ == '__main__':
  clip()
