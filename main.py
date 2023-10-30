from PIL import Image
from clip_interrogator import Config, Interrogator
from pathlib import Path
import click
import pyexiv2
import requests
import sys

def modify_metadata(path, description, metadata_type, metadata_key):
  with path.open(mode='rb') as file:
    with pyexiv2.ImageData(file.read()) as image:
      match metadata_type:
        case 'exif':
          image.modify_exif({ metadata_key: description })
        case 'iptc':
          image.modify_iptc({ metadata_key: description })
        case 'xmp':
          image.modify_xmp({ metadata_key: description })
        case _:
          raise ValueError("Unknown metadata type")
      return image.get_bytes()

@click.command(context_settings={'show_default': True})
@click.option('--save-to-file',
              type=click.Choice(['none', 'existing', 'new'], case_sensitive=False),
              default='none',
              help='Save the description to the metadata of the image or just print it without saving')
@click.option('--metadata-type',
              type=click.Choice(['exif', 'iptc', 'xmp'], case_sensitive=False),
              default='xmp',
              help='Store the description to different sections of the metadata')
@click.option('--metadata-key', default='Xmp.xmp.ClipInterrogatorDescription', help='The key used to save the description to the metadata')
@click.option('--model', default='ViT-L-14/openai', help='The name of the CLIP model used to generate descriptions')
@click.argument('files', nargs=-1)
def clip(files, save_to_file, metadata_type, metadata_key, model):
  """Generate descriptions for the given images by using clip-interrogator.
  The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.

  FILES are the filenames of the images to generate the descriptions for. They can include wildcards / glob patterns.
  """
  ci = Interrogator(Config(clip_model_name=model, quiet=True))

  for argument in files:
    for path in Path.cwd().glob(argument):
      with Image.open(path) as image:
        description = ci.interrogate_fast(image)
      print("%s: %s" % (path.relative_to(Path.cwd()), description))

      if save_to_file == 'existing' or save_to_file == 'new':
        image_bytes = modify_metadata(path, description, metadata_type, metadata_key)
        target = path if save_to_file == 'existing' else path.with_stem(path.stem + '.clip')
        mode = 'wb' if save_to_file == 'existing' else 'xb'
        with target.open(mode) as output_file:
          output_file.write(image_bytes)

if __name__ == '__main__':
  clip()
