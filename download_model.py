from clip_interrogator import Config, Interrogator
from transformers import BlipProcessor, BlipForConditionalGeneration

Interrogator(Config(clip_model_name="ViT-L-14/openai"))

BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
