from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

for file in os.listdir("./images"):
    prompt = "a photography of"
    image = Image.open("./images/" + file, mode='r')
    inputs = processor(image, prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print("%s: %s; text2 = %s" % ("./images/" + file, generated_text, generated_text2))
