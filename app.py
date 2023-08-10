import asyncio
import os
import tempfile
from aiohttp.web import Request
from aiohttp import web
import requests

import torch 
from PIL import Image

from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device='cpu'



model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)


async def predict(image : Image):
    img = image.convert('RGB')
    model.eval()
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]
    

async def download_image(url : str) -> dict:
    try:
        print(url)
        image_file = tempfile.NamedTemporaryFile(prefix="ad_image_")
        image_file.write(requests.get(url).content)
        result = await predict(image = Image.open(image_file.name))
        print(result)
        return {
            "url" : url,
            "caption" : result
        }
    except:
        return {
            "url" : url,
            "caption" : None
        }


async def handle(request : Request):
    name = request.match_info.get('name', "Anonymous")
    return web.Response(text="Hello, " + name)

async def handle_image_tags(request : Request):

    images = await request.json()

    image_futures = [download_image(url) for url in images]
    results = await asyncio.gather(*image_futures)    
    return web.json_response(data = results)


app = web.Application()
app.add_routes(
    [
        web.get('/', handle),
        web.get('/{name}', handle),
        web.post('/get_tags', handle_image_tags)
    ]
)


if __name__ == '__main__':
    web.run_app(app)