import os
import torch
import flask as f
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, GPT2TokenizerFast, BlipProcessor, BlipForConditionalGeneration, Pix2StructForConditionalGeneration, Pix2StructProcessor

app = f.Flask(__name__, template_folder='templates', static_folder='static')
app.config['IMAGE_UPLOADS'] = r'static\media'


def predict_using_model1(image, device):
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 128
    num_beams = 10
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def predict_using_model2(image, device):
    model_name = "bipin/image-caption-generator"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 128
    num_beams = 10
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def predict_using_model3(image, device):
    model_name = "Abdou/vit-swin-base-224-gpt2-image-captioning"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    max_length = 128
    num_beams = 10
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def predict_using_model4(image, device):
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs)
    preds = processor.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def predict_using_model5(image, device):
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs)
    preds = processor.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def predict_using_model6(image, device):
    model_name = "google/pix2struct-textcaps-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = Pix2StructProcessor.from_pretrained(model_name)
    inputs = processor(images=[image], return_tensors="pt").to(device)
    output_ids = model.generate(**inputs)
    preds = processor.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if f.request.method == 'GET':
        return f.render_template('index.html')
    img = f.request.files['image']
    img_path = os.path.join(app.config['IMAGE_UPLOADS'], img.filename)
    img.save(img_path)
    img = Image.open(img_path).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred1 = predict_using_model1(img, device)
    pred2 = predict_using_model2(img, device)
    pred3 = predict_using_model3(img, device)
    pred4 = predict_using_model4(img, device)
    pred5 = predict_using_model5(img, device)
    pred6 = predict_using_model6(img, device)
    return f.render_template('index.html', img_path=img_path, pred1=pred1, pred2=pred2, pred3=pred3, pred4=pred4, pred5=pred5, pred6=pred6)


if __name__ == '__main__':
    app.run(debug=True)
