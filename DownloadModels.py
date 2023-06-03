from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, GPT2TokenizerFast, BlipProcessor, BlipForConditionalGeneration, Pix2StructForConditionalGeneration, Pix2StructProcessor

model_name = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = "bipin/image-caption-generator"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_name = "Abdou/vit-swin-base-224-gpt2-image-captioning"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
model_name = "google/pix2struct-textcaps-base"
model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
processor = Pix2StructProcessor.from_pretrained(model_name)
