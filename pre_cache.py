from transformers import BertTokenizerFast  # TODO: add to reuquirements
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
t5_version="google/t5-v1_1-large"
T5Tokenizer.from_pretrained(t5_version)
T5EncoderModel.from_pretrained(t5_version)

clip_version="openai/clip-vit-large-patch14"
CLIPTokenizer.from_pretrained(clip_version)
CLIPTextModel.from_pretrained(clip_version)

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
