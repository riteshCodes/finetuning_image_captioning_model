import torch
from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    ProcessorMixin,
)


@dataclass
class ExperimentConfig:
    # Experiment Identification
    config_name: str  # This should be one of the model types in GPTConfig
    exp_name: str = "captioning"

    lr: float = 3e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 30000
    warmup_steps: int = 1000
    eval_every_n_steps: int = 1000

    image_encoder_checkpoint: str = "apple/aimv2-large-patch14-224"
    decoder_checkpoint: str = "HuggingFaceTB/SmolLM2-135M"
    crop_size: int = None  # If None, use the default crop size of the image encoder
    encoder_hidden_size: int = None
    decoder_hidden_size: int = None

    processor: AutoProcessor = None
    lora_r: int = 8
    lora_alpha: float = 16
    train_encoder: bool = False
    train_decoder: bool = True
    decoder_lora_modules: list = None
    image_encoder_lora_modules: list = None
    image_pooling_factor: int = 2  # image tokens are pooled by this factor
    image_pooling_type: str = "avg"  # or max # TODO: better handle this part

    compute_bleu: bool = False
    # If False, only compute BLEU at the beginning and end of training
    # Computing BLEU takes 5 minutes for a validation set

    visualization_samples: int = 3

    device: torch.device = None
    log_to_tensorboard: bool = True
    save_model: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

        # Initialize the image text processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.image_encoder_checkpoint,
            use_fast=False,  # aimv2 does not support fast tokenizer
        )
        if self.crop_size is not None:
            image_processor.crop_size["height"] = self.crop_size
            image_processor.crop_size["width"] = self.crop_size

        tokenizer = AutoTokenizer.from_pretrained(self.decoder_checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        self.processor = ImageTextProcessor(image_processor, tokenizer)

        if (
            self.image_encoder_lora_modules is None
            and self.decoder_lora_modules is None
        ):
            print("No LoRA modules provided, training full model!")

        # print something
        print(f"Image encoder: {self.image_encoder_checkpoint}")
        print(f"Decoder: {self.decoder_checkpoint}")


class ImageTextProcessor(ProcessorMixin):
    """Combines an image processor and a text tokenizer into a single processor"""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if images is not None:
            # Removing text related kwargs
            # (still works if they are not removed but will raise a warning)
            img_kwargs = kwargs.copy()
            if "padding" in kwargs:
                img_kwargs.pop("padding")
            if "truncation" in kwargs:
                img_kwargs.pop("truncation")
            image_features = self.image_processor(
                images, return_tensors=return_tensors, **img_kwargs
            )
        else:
            image_features = {}

        if text is not None:
            text_features = self.tokenizer(
                text, return_tensors=return_tensors, **kwargs
            )
        else:
            text_features = {}

        combined_features = {**image_features, **text_features}
        return combined_features
