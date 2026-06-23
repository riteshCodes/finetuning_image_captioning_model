import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM


class MaskMixin:
    """Handles generation of attention masks for huggingface (especially LLaMA) models that accept image and text embeddings.

    Some notes:
    - Attention mask produced by HF tokenizer: torch.int64, {0, 1}
    - Attention mask accepted by PyTorch sdpa attention: float or bool
    - But attention mask used by HF LLaMA models: float, {-inf, 0} since it's additive
    """

    ATTEND = 0
    IGNORE = float("-inf")

    @staticmethod
    def check_mask(mask):
        assert mask.dtype == torch.float32, (
            f"Attention mask must be float32, got {mask.dtype}"
        )
        assert torch.all((mask == MaskMixin.ATTEND) | (mask == MaskMixin.IGNORE)), (
            f"Attention mask must be {MaskMixin.ATTEND} or {MaskMixin.IGNORE}, got {mask.unique()}"
        )

    @staticmethod
    def image_text_attention_mask(batch_size, img_len, text_len, device):
        """Create an attention mask for [image, text] -> [image, text] attention.

        Args:
            batch_size: int, the number of samples in the batch
            img_len: int, the length of the image token sequence
            text_len: int, the maximum length of the text token sequence

        Returns:
            attn_mask: float tensor of shape (batch_size, 1, img_len+text_len, img_len+text_len)
                It looks like this (for img_len=2 and text_len=3):
                    [ 0,  0, -inf, -inf, -inf]
                    [ 0,  0, -inf, -inf, -inf]
                    [ 0,  0,    0, -inf, -inf]
                    [ 0,  0,    0,    0, -inf]
                    [ 0,  0,    0,    0,    0]
                where image can see all image but cannot see text, and text can see image,
                and current text can see all previous text.
        """
        img2text_mask = torch.full((img_len, text_len), MaskMixin.IGNORE)
        img2img_mask = torch.full((img_len, img_len), MaskMixin.ATTEND)
        text2img_mask = torch.full((text_len, img_len), MaskMixin.ATTEND)
        text2text_mask = torch.full((text_len, text_len), MaskMixin.IGNORE)
        text2text_mask = torch.triu(text2text_mask, diagonal=1)

        img2all = torch.cat([img2img_mask, img2text_mask], dim=1)
        text2all = torch.cat([text2img_mask, text2text_mask], dim=1)
        attn_mask = torch.cat([img2all, text2all], dim=0).float()

        attn_mask = attn_mask[None, None, :, :]
        attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
        attn_mask = attn_mask.to(device)
        return attn_mask

    @staticmethod
    def padding_mask(batch_size, img_len, text_padding_mask, device):
        """Extend the text padding mask to include the image tokens (which are always attended to).

        Args:
            batch_size: int, the number of samples in the batch
            img_len: int, the length of the image token sequence
            text_padding_mask: float tensor of shape (batch_size, text_seq_len)

        Returns:
            padding_mask: tensor of shape (batch_size, 1, 1, img_len+text_len)
        """
        assert text_padding_mask.device == device, "Device mismatch"
        MaskMixin.check_mask(text_padding_mask)

        image_padding_mask = torch.full((batch_size, 1, 1, img_len), MaskMixin.ATTEND)
        image_padding_mask = image_padding_mask.float().to(device)

        text_padding_mask = text_padding_mask[:, None, None, :]
        padding_mask = torch.cat([image_padding_mask, text_padding_mask], dim=-1)
        padding_mask = padding_mask
        return padding_mask

    @staticmethod
    def convert_to_float(mask):
        """Converts the attention mask to float and expected values for LLaMA models."""
        if mask.dtype == torch.int64:
            return_mask = mask.clone().float()
            return_mask[mask == 1] = MaskMixin.ATTEND
            return_mask[mask == 0] = MaskMixin.IGNORE
            MaskMixin.check_mask(return_mask)
            return return_mask
        elif mask.dtype == torch.bool:
            return_mask = mask.clone().float()
            return_mask[mask == 1] = MaskMixin.ATTEND
            return_mask[mask == 0] = MaskMixin.IGNORE
            MaskMixin.check_mask(return_mask)
            return return_mask
        else:
            raise ValueError(f"Unknown mask dtype: {mask.dtype}")

    @staticmethod
    def combine_masks(mask1, mask2):
        """Encode the logical AND operation between two masks."""
        MaskMixin.check_mask(mask1)
        MaskMixin.check_mask(mask2)
        return mask1 + mask2


class ImageCaptioningModel(nn.Module, MaskMixin):
    """Image captioning model that use a language model to process both image embeddings and text tokens."""

    def __init__(self, config):
        super().__init__()
        self.train_encoder = config.train_encoder
        self.processor = config.processor

        self.image_encoder = AutoModel.from_pretrained(
            config.image_encoder_checkpoint,
            trust_remote_code=True,
        )

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_checkpoint,
            trust_remote_code=True,
        )

        # Extract the text embedding layer from the language model
        self.text_embedding = self.decoder.get_input_embeddings()

        # Project the image embeddings to the same size as the text embeddings
        img_hidden_size = config.encoder_hidden_size
        text_hidden_size = config.decoder_hidden_size
        self.image_out_proj = nn.Linear(img_hidden_size, text_hidden_size)
        print("Image encoder hidden size:", img_hidden_size)
        print("Decoder hidden size:", text_hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_images(self, images=None, image_features=None):
        """Encode images or process pre-computed features.

        Args:
            images: Optional float tensor of shape (batch_size, 3, height, width)
            image_features: Optional float tensor of shape (batch_size, seq_len, hidden_size). Must be the output from the same image encoder architecture

        Returns:
            image_embeds: float tensor of shape (batch_size, seq_len, hidden_size)
        """
        if images is not None and image_features is not None:
            raise ValueError("Only one of images or image_features should be provided")

        if images is None and image_features is None:
            raise ValueError("Either images or image_features must be provided")

        if image_features is not None:
            return image_features

        if images is not None:
            image_features = self.image_encoder(pixel_values=images).last_hidden_state

            if image_features.dim() == 4:
                # this means (B, C, H, W). We need to flatten it to (B, H*W, C)
                B, C, H, W = image_features.shape
                image_features = image_features.permute(0, 2, 3, 1)
                image_features = image_features.reshape(B, H * W, C)
                image_features = image_features.contiguous()

            img_embeds = self.image_out_proj(image_features)

            if not self.train_encoder:
                img_embeds = img_embeds.detach()

            return img_embeds

    def forward(self, texts, text_padding_mask, images=None, image_features=None):
        """
        Args:
            texts: int tensor of shape (batch_size, seq_len)
            text_padding_mask: tensor of shape (batch_size, seq_len) generted by the tokenizer
            images: flaot tensor of shape (batch_size, 3, 224, 224)
            image_features: float tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            logits: float tensor of shape (batch_size, seq_len, vocab_size)
        """
        # encode images
        image_embeds = self.encode_images(images, image_features)

        # encode texts
        text_embeds = self.text_embedding(texts)

        # concatenate image and text embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # create attention mask
        B = image_embeds.shape[0]
        img_len = image_embeds.shape[1]
        text_len = text_embeds.shape[1]
        attn_mask = self.image_text_attention_mask(B, img_len, text_len, self.device)
        # logical and
        text_padding_mask = self.convert_to_float(text_padding_mask)
        padding_mask = self.padding_mask(B, img_len, text_padding_mask, self.device)
        attn_mask = self.combine_masks(attn_mask, padding_mask)

        # forward pass through the decoder
        # self.check_mask(attn_mask)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        return outputs.logits[:, img_len:, :].contiguous()

    @torch.no_grad()
    def generate(self, images, max_length=50):
        """Given a batch of images, generate captions using greedy decoding.

        Args:
            images: list of PIL images
            max_length: int, the maximum length of the generated sequence

        Returns:
            captions: list of predicted captions strings
            full_text: list of predicted captions strings with special tokens
        """
        B = len(images)
        # Convert images and start tokens to tensors
        start_token = [self.processor.tokenizer.bos_token for _ in range(B)]
        inputs = self.processor(images=images, text=start_token, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(self.device)
        generated_ids = inputs["input_ids"].to(self.device)
        text_padding_mask = inputs["attention_mask"].to(self.device)

        # encode images
        image_embeds = self.encode_images(images=pixel_values)

        # maintain a list of finished sequences
        is_finished = torch.zeros((B, 1), dtype=torch.bool, device=self.device)

        # generate captions
        for i in range(max_length):
            logits = self.forward(
                texts=generated_ids,
                text_padding_mask=text_padding_mask,
                image_features=image_embeds,
            )
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # update finished sequences
            is_finished |= next_token_id == self.processor.tokenizer.eos_token_id
            next_token_id[is_finished] = self.processor.tokenizer.pad_token_id

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            _next_mask = (~is_finished).to(torch.int64)
            text_padding_mask = torch.cat([text_padding_mask, _next_mask], dim=1)

            if is_finished.all():
                break

        # cut off the start token
        generated_ids = generated_ids[:, 1:]

        # decode the generated token ids
        captions = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return captions
