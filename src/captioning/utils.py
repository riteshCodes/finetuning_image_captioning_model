import textwrap

import matplotlib.pyplot as plt
import sacrebleu
import torch.nn.functional as F
from tqdm import tqdm

from captioning.flickr_dataset import get_pil_image, get_ref_captions


def visualize_captioning_results(images, ref_captions, pred_captions, wrap_width=40):
    """
    Visualize image captioning results with images, predicted captions, and numbered reference captions.
    Images are displayed on top, followed by the predicted caption, and then numbered reference captions.

    Args:
    - images: List of PIL images.
    - ref_captions: List of lists of reference captions.
    - pred_captions: List of predicted captions.
    - wrap_width: Maximum line width for text wrapping.
    """
    # Resize all images to 224x224
    images_resized = [img.resize((224, 224)) for img in images]

    # Number of results
    num_results = len(images_resized)

    # Create a figure with 1 row and `num_results` columns
    fig, axes = plt.subplots(1, num_results, figsize=(num_results * 3, 9))

    # If there's only one result, axes won't be iterable, so we make it a list
    if num_results == 1:
        axes = [axes]

    for i, (image, refs, pred) in enumerate(
        zip(images_resized, ref_captions, pred_captions)
    ):
        assert isinstance(refs, list), "References must be a list of strings"
        assert isinstance(refs[0], str), "References must be a list of strings"
        assert isinstance(pred, str), "Prediction must be a string"

        # Display the image
        axes[i].imshow(image)
        axes[i].axis("off")  # Hide the axes

        # Wrap the predicted and reference captions
        wrapped_pred = "\n".join(textwrap.wrap(f"Pred: {pred}", wrap_width))
        wrapped_refs = "References:\n" + "\n".join(
            [textwrap.fill(f"{j + 1}. {ref}", wrap_width) for j, ref in enumerate(refs)]
        )

        # Add text below the image
        caption_text = rf"{wrapped_pred}" + f"\n\n{wrapped_refs}"
        axes[i].text(
            0.5,
            -0.2,
            caption_text,
            fontsize=10,
            ha="center",
            va="top",
            transform=axes[i].transAxes,
        )

    # Adjust layout
    plt.tight_layout()
    return fig


def compute_bleu(model, val_loader):
    """Compute BLEU score for image captioning model."""
    hypotheses = []
    references = []

    total_batches = len(val_loader)
    progress_bar = tqdm(
        enumerate(val_loader),
        total=total_batches,
        desc="Computing BLEU score",
        unit="batch",
    )
    for i, batch in progress_bar:
        images, captions, attn_mask, indices = batch
        ref_captions = get_ref_captions(val_loader.dataset, indices)
        images = get_pil_image(val_loader.dataset, indices)
        pred_captions = model.generate(images)

        # Log translations
        hypotheses.extend(pred_captions)
        references.extend(ref_captions)

    # Compute BLEU score
    score = sacrebleu.corpus_bleu(hypotheses, list(zip(*references))).score
    return score


def compute_loss(cap_logits, cap_target, attn_mask_target):
    """Cross entropy loss with with padding mask.

    Args:
        cap_logits: float tensor of shape (B, T, vocab_size)
        cap_target: int tensor of shape (B, T)
        attn_mask_target: int tensor of shape (B, T)

    Returns:
        loss: float tensor of shape (1)
    """
    # Reshape logits and targets
    B, T, V = cap_logits.shape
    logits = cap_logits.reshape(-1, V)
    targets = cap_target.reshape(-1)

    # Create a mask that excludes padding
    # attn_mask_target is True for all valid positions including EOS
    valid_mask = (attn_mask_target == 1).reshape(-1)

    # Only consider non-padding positions in loss
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    loss = F.cross_entropy(valid_logits, valid_targets)

    return loss
