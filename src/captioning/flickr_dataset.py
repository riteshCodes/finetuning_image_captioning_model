import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
from PIL import Image
from typing import List, Tuple
import os


class FlickrDataset(Dataset):
    """Flickr30k dataset for image captioning

    Each image has multiple reference captions. __getitem__ returns a image and a random caption for that image.
    """

    def __init__(self, dataset, indices, processor):
        self.dataset = dataset
        self.indices = indices  # the actual indices of the images in the dataset
        self.processor = processor
        assert len(self.indices) > 0, "No indices provided"

    def __repr__(self) -> str:
        return f"FlickrDataset with {len(self)} items"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL image
            chosen_caption: Randomly chosen caption for the image. str
            actual_idx: Index of the image in the dataset
        """
        actual_idx = self.indices[idx]
        item = self.dataset[actual_idx]
        image = item["image"]
        captions = item["caption"]
        chosen_caption = random.choice(captions)
        # add bos and eos tokens
        bos_token = self.processor.tokenizer.bos_token
        eos_token = self.processor.tokenizer.eos_token
        chosen_caption = bos_token + chosen_caption + eos_token
        return image, chosen_caption, actual_idx


class BatchCollator:
    """Handles the collation of batches with padding."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, batch: List[Tuple[Image.Image, List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            images: Tensor of images (B, C, H, W)
            captions: Tensor of captions (B, L)
            attn_mask: Tensor of attention mask (B, L)
            indices: Tensor of indices (B)
        """
        images, captions, indices = zip(*batch)
        tensor_batch = self.processor(
            images=images, text=captions, return_tensors="pt", padding=True
        )
        images = tensor_batch["pixel_values"]
        captions = tensor_batch["input_ids"]
        attn_mask = tensor_batch["attention_mask"]
        indices = torch.tensor(indices)
        return images, captions, attn_mask, indices


def create_datasets(config):
    dataset = load_dataset("nlphuji/flickr30k")["test"]

    train_indices = []
    val_indices = []

    splits = dataset["split"]
    img_ids = dataset["img_id"]

    for split, img_id in zip(splits, img_ids):
        if split == "train" or split == "test":
            # If the split is train or test, we add it to the train set
            train_indices.append(int(img_id))
        elif split == "val":
            val_indices.append(int(img_id))
        else:
            raise ValueError(f"Unknown split {split}")

    train_dataset = FlickrDataset(dataset, train_indices, config.processor)
    val_dataset = FlickrDataset(dataset, val_indices, config.processor)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    return train_dataset, val_dataset


def create_dataloaders(config):
    train_dataset, val_dataset = create_datasets(config)

    collator = BatchCollator(config.processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=max(1, os.cpu_count() // 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=max(1, os.cpu_count() // 2),
    )

    return train_loader, val_loader


def get_pil_image(dataset, indices):
    """Get PIL image for the given indices."""
    assert isinstance(dataset, FlickrDataset), "Dataset must be FlickrDataset"
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    return [dataset.dataset[i]["image"] for i in indices]


def get_ref_captions(dataset, indices):
    """Get reference captions for the given indices."""
    assert isinstance(dataset, FlickrDataset), "Dataset must be FlickrDataset"
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    return [dataset.dataset[i]["caption"] for i in indices]


def sample_random_batch(loader, n_samples, processor):
    """Sample n_samples random items from dataset for visualization.

    Args:
        loader: DataLoader
        n_samples: int
        img_process_fn: callable

    Returns:
        images_pil: List of PIL images
        images_tensor: Tensor of images (B, C, H, W)
        ref_captions: List of reference captions. List[List[str]]
    """
    dataset = loader.dataset
    assert isinstance(dataset, FlickrDataset), "Dataset must be FlickrDataset"

    indices = random.sample(range(len(dataset)), n_samples)
    images_pil = [dataset[i][0] for i in indices]
    images_tensor = processor(images=images_pil, return_tensors="pt")["pixel_values"]

    data_indices = [dataset[i][2] for i in indices]
    ref_captions = get_ref_captions(dataset, data_indices)
    return images_pil, images_tensor, ref_captions
