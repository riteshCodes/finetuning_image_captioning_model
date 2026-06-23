import torch
import torch.optim as optim

from captioning.core.base_trainer import BaseTrainer, make_summary_writer
from captioning.flickr_dataset import create_dataloaders, sample_random_batch
from captioning.utils import compute_bleu, compute_loss, visualize_captioning_results


def create_optimizer(model, config):
    return optim.AdamW(model.parameters(), lr=config.lr)


def create_scheduler(optimizer, config):
    return optim.lr_scheduler.ChainedScheduler(
        [
            # Linear warmup
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # start from 1% of base lr
                end_factor=1.0,
                total_iters=config.warmup_steps,
            ),
            # Cosine decay
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps),
        ]
    )


class Trainer(BaseTrainer):
    def __init__(self, model, config):
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        train_loader, val_loader = create_dataloaders(config)

        if config.log_to_tensorboard:
            logger = make_summary_writer(config.exp_name, config.config_name)
        else:
            logger = None

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            max_steps=config.max_steps,
            eval_every_n_steps=config.eval_every_n_steps,
            logger=logger,
            scheduler=scheduler,
            config=config,
        )

    def training_step(self, batch):
        images, captions, attn_mask, _ = batch
        B, T = captions.shape
        cap_input = captions[:, :-1].contiguous()
        cap_target = captions[:, 1:].contiguous()
        attn_mask_input = attn_mask[:, :-1].contiguous()
        attn_mask_target = attn_mask[:, 1:].contiguous()

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            cap_logits = self.model(cap_input, attn_mask_input, images=images)
            loss = compute_loss(cap_logits, cap_target, attn_mask_target)
        return {"loss": loss}

    def validation_step(self, batch):
        return self.training_step(batch)

    def run_experiment(self):
        for step, _, _ in self.fit():
            # if step % (self.config.eval_every_n_steps * 1) == 0 or step == self.config.max_steps:
            if (
                self.config.compute_bleu
            ):  # or step == self.config.max_steps or step == 1:
                print(f"Computing BLEU score at step {step}")
                blue_score = compute_bleu(self.model, self.val_loader, self.config)
                self.logger.add_scalar(
                    f"{self.config.exp_name}/val_bleu", blue_score, step
                )

            # Visualize captioning results
            images_pil, images_tensor, ref_captions = sample_random_batch(
                self.val_loader, 3, self.config.processor
            )
            pred_captions = self.model.generate(images_pil)
            fig = visualize_captioning_results(images_pil, ref_captions, pred_captions)
            self.logger.add_figure("visualization", fig, step)
            self.logger.flush()

        if self.config.save_model:
            torch.save(self.model.state_dict(), "model.pth")
