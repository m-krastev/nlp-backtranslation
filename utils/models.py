from pathlib import Path

import evaluate
import torch
from pytorch_lightning import LightningModule
from transformers import FSMTForConditionalGeneration


class TranslationLightning(LightningModule):
    def __init__(
        self,
        model: FSMTForConditionalGeneration,
        tokenizer,
        lr=1e-4,
        adam_beta=(0.9, 0.98),
        weight_decay=1e-4,
        test_folder="test",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer", "test_folder"])
        self.model = model
        self.tokenizer = tokenizer

        self.sacrebleu = evaluate.load("sacrebleu")
        self.test_folder = Path(test_folder)

    def forward(self, batch):
        input_ids = batch.input_ids.squeeze(1)
        labels = batch.labels.squeeze(1)

        outputs = self.model.generate(input_ids)
        hyps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        srcs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        bleu = self.sacrebleu.compute(predictions=hyps, references=refs)

        self.hyp_handle.write("\n".join(hyps) + "\n")
        self.ref_handle.write("\n".join(refs) + "\n")
        self.src_handle.write("\n".join(srcs) + "\n")

        return hyps, refs, bleu

    def on_predict_start(self) -> None:
        super().on_predict_start()
        self.hypothesis_file = self.test_folder / "hypothesis.hyp"
        self.reference_file = self.test_folder / "reference.ref"
        self.source_file = self.test_folder / "source.src"

        self.hyp_handle = open(self.hypothesis_file, "w")
        self.ref_handle = open(self.reference_file, "w")
        self.src_handle = open(self.source_file, "w")

    def on_predict_end(self) -> None:
        super().on_predict_end()
        self.hyp_handle.close()
        self.ref_handle.close()
        self.src_handle.close()

    def training_step(self, batch, batch_idx):
        # There are some shenanigans with the batch, so we need to extract the input_ids, attention_mask, and labels
        input_ids = batch.input_ids.squeeze(1)
        attention_mask = batch.attention_mask.squeeze(1)
        labels = batch.labels.squeeze(1)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.input_ids.squeeze(1)
        labels = batch.labels.squeeze(1)

        outputs = self.model.generate(input_ids)
        hyps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = self.sacrebleu.compute(predictions=hyps, references=refs)

        self.log("val_bleu", bleu["score"], prog_bar=True)
        return bleu

    def configure_optimizers(self):
        lr, adam_beta, weight_decay = (
            self.hparams.lr,
            self.hparams.adam_beta,
            self.hparams.weight_decay,
        )
        return torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=adam_beta, weight_decay=weight_decay
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
