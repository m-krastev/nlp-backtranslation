from pathlib import Path

import evaluate
import torch
from pytorch_lightning import LightningModule
from transformers import FSMTForConditionalGeneration
import bitsandbytes


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
        input_ids = batch.input_ids
        labels = batch.labels
        labels[labels == -100] = self.tokenizer.pad_token_id

        outputs = self.model.generate(input_ids)
        hyps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        srcs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        bleu = self.sacrebleu.compute(predictions=hyps, references=refs)

        self.hyp_handle.write("\n".join(hyps) + "\n")
        self.ref_handle.write("\n".join(refs) + "\n")
        self.src_handle.write("\n".join(srcs) + "\n")
        self.hyp_handle.flush()
        self.ref_handle.flush()
        self.src_handle.flush()

        return bleu["score"]

    def on_predict_start(self) -> None:
        super().on_predict_start()
        if not self.test_folder.exists():
            self.test_folder.mkdir(exist_ok=True, parents=True)
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
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.input_ids
        labels = batch.labels
        labels[labels == -100] = self.tokenizer.pad_token_id

        outputs = self.model.generate(input_ids)
        hyps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = self.sacrebleu.compute(predictions=hyps, references=refs)

        self.log("val_bleu", bleu["score"], prog_bar=True)
        return bleu["score"]

    def configure_optimizers(self):
        lr, adam_beta, weight_decay = (
            self.hparams.lr,
            self.hparams.adam_beta,
            self.hparams.weight_decay,
        )
        return bitsandbytes.optim.Adam8bit(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=lr,
            betas=adam_beta,
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
