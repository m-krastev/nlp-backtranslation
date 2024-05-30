from pathlib import Path
from typing import Any, List, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import FSMTTokenizer, DataCollatorForSeq2Seq


class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=1024):
        # NOTE: Not optimal to store everything in memory, but it's fine for now
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        return self.tokenizer(
            src_text,
            text_target=tgt_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )


class TranslationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        src: str,
        tgt: str,
        tokenizer: FSMTTokenizer,
        model: Any = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = model
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding=True)

    def collate_fn(self, batch):
        collated = self.collator(batch)
        collated["labels"][collated["labels"] == self.tokenizer.pad_token_id] = -100
        return collated

    def setup(self, stage: str) -> None:
        src, tgt = load_dataset(self.data_dir / "train", self.src, self.tgt)
        self.train = TranslationDataset(src, tgt, self.tokenizer, self.max_length)

        src, tgt = load_dataset(self.data_dir / "dev", self.src, self.tgt)
        self.val = TranslationDataset(src, tgt, self.tokenizer, self.max_length)

        src, tgt = load_dataset(self.data_dir / "test", self.src, self.tgt)
        self.test = TranslationDataset(src, tgt, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


def load_dataset(path: Path, src: str, tgt: str) -> Tuple[List[str], List[str]]:
    # NOTE: This only takes in raw text files (e.g. train.en/de)
    src_path = path.with_suffix(f".{src}")
    tgt_path = path.with_suffix(f".{tgt}")
    with open(src_path) as f:
        src_texts = f.read().splitlines()
    with open(tgt_path) as f:
        tgt_texts = f.read().splitlines()

    return src_texts, tgt_texts
