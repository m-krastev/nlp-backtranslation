from pathlib import Path
from typing import Any, List, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import FSMTTokenizer, DataCollatorForSeq2Seq
from utils.metric import apply_diversity_metric


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
        use_combined_data: bool = False,
        generation_folder: str = None,
        top_percentage: float = 0.5
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
        self.use_combined_data = use_combined_data
        self.generation_folder = generation_folder
        self.top_percentage = top_percentage

    def collate_fn(self, batch):
        collated = self.collator(batch)
        collated["labels"][collated["labels"] == self.tokenizer.pad_token_id] = -100
        return collated

    def setup(self, stage: str) -> None:
        if self.use_combined_data and self.generation_folder is not None:
            src_tgt_pairs = load_combined_dataset(self.data_dir, self.generation_folder, self.src, self.tgt)
            selected_pairs = apply_diversity_metric(src_tgt_pairs, top_percentage=self.top_percentage)
            src, tgt = zip(*selected_pairs)
            self.train = TranslationDataset(src, tgt, self.tokenizer, self.max_length)
            dev_dir = self.data_dir / self.generation_folder / "dev"
            test_dir = self.data_dir / self.generation_folder / "test"

        else: 
            train_dir = self.data_dir / "train"
            dev_dir = self.data_dir / "dev"
            test_dir = self.data_dir / "test"
            src, tgt = load_dataset(train_dir, self.src, self.tgt)
            self.train = TranslationDataset(src, tgt, self.tokenizer, self.max_length)

        src, tgt = load_dataset(dev_dir, self.src, self.tgt)
        self.val = TranslationDataset(src, tgt, self.tokenizer, self.max_length)

        src, tgt = load_dataset(test_dir, self.src, self.tgt)
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
    with open(src_path, encoding='utf-8') as f:
        src_texts = f.read().splitlines()
    with open(tgt_path, encoding='utf-8') as f:
        tgt_texts = f.read().splitlines()

    return src_texts, tgt_texts

def load_combined_dataset(data_dir: Path, generation_folder: str, src: str, tgt: str) -> Tuple[List[str], List[str]]:
    combined_data_dir = data_dir / generation_folder

    src_texts, tgt_texts = [], []
    
    for split in ['train', 'dev', 'test']:
        src_split, tgt_split = load_dataset(combined_data_dir / split, src, tgt)
        src_texts.extend(src_split)
        tgt_texts.extend(tgt_split)
    
    return list(zip(src_texts, tgt_texts))
