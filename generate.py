from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from pytorch_lightning import Trainer, seed_everything
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from utils.data import TranslationDataModule
from utils.models import TranslationLightning
import argparse
from statistics import mean, stdev

torch.set_float32_matmul_precision("medium")


def main():
    parser = argparse.ArgumentParser(description="Use a model ONLY for generation. ")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/it-parallel",
        help="The directory containing the data for generation.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size to use."
    )
    parser.add_argument(
        "--max_length", type=int, default=64, help="The maximum length of the input."
    )

    parser.add_argument(
        "--r", type=int, default=8, help="The number of LoRA heads to use."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="The alpha value for LoRA."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.03, help="The dropout rate for LoRA."
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="The checkpoint to load from.",
    )

    parser.add_argument(
        "--srclang", type=str, default="de", help="The source language."
    )
    parser.add_argument(
        "--tgtlang", type=str, default="en", help="The target language."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The directory to store model generation outputs in.",
    )

    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    SRC = args.srclang
    TGT = args.tgtlang
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length

    data_dir = Path(args.data_dir)
    test_folder = Path(args.output_dir)

    # Load the model and tokenizer
    mname = f"facebook/wmt19-{SRC}-{TGT}"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

    # Load the LoRA model
    config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["v_proj", "q_proj"],
    )

    # Apply the LoRA model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load from checkpoint if specified
    if args.load_from_checkpoint:
        model_pl = TranslationLightning.load_from_checkpoint(
            args.load_from_checkpoint, model=model, tokenizer=tokenizer
        )
    else:
        model_pl = TranslationLightning(
            model,
            tokenizer,
            lr=args.lr,
            adam_beta=(0.9, 0.98),
            weight_decay=1e-4,
            test_folder=test_folder,
        )


    # Create the trainer
    trainer = Trainer(
        precision="16-mixed",
    )

    # Only predict if specified
    test_data = TranslationDataModule(
        data_dir,
        SRC,
        TGT,
        tokenizer,
        model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    model_pl.test_dir = Path(args.output_dir)
    results = trainer.predict(model_pl, test_data)
    print(f"BLEU score: {mean(results):.3f}±{stdev(results):.3f}")
    exit(0)

if __name__ == "__main__":
    main()
