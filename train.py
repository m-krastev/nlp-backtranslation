from pathlib import Path

from statistics import mean, stdev
import torch
from peft import LoraConfig, get_peft_model
from pytorch_lightning import Trainer, seed_everything
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from utils.data import TranslationDataModule, load_combined_dataset
from utils.models import TranslationLightning
from utils.metric import apply_diversity_metric
import argparse

torch.set_float32_matmul_precision("medium")


def main():
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size to use."
    )
    parser.add_argument(
        "--max_length", type=int, default=64, help="The maximum length of the input."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="The number of epochs to train for."
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help = "Resume training from a given checkpoint.")
    parser.add_argument(
        "--only_predict", action="store_true", help="Only run predictions. Note: only test resource will be used."
    )
    parser.add_argument(
        "--srclang", "--src", type=str, default="de", help="The source language."
    )
    parser.add_argument(
        "--tgtlang","--tgt", type=str, default="en", help="The target language."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data",
        help="The directory containing the data for either training or prediction, or both.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The directory to store model generation outputs in.",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="The learning rate to use."
    )
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")

    parser.add_argument("--use_diversity_metric", action="store_true", help="Use the diversity metric for data selection.")

    parser.add_argument("--top_percentage", type=float, default=0.5, help="The top percentage of data to use based on the diversity metric.")

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    SRC = args.srclang
    TGT = args.tgtlang
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

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
            output_dir=output_dir,
        )

    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

    wandb_logger = WandbLogger(project="huggingface")
    tensorboard_logger = TensorBoardLogger(".")

    # Create the trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=0.3,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_val_batches=0.25,
        logger=[wandb_logger, tensorboard_logger],
        precision="16-mixed",
    )
    train_data = TranslationDataModule(
        data_dir,
        SRC,
        TGT,
        tokenizer,
        model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        use_combined_data=args.use_diversity_metric,
        generation_folder="generation+it-parallel-en-de",
        top_percentage=args.top_percentage
    )
    train_data.setup('fit')
    trainer.fit(model_pl, train_data)

    # Only predict if specified
    if args.only_predict:
        test_data = TranslationDataModule(
            data_dir,
            SRC,
            TGT,
            tokenizer,
            model,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        model_pl.output_dir = Path(args.output_dir)
        results = trainer.predict(model_pl, test_data)
        print(f"BLEU score: {mean(results):.2f}±{stdev(results):.1f}")
        exit(0)
    # Train the model
    train_data = TranslationDataModule(
        data_dir,
        SRC,
        TGT,
        tokenizer,
        model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    trainer.fit(model_pl, datamodule=train_data, ckpt_path=args.resume_from_checkpoint)
    results = trainer.predict(model_pl, datamodule=train_data)
    print(f"BLEU score: {mean(results):.2f}±{stdev(results):.1f}")


if __name__ == "__main__":
    main()
