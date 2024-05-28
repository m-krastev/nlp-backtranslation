# %% [markdown]
# # The Backtranslation Notebook to Rule Over Them All
# Please do ignore the title, I just wanted to make it sound cool. This notebook is a simple demonstration of how to use backtranslation to improve the performance of a machine learning model. The notebook is divided into the following sections:
# 1. Introduction and Simple Generation
# 2. Applying Data Preparation/Filtering
# 3. Investigating Iterative Backtranslation
# 4. Applying a similar pipeline to a more complex model (ALMA-R)
# 5. Conclusion

# %%
# First cell in the notebook to enable autoreload of modules
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from utils.data import TranslationDataModule
from utils.models import TranslationLightning
from pytorch_lightning import Trainer
from peft import get_peft_model, LoraConfig
# Isn't it so nice and clean now? I went through FOUR different ways of doing this before I thought of this one. WDWFDGFEQWDQWFGA
import torch
torch.set_float32_matmul_precision('medium')

SRC = "de"
TGT = "en"
BATCH_SIZE = 8
MAX_LENGTH = 512

cwd = Path.cwd()
data_dir = cwd / "Data"

it_parallel = "it-parallel"
news_dataset = "train-euro-news-big"
it_mono = "it-mono"

test_folder = cwd / "tests"

mname = f"facebook/wmt19-{SRC}-{TGT}"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

config = LoraConfig(
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.2,
    target_modules = ["v_proj", "q_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


# input_ids = tokenizer.encode( "Maschinelles Lernen ist großartig, oder?", return_tensors="pt")
# outputs = model.generate(input_ids)
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(decoded)  # Machine Learning is great, isn't it?

# Copying settings from the original file though probably unneeded..
# model.generation_config.length_penalty = 1.2
# model.generation_config.num_beams = 5

model_pl = TranslationLightning(
    model,
    tokenizer,
    lr=1e-4,
    adam_beta=(0.9, 0.98),
    weight_decay=1e-4,
    test_folder=test_folder,
)
trainer = Trainer(max_epochs=5, gradient_clip_val=0.1, precision="bf16-mixed")


# %% [markdown]
# ## Simple Generation
# Here we simply load a pre-trained model and run inference on the test set. The model loaded is a pre-trained model from the [Hugging Face Transformers](https://huggingface.co/transformers/) library.
#
# TODO: Evaluate the model on the NEWS set and report the BLEU score.
#
# | Model | it-parallel | NEWS |
# | --- | --- | --- |
# | Base | 0.0 | 0.0 |
# | FT on IT-parallel | 0.0 | 0.0 |

# %%
data = TranslationDataModule(
    data_dir / it_parallel,
    SRC,
    TGT,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
)

results = trainer.predict(model_pl, datamodule=data)
average_bleu = sum([r[2]["score"] for r in results]) / len(results)
print(f"Average BLEU: {average_bleu}")

exit
# %%
# Now we can train the model
model_pl.test_folder = model_pl.test_folder / ("ft-"+it_parallel)
trainer.fit(model_pl, datamodule=data)
# model_pl = TranslationLightning.load_from_checkpoint("/home/mkrastev/nlp2/lightning_logs/version_6413306/checkpoints/epoch=0-step=2500.ckpt", model=model, tokenizer=tokenizer)
results = trainer.predict(model_pl, datamodule=data)
average_bleu = sum([r[2]["score"] for r in results]) / len(results)
print(f"Average BLEU: {average_bleu}")