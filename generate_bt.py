# %% [markdown]
# # The Backtranslation Notebook to Rule Over Them All
# Please do ignore the title, I just wanted to make it sound cool. This notebook is a simple demonstration of how to use backtranslation to improve the performance of a machine learning model. The notebook is divided into the following sections:
# 1. Introduction and Simple Generation
# 2. Applying Data Preparation/Filtering
# 3. Investigating Iterative Backtranslation
# 4. Applying a similar pipeline to a more complex model (ALMA-R)
# 5. Conclusion


# %%
from pathlib import Path
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from utils.data import TranslationDataModule
from utils.models import TranslationLightning
from pytorch_lightning import Trainer
from peft import LoraConfig, get_peft_model
import torch
torch.set_float32_matmul_precision('medium')
# Isn't it so nice and clean now? I went through FOUR different ways of doing this before I thought of this one. WDWFDGFEQWDQWFGA

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
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.03,
    target_modules = ["v_proj", "q_proj"]
)

it_parallel_data = TranslationDataModule(data_dir / it_parallel, SRC, TGT, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
# Add news_dataset


model = get_peft_model(model, config)
print(model.print_trainable_parameters())

# input_ids = tokenizer.encode( "Maschinelles Lernen ist großartig, oder?", return_tensors="pt")
# outputs = model.generate(input_ids)
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(decoded)  # Machine Learning is great, isn't it?

# Copying settings from the original file though probably unneeded..
# model.generation_config.length_penalty = 1.2
# model.generation_config.num_beams = 5


model_pl = TranslationLightning(model, tokenizer, lr=3e-4, adam_beta=(0.9, 0.98), weight_decay=1e-4, test_folder = test_folder)

trainer = Trainer(max_epochs=10,  gradient_clip_val=0.1, val_check_interval = 0.25, limit_val_batches=0.25, precision='16-mixed')

# %% [markdown]
# ## Simple Generation
# Here we simply load a pre-trained model and run inference on the test set. The model loaded is a pre-trained model from the [Hugging Face Transformers](https://huggingface.co/transformers/) library. 
# 
# TODO: Evaluate the model on the NEWS set and report the BLEU score.
# 
# | Model | it-parallel | NEWS |
# | --- | --- | --- |
# | Base | 38.307622648987454 | 0.0 |
# | FT on IT-parallel | 29.919114415962934 | 0.0 |
# | FT on BT | 29.919114415962934 | 0.0 |
# | FT on IT-parallel and BT | 29.919114415962934 | 0.0 |

# %%
# IT-parallel
# results = trainer.predict(model_pl, datamodule=it_parallel_data)
# bleu = sum(r[2]["score"] for r in results) / len(results)
# print("Average BLEU:", bleu)

# News corpus
# ...

# %%
# Now we can train the model
# trainer.fit(model_pl, datamodule=it_parallel_data)

# # IT-parallel
# results = trainer.predict(model_pl, datamodule=it_parallel_data)
# bleu = sum(r[2]["score"] for r in results) / len(results)
# print("Average BLEU:", bleu)

# News corpus

# %% [markdown]
# ### Now the key ingredient: Backtranslation
# Backtranslation uses a reverse model to generate synthetic data. This synthetic data is then used to train the model. The idea is that the synthetic data will help the model generalize better.

# %%
# reverse_model = FSMTForConditionalGeneration.from_pretrained(f"facebook/wmt19-{TGT}-{SRC}")
# reverse_tokenizer = FSMTTokenizer.from_pretrained(f"facebook/wmt19-{TGT}-{SRC}")

# it_mono_data = TranslationDataModule(data_dir / it_mono, TGT, SRC, reverse_tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

output_folder = test_folder / f"generation-{TGT}-{SRC}"
# reverse_lightning = TranslationLightning(reverse_model, reverse_tokenizer, lr=3e-4, adam_beta=(0.9, 0.98), weight_decay=1e-4, test_folder = output_folder)

# it_mono_data.setup('fit')
# trainer.predict(reverse_lightning, it_mono_data.train_dataloader())
# %% [markdown]
# Now that we have generated the data, we can now copy in the generated data.

# %%
import shutil
bt_dir = data_dir / output_folder.name
if not bt_dir.exists():
    bt_dir.mkdir(exist_ok=True, parents=True)

shutil.copy(output_folder / "hypothesis.hyp", bt_dir / f"train.{SRC}")
shutil.copy(output_folder / "source.src", bt_dir / f"train.{TGT}")
shutil.copy(data_dir / it_mono / f"dev.{SRC}", bt_dir / f"dev.{SRC}")
shutil.copy(data_dir / it_mono / f"dev.{TGT}", bt_dir / f"dev.{TGT}")
shutil.copy(data_dir / it_mono / f"test.{SRC}", bt_dir / f"test.{SRC}")
shutil.copy(data_dir / it_mono / f"test.{TGT}", bt_dir / f"test.{TGT}")


# %%
# And now we can finally train...
it_mono_bt = TranslationDataModule(bt_dir, SRC, TGT, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
trainer.fit(model_pl, datamodule=it_mono_bt)

# # %%
model_pl.test_folder = test_folder / f"bt_{it_mono}"
# # IT-parallel
results = trainer.predict(model_pl, datamodule=it_parallel_data)
# print("IT-parallel", sum(r[2]["score"] for r in results) / len(results))
# # News corpus
# # ...

# # %% [markdown]
# # And now finally we can concatenate both the original and the generated data and train the model on it.

# # %%
# parallel_plus_bt = data_dir / f"parallel+bt_generated_{it_mono}"
# shutil.copytree(bt_dir, parallel_plus_bt)
# train_file = parallel_plus_bt /f"train.{SRC}"
# with open(train_file, "r+") as f:
#     with open(data_dir / it_parallel / f"train.{SRC}", "r") as f2:
#         f.write("\n" + f2.read())
        
# train_file_target = parallel_plus_bt / f"train.{TGT}"
# with open(train_file_target, "r+") as f:
#     with open(data_dir / it_parallel / f"train.{TGT}", "r") as f2:
#         f.write("\n" + f2.read())


# # %%
# # Fitting the model on the combined data
# parallel_plus_bt_data = TranslationDataModule(parallel_plus_bt, SRC, TGT, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
# trainer.fit(model_pl, datamodule=parallel_plus_bt_data)

# # %%
# # IT-parallel
# results = trainer.predict(model_pl, datamodule=it_parallel_data)
# print("IT-parallel BLEU:", sum(r[2]["score"] for r in results) / len(results))
# # News corpus
# # ...


