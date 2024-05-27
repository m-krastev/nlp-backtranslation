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

# Isn't it so nice and clean now? I went through FOUR different ways of doing this before I thought of this one. WDWFDGFEQWDQWFGA

SRC = "de"
TGT = "en"
BATCH_SIZE = 16
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

# input_ids = tokenizer.encode( "Maschinelles Lernen ist großartig, oder?", return_tensors="pt")
# outputs = model.generate(input_ids)
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(decoded)  # Machine Learning is great, isn't it?

# Copying settings from the original file though probably unneeded..
model.generation_config.length_penalty = 1.2
model.generation_config.num_beams = 5

model_pl = TranslationLightning(
    model,
    tokenizer,
    lr=1e-4,
    adam_beta=(0.9, 0.98),
    weight_decay=1e-4,
    test_folder=test_folder,
)
trainer = Trainer(max_epochs=20, gradient_clip_val=0.1)

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

# results = trainer.predict(model_pl, datamodule=data)

# %%
# Now we can train the model
trainer.fit(model_pl, datamodule=data)
trainer.predict(model_pl, datamodule=data)
