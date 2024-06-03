# Going Back and Forth: A Study into Back Translation for Data Augmentation in Neural Machine Translation

The following repository contains the code and data used in the paper "Going Back and Forth in Neural Machine Translation", written in submission for the course "Natural Language Processing 2" at the University of Amsterdam.

## Project

We investigate the use of back translation as a data augmentation technique for neural machine translation. We take a closer look at the effects of back translation on the quality of the generated data and the performance of the model. 

### Research Questions

-  Does BT perform better for low-resource domain adaptation than
  fine-tuning on a limited parallel corpus? Furthermore, are they
  complementary? 
- How does data selection improve backtranslation and MT performance? In particular, we aim to enhance the quality of synthetic training data by using a diversity metric. This metric filters the synthetic source-side data to select the most diverse and informative samples, based on type-token ratio. This approach seeks to balance the trade-off between data quality and quantity, aiming to reduce noise and redundancy in the training data and thereby improve overall machine translation performance. 
-  How can backtranslation be optimised or extended for this low-resource
  scenario? Iterative backtranslation has been shown to an effective approch in order to boost model performance \cite{hoang2018iterative}. We explore its effects and provide an additional baseline.
-  Can we utilize current SOTA NMT systems in order to adapt less intense past NMT models? We hypothesize that introducing such teacher-student relationship to the backtranslation pipeline can improve the performance of older systems and make them more viable within target applications.

### 

### Code

The code is structured as follows:

```text
.
|-- Data # Contains the data used in the project
|-- scripts # Contains the main scripts for data generation and training using fairseq
|-- utils # Contains the main model and data processing utilities
|-- tests # Contains generated translations from the test set and more
├── notebooks
│   ├── data_filtering.ipynb # Data filtering and selection notebook
│   ├── notebook.ipynb # Main working notebook
│   └── results.ipynb # Results and analysis notebook
|-- train.py # Main training script
|-- generate.py # Data generation script
|-- requirements.txt # Required packages
|-- calculate_bleu.py # BLEU and COMET score calculation utility
```

Most of the available data is stored in the `Data` directory. For reproductions, we also provide sample generated machine translations in the `tests` directory.

## Getting Started

### Dependencies

* Python 3.x
* See `requirements.txt` for required Python packages

### Installing

* Clone the repository
* Install the required packages with `pip install -r requirements.txt`

### Executing program

* Run `python train.py` to train the model; additional arguments can be found in the script by running `python train.py --help`
* Run `python generate.py` to generate new data; additional arguments can be found as above
