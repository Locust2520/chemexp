# Chemprop, why?

A Chemprop explainer based on ![LIME](https://github.com/marcotcr/lime) and ![PathExplain](https://github.com/suinleelab/path_explain).

## Installation

- Make sure your Python version is >= **3.8**
- `conda` is required for ChemProp. If it is not installed on your machine, you can take a look at ![miniconda](https://docs.conda.io/en/latest/miniconda.html)

1. Install ![Chemprop](https://github.com/chemprop/chemprop)
2. Then simply run `pip3 install cpwhy`
3. You're good to go!

## Explanation

### Using Python


import cpwhy
from chemprop.utils import load_checkpoint

model = cpwhy.ExplanationModel(load_checkpoint("models/fusion_db_1.pt"))
exp = model.explain_molecule("COC(=N)Cc1ccccc1")
cpwhy.exp_to_png(exp, "mol.png")


A little more detail is available in ![test/test.py](test/test.py).

### Web interface

--> Go to the web interface repository
