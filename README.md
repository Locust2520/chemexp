# Chemexp

A Chemprop explainer based on [LIME](https://github.com/marcotcr/lime) and [PathExplain](https://github.com/suinleelab/path_explain).

![](images/exp_atom_0.png) ![](images/exp_bond_2.png)


## Introduction 💊

**[Chemprop](https://github.com/chemprop/chemprop)** is the implementation of message passing neural networks for molecular property prediction and classification.  

**Chemexp** (this repo) is built on top of Chemprop in order to explain *why* a classification has been made.
We are then able to visualize the importance of structures or properties inside the molecule, w.r.t. the classification.  
Chemexp is therefore a way to visualize *what has been learned* from Chemprop.

This project has been done as part of my end-of-study internship in the [Orpailleur](https://orpailleur.loria.fr/) team (LORIA, Nancy, France).  
It has been applied in the discovery of key properties of molecules that provide them antibiotic behaviors. ([read more](https://hal.inria.fr/hal-03371070))


## Installation 🖥️

- Make sure your Python version is >= **3.8**
- `conda` is required for ChemProp. If it is not installed on your machine, you can take a look at [miniconda](https://docs.conda.io/en/latest/miniconda.html)

1. Install [Chemprop](https://github.com/chemprop/chemprop)
2. Then simply run `pip install chemexp`
3. You're good to go!


## Explanation 🔎

### Using Python 🐍

```python
import chemexp
from chemprop.utils import load_checkpoint

model = chemexp.ExplanationModel(load_checkpoint("models/fusion_db_1.pt"))
exp = model.explain_molecule("COC(=N)Cc1ccccc1")
chemexp.exp_to_png(exp, "mol.png")
```

A little more detail is available in [test/test.py](test/test.py).

### Web interface 🐧

![](images/screenshot.png)

Although you can use this Python module in your scripts, you can also experiment a user-friendly interface.  
⚠️ **Warning:** this only works on Linux for the moment *(this is due to the [linux-specific paths](chemexp/server.py#L26) used to save some files, but it can be adapted for Windows or Mac)*.

Use the following command to run the web server:

```shell
python -m chemexp <path>
```

where `path` is a folder containing chemprop models / checkpoints (.pt files)


## Authors 👥
- BELLANGER Clément
- [BERNIER Fabien](https://wwwfr.uni.lu/snt/people/fabien_bernier)
- MAIGRET Bernard
- [NAPOLI Amedeo](https://members.loria.fr/ANapoli/)

To cite our work:
```
@mastersthesis{bernier:hal-03371070,
  TITLE = {{A Study about Explainability and Fairness in Machine Learning and Knowledge Discovery}},
  AUTHOR = {Bernier, Fabien},
  URL = {https://hal.inria.fr/hal-03371070},
  PAGES = {58},
  SCHOOL = {{TELECOM Nancy}},
  YEAR = {2021},
  MONTH = Sep,
  KEYWORDS = {machine learning ; explainability ; fairness ; antibiotic classification ; machine learning ; explicabilit{\'e} ; fairness ; classification antibiotique},
  PDF = {https://hal.inria.fr/hal-03371070/file/M_moire_ing_nieur_Fabien.pdf#chapter.5},
  HAL_ID = {hal-03371070},
  HAL_VERSION = {v1},
}
```
