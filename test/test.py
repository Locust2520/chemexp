import chemexp
from chemprop.utils import load_checkpoint

explainer = chemexp.PathExplainer(load_checkpoint("fusion_db_1.pt"))
exp = explainer.explain_molecule("COC(=N)Cc1ccccc1", n=0)
cpwhy.exp_to_png(exp, "/tmp/mol.png")
