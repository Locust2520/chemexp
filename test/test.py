import cpwhy
from chemprop.utils import load_checkpoint

# model = ExplanationModel(load_checkpoint("fusion_mc_937_x1_wo0_fp/fold_0/model_0/model.pt"))
explainer = cpwhy.PathExplainer(load_checkpoint("/home/fabien/Documents/Orpa/antibio-classification/models/fusion_db_1.pt"))
mols = [
    "CCN1CCN(C(=O)N[C@@H](C(=O)N[C@@H]2C(=O)N3C(C(=O)[O-])=C(C[N+]4(CCNC(=O)c5ccc(O)c(O)c5Cl)CCCC4)CS[C@H]23)c2ccccc2)C(=O)C1=O",
    "COC(=O)c1c(O)ccc2nc3c(C(=O)O)cccc3nc12",
    "CC(O)C1C(=O)N2C(C(=O)O)=C(SC3CNC(C(=O)N4CCCN(CCO)CC4)C3)C(C)C12",
    "O=C(O)CS(=O)(=O)c1ccc(OCc2cccc(-c3ccccc3Cl)c2)cc1",        # pas antibio (les autres le sont)
    "CCO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)[O-])=C(C[n+]3ccc(SCCCN)cc3)CSC12)c1csc(N)n1"
]

# for i, mol in enumerate(mols):
#     print("Explaining mol", i)
#     exp = model.explain_molecule(mol)
#     exp_to_png(exp, f"/tmp/mol{i}.png")
# mol_to_png("O=C(O)CS(=O)(=O)c1ccc(OCc2cccc(-c3ccccc3Cl)c2)cc1", "/tmp/test.png")
exp = explainer.explain_molecule("COC(=N)Cc1ccccc1", n=0)
cpwhy.exp_to_png(exp, "/tmp/mol2.png")
