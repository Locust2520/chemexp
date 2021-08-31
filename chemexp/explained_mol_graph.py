from chemprop.features.featurization import MolGraph
import numpy as np

class ExplainedMolGraph(MolGraph):
    def __init__(self, smiles, f_atoms, f_bonds):
        super(ExplainedMolGraph, self).__init__(smiles)
        self.f_atoms = np.array(f_atoms)
        self.f_bonds = np.array(f_bonds)
        self.smiles = smiles
