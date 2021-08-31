from chemprop.features.featurization import MolGraph, BatchMolGraph
from chemprop.models import MoleculeModel
from .explained_mol_graph import ExplainedMolGraph
from torch.nn.functional import softmax
from lime.lime_tabular import LimeTabularExplainer
from copy import deepcopy
from torch import Tensor
from torch.nn import functional
import numpy as np


class LIMEExplainer:
    # +1 is to include room for common values, as mentioned in chemprop.features.featurization, line 31
    feature_sizes = {
        "atom_type": 100+1,
        "#bonds": 6+1,
        "formal_charge": 5+1,
        "chirality": 4+1,
        "#Hs": 5+1,
        "hybridization": 5+1
    }.values()

    possible_atoms = [[i%101, i%7, i%6, i%5, i%6, i%6, i%2, i/100] for i in range(101)]
    possible_bonds = [[i%5, i%2, i%2, i%7] for i in range(7)]

    def __init__(self, model: MoleculeModel, mol: str=None):
        self.model = model
        self.model.eval() # Trigger evaluation mode
        if mol is not None:
            self._set_molecule(mol)
        else:
            self.mol = None

    def _set_molecule(self, mol: str):
        mol = MolGraph(mol)
        self.mol = mol
        input_features = []
        self.categorical = []
        self.atom_slices = [0]
        values = []
        i = 0
        # atoms to array
        for f_atom in mol.f_atoms:
            values = []
            j = 0
            for size in self.feature_sizes:
                values.append(f_atom[j:j+size].index(1))
                j += size
            values.extend(f_atom[j:])
            input_features += values
            self.categorical += list(range(i, i+len(values)-1))
            i += len(values)
            self.atom_slices.append(i)
        self.atom_size = len(values)
        # bonds to array
        self.bond_slices = [i]
        for f_bond in mol.f_bonds:
            f_bond = f_bond[-14:]
            values = []
            values.append(f_bond[:5].index(1))  # bond type
            values.extend(f_bond[5:7])          # conjugated / in ring
            values.append(f_bond[7:].index(1))  # stereo (none, any, E/Z, or cis/trans)
            input_features += values
            self.categorical += list(range(i, i+4))
            i += 4
            self.bond_slices.append(i)
        self.bond_size = len(values)
        self.input = np.array(input_features, dtype="float32")

    def _array_to_mol(self, array) -> MolGraph:
        # parsing atoms features
        f_atoms = []
        for i, j in zip(self.atom_slices[:-1], self.atom_slices[1:]):
            values = array[i:j]
            f_atom = []
            for k, size in enumerate(self.feature_sizes):
                feature = [0]*size
                feature[int(values[k])] = 1
                f_atom += feature
            f_atom.extend(values[-2:])
            f_atoms.append(f_atom)
        # parsing bonds features (and include atoms features)
        f_bonds = []
        bounds = zip(self.bond_slices[:-1], self.bond_slices[1:])
        for a, (i, j) in zip(self.mol.b2a, bounds):
            values = array[i:j]
            f_bond = [0]*14
            f_bond[int(values[0])] = 1
            f_bond[5:7] = values[1:3]
            f_bond[7 + int(values[3])] = 1
            f_bonds.append(np.concatenate((f_atoms[a], f_bond)))
        # creating a copy of the MolGraph and change its features
        mol = deepcopy(self.mol)
        mol.f_atoms = f_atoms
        mol.f_bonds = f_bonds
        return mol

    def predict_proba(self, inputs, softmax_output=False):
        """Compute the model's prediction for arrays inputs

        inputs: list of arrays, containing features for the current molecule
        softmax_output: flag to apply a softmax. If False the original linear output is returned

        Returns a numpy array of predictions, of shape (len(inputs), nb_classes)
        """
        if self.mol is None:
            raise AttributeError("You must set a molecule first (set_molecule method).")
        # converting arrays to MolGraphs
        mols = list(map(self._array_to_mol, inputs))
        # getting the model predictions for the molecules
        output = self.model([BatchMolGraph(mols)]).detach()
        output = Tensor.cpu(output)
        #if softmax_output:
        #    output = softmax(output, dim=1)
        #else:
        #    output=functional.normalize(output, p=1, dim=1)
        return output.numpy()

    def _select_class(self, predictions, n):
        """
        predictions: array of shape (_, nb_classes)
            (warning: coefficients of this array MUST be between 0 and 1)
        n: class to select

        Return an array of shape (_, 2):
            - first column corresponding to 1 - the class output
            - second column corresponding to the class output
        """
        class_output = predictions[:,n:n+1]
        return np.concatenate([1-class_output, class_output], axis=1)

    def explain_molecule(self, mol: str, n: int = 1) -> ExplainedMolGraph:
        """
        mol: molecule in SMILES format
        n: class number to explain (starting at 0)
        """
        self._set_molecule(mol)
        # Initializing the explainer
        dataset = []
        for i in range(101):
            instance = []
            for j in range(self.mol.n_atoms):
                instance += self.possible_atoms[(i+j)%101]
            for j in range(self.mol.n_bonds):
                instance += self.possible_bonds[(i+j)%7]
            dataset.append(instance)
        explainer = LimeTabularExplainer(np.array(dataset), categorical_features=self.categorical,
                                         discretize_continuous=False)
        # Getting the mean local contribution for each feature
        contribs = np.zeros(len(self.input))
        for _ in range(5):
            exp = explainer.explain_instance(self.input, lambda x: self._select_class(self.predict_proba(x), n),
                                             num_features=len(self.input), num_samples=1000)
            for i, value in exp.local_exp[1]:
                contribs[i] += value / 5
        # Making an ExplainedGraphMol and returning it
        f_atoms = contribs[:self.mol.n_atoms*self.atom_size].reshape((-1, self.atom_size))
        f_bonds = contribs[self.mol.n_atoms*self.atom_size:].reshape((-1, self.bond_size))
        exp_mol = ExplainedMolGraph(mol, f_atoms, f_bonds)
        return exp_mol

    def __call__(self, *args, **kwargs):
        return self.predict_proba(*args, **kwargs)
