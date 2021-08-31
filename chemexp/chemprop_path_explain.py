from chemprop.features.featurization import MolGraph, BatchMolGraph
from chemprop.models import MoleculeModel
from .explained_mol_graph import ExplainedMolGraph
from .path_explainer_torch import PathExplainerTorch
from copy import deepcopy
import torch
import numpy as np


class PathExplainer:
    # +1 is to include room for common values, as mentioned in chemprop.features.featurization, line 31
    feature_sizes = [
        100 + 1, # atom_type
        6 + 1,   # #bonds
        5 + 1,   # formal_charge
        4 + 1,   # charilaty
        5 + 1,   # #Hs
        5 + 1,   # hybridization
        1,       # aromaticity
        1        # atomic_mass
    ]

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
        self.batch_mol = BatchMolGraph([mol])
        atom_size = len(mol.f_atoms[0])
        atoms = np.concatenate(mol.f_atoms)
        bonds = np.concatenate(list(map(lambda x: x[atom_size:], mol.f_bonds)))
        atom_size = len(mol.f_atoms[0])
        bond_size = len(mol.f_bonds[0]) - atom_size
        self.input = torch.Tensor(np.concatenate([np.zeros(atom_size), atoms, np.zeros(bond_size), bonds]))

    def _molecule_to_compact(self, mol: str):
        mol = MolGraph(mol)
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
        return np.array(input_features, dtype="float32")

    def _tensor_to_batch_mol(self, array) -> BatchMolGraph:
        # parsing atoms features
        atom_size = len(self.mol.f_atoms[0])
        bond_size = len(self.mol.f_bonds[0]) - atom_size
        atoms_length = (self.mol.n_atoms + 1) * atom_size
        f_atoms = array[:atoms_length].reshape((-1, atom_size))
        f_bonds = array[atoms_length:].reshape((-1, bond_size))
        # creating a copy of the MolGraph and change its features
        batch_mol = deepcopy(self.batch_mol)
        f_bonds = torch.cat([f_atoms[batch_mol.b2a], f_bonds], dim=1)
        batch_mol.f_atoms = f_atoms
        batch_mol.f_bonds = f_bonds
        return batch_mol

    def _feature_to_indexes(self, feature: int):
        atom_size = self.batch_mol.f_atoms.shape[1]
        bond_size = self.batch_mol.f_bonds.shape[1]
        if feature < len(self.feature_sizes):
            i = sum(self.feature_sizes[:feature])
            j = i + self.feature_sizes[feature]
            return [atom_size*k + i + self.batch_mol.f_atoms[k,i:j].argmax() for k in range(1, self.mol.n_atoms+1)]
        else:
            feature -= len(self.feature_sizes)
            feature_sizes = [5, 1, 1, 7]
            i = atom_size + sum(feature_sizes[:feature])
            j = i + feature_sizes[feature]
            return [i + bond[i:j].argmax() for bond in self.batch_mol.f_bonds[1:]]

    def predict_proba(self, inputs, softmax_output=False):
        """Compute the model's prediction for arrays inputs

        inputs: list of arrays, containing features for the current molecule
        softmax_output: flag to apply a softmax. If False the original linear output is returned

        Returns a numpy array of predictions, of shape (len(inputs), nb_classes)
        """
        if self.mol is None:
            raise AttributeError("You must set a molecule first (set_molecule method).")
        # converting arrays to MolGraphs
        mols = list(map(self._tensor_to_batch_mol, inputs))
        # getting the model predictions for the molecules
        output = self.model(mols)
        output = torch.Tensor.cpu(output)
        #if softmax_output:
        #    output = softmax(output, dim=1)
        #else:
        #    output=functional.normalize(output, p=1, dim=1)
        return output

    def explain_molecule(self, mol: str, n=1) -> ExplainedMolGraph:
        self._set_molecule(mol)
        explainer = PathExplainerTorch(lambda batch: self.predict_proba(batch)[:,n:n+1])
        input_tensor = self.input.reshape((1, -1))
        input_tensor.requires_grad_(True)
        baseline = torch.zeros_like(input_tensor)
        attributions = explainer.attributions(
            input_tensor=input_tensor,
            baseline=baseline,
            num_samples=256,
            use_expectation=True
        )
        atom_size = len(self.mol.f_atoms[0])
        bond_size = len(self.mol.f_bonds[0]) - atom_size
        atoms_length = (self.mol.n_atoms + 1) * atom_size
        atoms_att = attributions[0][:atoms_length].detach().split(atom_size)
        bonds_att = attributions[0][atoms_length:].detach().split(bond_size)
        atoms_att = atoms_att[1:]
        bonds_att = bonds_att[1:]
        f_atoms = []
        indexes = [0] + np.cumsum(self.feature_sizes).tolist()
        for atom in atoms_att:
            exp = np.zeros(len(self.feature_sizes))
            for i, (j, k) in enumerate(zip(indexes[:-1], indexes[1:])):
                exp[i] = atom[j:k].sum()
            f_atoms.append(exp)
        f_bonds = []
        for bond in bonds_att:
            bond = bond[-14:]
            exp = np.zeros(4)
            exp[0] = bond[:5].sum()  # bond type
            exp[1:3] = bond[5:7]     # conjugated / in ring
            exp[3] = bond[7:].sum()  # stereo (none, any, E/Z, or cis/trans)
            f_bonds.append(exp)
        exp = ExplainedMolGraph(mol, f_atoms, f_bonds)
        return exp

    def explain_interactions(self, mol: str, feature1: int, feature2: int, n=1) -> ExplainedMolGraph:
        self._set_molecule(mol)
        explainer = PathExplainerTorch(lambda batch: self.predict_proba(batch)[:,n:n+1])
        input_tensor = self.input.reshape((1, -1))
        input_tensor.requires_grad_(True)
        baseline = torch.zeros_like(input_tensor)
        featuresA = self._feature_to_indexes(feature1)
        featuresB = self._feature_to_indexes(feature2)
        interactions = explainer.single_interactions(
            input_tensor=input_tensor,
            baseline=baseline,
            featuresA=featuresA,
            featuresB=featuresB,
            num_samples=256,
            use_expectation=True,
            verbose=True
        )[0].detach().unsqueeze(-1).numpy()
        if feature1 < len(self.feature_sizes):
            f_atoms = [interactions[i,i] for i in range(len(featuresA))]
            f_bonds = []
        else:
            f_atoms = []
            f_bonds = [interactions[i,i] for i in range(len(featuresA))]
        return ExplainedMolGraph(mol, f_atoms, f_bonds)

    def __call__(self, *args, **kwargs):
        return self.predict_proba(*args, **kwargs)
