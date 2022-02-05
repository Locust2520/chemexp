from .explained_mol_graph import ExplainedMolGraph
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.pyplot import cm
import numpy as np
import shutil
import json


def exp_to_json(exp: ExplainedMolGraph, filename: str):
    with open(filename, 'w') as f:
        json.dump({
            "a2b": exp.a2b,
            "b2a": exp.b2a,
            "b2revb": exp.b2revb,
            "f_atoms": exp.f_atoms.tolist(),
            "f_bonds": exp.f_bonds.tolist(),
            "n_atoms": exp.n_atoms,
            "n_bonds": exp.n_bonds,
            "smiles": exp.smiles
        }, f)


def json_to_exp(filename: str):
    with open(filename, 'r') as f:
        exp = json.load(f)
    return ExplainedMolGraph(exp["smiles"], exp["f_atoms"], exp["f_bonds"])


def exp_to_png(exp: ExplainedMolGraph, filename: str, cmap="bwr", size=(1200, 675)):
    # define a color map
    cmap = cm.get_cmap(cmap)
    color = lambda x: cmap(x)[:3]
    # get the corresponding molecule with RDKit
    mol = Chem.MolFromSmiles(exp.smiles)
    # create a dictionary Atom id --> Mean contribution
    atom_colors = dict()
    for i, e in enumerate(exp.f_atoms):
        atom_colors[i] = np.mean(e)

    # create a dictionary Bond id --> Mean contribution
    # (however the bonds are oriented, so we take the average of the two directions)
    bond_colors = dict()
    i = 0
    for a1 in range(exp.n_atoms):
        for a2 in range(a1 + 1, exp.n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            e = np.mean(np.concatenate((exp.f_bonds[i], exp.f_bonds[i + 1])))
            bond_colors[bond.GetIdx()] = e
            i += 2

    # convert mean contributions to colors
    values = list(atom_colors.values()) + list(bond_colors.values())
    maxi = max(abs(min(values)), abs(max(values)))
    for colors in atom_colors, bond_colors:
        for k in colors:
            colors[k] = color(colors[k]/maxi + 0.5)

    # draw the molecule with RDKit
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    options = d.drawOptions()
    options.setAtomPalette({})
    d.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol,
                                       highlightAtoms=list(range(exp.n_atoms)),
                                       highlightAtomColors=atom_colors,
                                       highlightBonds=list(range(exp.n_bonds // 2)),
                                       highlightBondColors=bond_colors)
    # save the figure
    d.WriteDrawingText(filename)
    print("Figure saved to " + filename)


def atoms_exp_to_png(exp: ExplainedMolGraph, feature: int, filename: str, cmap="bwr", size=(1200, 675)):
    # define a color map
    cmap = cm.get_cmap(cmap)
    color = lambda x: cmap(x)[:3]
    # get the corresponding molecule with RDKit
    mol = Chem.MolFromSmiles(exp.smiles)
    # create a dictionary Atom id --> Mean contribution
    atom_colors = dict()
    bonds_max = abs(exp.f_bonds).max() if len(exp.f_bonds) > 0 else 0
    atoms_max = abs(exp.f_atoms).max() if len(exp.f_atoms) > 0 else 1
    max_exp = max(bonds_max, atoms_max) * 2
    if max_exp == 0: max_exp = 1
    for i, e in enumerate(exp.f_atoms):
        atom_colors[i] = e[feature] / max_exp

    # convert mean contributions to colors
    for k in atom_colors:
        atom_colors[k] = color(atom_colors[k] + 0.5)

    # draw the molecule with RDKit
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    options = d.drawOptions()
    options.setAtomPalette({})
    d.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol,
                                       highlightAtoms=list(range(exp.n_atoms)),
                                       highlightAtomColors=atom_colors)
    # save the figure
    d.WriteDrawingText(filename)
    print("Figure saved to " + filename)


def bonds_exp_to_png(exp: ExplainedMolGraph, feature: int, filename: str, cmap="bwr", size=(1200, 675)):
    # define a color map
    cmap = cm.get_cmap(cmap)
    color = lambda x: cmap(x)[:3]
    # get the corresponding molecule with RDKit
    mol = Chem.MolFromSmiles(exp.smiles)

    # create a dictionary Bond id --> Mean contribution
    # (however the bonds are oriented, so we take the average of the two directions)
    bond_colors = dict()
    bonds_max = abs(exp.f_bonds).max() if len(exp.f_bonds) > 0 else 0
    atoms_max = abs(exp.f_atoms).max() if len(exp.f_atoms) > 0 else 1
    max_exp = max(bonds_max, atoms_max) * 2
    if max_exp == 0: max_exp = 1
    i = 0
    for a1 in range(exp.n_atoms):
        for a2 in range(a1 + 1, exp.n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            e = (exp.f_bonds[i][feature] + exp.f_bonds[i + 1][feature]) / 2
            bond_colors[bond.GetIdx()] = e / max_exp
            i += 2

    # convert mean contributions to colors
    for k in bond_colors:
        bond_colors[k] = color(bond_colors[k] + 0.5)

    # draw the molecule with RDKit
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    options = d.drawOptions()
    options.setAtomPalette({})
    d.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol,
                                       highlightBonds=list(range(exp.n_bonds // 2)),
                                       highlightBondColors=bond_colors)
    # save the figure
    d.WriteDrawingText(filename)
    print("Figure saved to " + filename)


def mol_to_png(mol, filename: str, size=(1200, 675), use_color=False):
    # get the corresponding molecule with RDKit
    try:
        mol = Chem.MolFromSmiles(mol)
    except:
        mol = None
    if mol is None:
        cd = "/".join(__file__.split("/")[:-1])
        if size[0]/size[1] < 1.4:
            shutil.copyfile(cd + "/static/smiles_error.png", filename)
        else:
            shutil.copyfile(cd + "/static/smiles_error_rect.png", filename)
    else:
        # draw the molecule with RDKit
        d = rdMolDraw2D.MolDraw2DCairo(*size)
        if not use_color:
            options = d.drawOptions()
            options.setAtomPalette({})
            d.SetDrawOptions(options)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        # save the figure
        d.WriteDrawingText(filename)
