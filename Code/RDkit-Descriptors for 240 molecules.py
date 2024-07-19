import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
from rdkit.Chem import Lipinski
from sklearn.preprocessing import MinMaxScaler
import os


def calculate_descriptors_SE(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    descriptors = {
        'NumAtoms': mol.GetNumAtoms(),
        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'FractionCSP3': Lipinski.FractionCSP3(mol),
        'FormalCharge': sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MolWt': Descriptors.MolWt(mol),
        'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol)
    }

    # Embed molecule and optimize
    AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=5000)
    AllChem.MMFFOptimizeMolecule(mol)

    # Add hydrogens and re-embed
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 3D Descriptors
    descriptors.update({
        'Asphericity': Descriptors3D.Asphericity(mol),
        'Eccentricity': Descriptors3D.Eccentricity(mol),
        'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
        'NPR1': Descriptors3D.NPR1(mol),
        'NPR2': Descriptors3D.NPR2(mol),
        'PMI1': Descriptors3D.PMI1(mol),
        'PMI2': Descriptors3D.PMI2(mol),
        'PMI3': Descriptors3D.PMI3(mol),
        'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
        'SpherocityIndex': Descriptors3D.SpherocityIndex(mol)
    })

    return pd.Series(descriptors)


def calculate_descriptors_AL(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(smiles)
        raise ValueError("Invalid SMILES string")

    descriptors = {
        'NumAtoms': mol.GetNumAtoms(),
        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'FractionCSP3': Lipinski.FractionCSP3(mol),
        'FormalCharge': sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles(mol),
        'LogP': Descriptors.MolLogP(mol),
        'MR': Descriptors.MolMR(mol),
    }

    # Calculate Valence_S
    valence_s = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            valence_s = atom.GetTotalValence()
            break
    descriptors['Valence_S'] = valence_s

    # Calculate FractionCSP2
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2)
    sp2_ratio = sp2_carbons / total_carbons if total_carbons > 0 else 0
    descriptors['FractionCSP2'] = sp2_ratio

    # Calculate conjugated_pi_electrons
    conjugated_pi_electrons = sum(2 for bond in mol.GetBonds() if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.AROMATIC])
    conjugated_pi_electrons += sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2)
    descriptors['conjugated_pi_electrons'] = conjugated_pi_electrons

    # Embed molecule and optimize
    AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=5000)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 3D Descriptors
    descriptors.update({
        'Asphericity': Descriptors3D.Asphericity(mol),
        'Eccentricity': Descriptors3D.Eccentricity(mol),
        'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
        'NPR1': Descriptors3D.NPR1(mol),
        'NPR2': Descriptors3D.NPR2(mol),
        'PMI1': Descriptors3D.PMI1(mol),
        'PMI2': Descriptors3D.PMI2(mol),
        'PMI3': Descriptors3D.PMI3(mol),
        'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
        'SpherocityIndex': Descriptors3D.SpherocityIndex(mol)
    })

    return pd.Series(descriptors)





if __name__ == '__main__':
    data = pd.read_excel('./Data/molecules_end.xlsx')
    #data['Anode Limit'].fillna(0, inplace=True)
    #data['Solubility Energy'].fillna(0, inplace=True)
    print(data.shape)
    #data['SMILES'] = data['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    descriptors_sol = data['SMILES'].apply(calculate_descriptors_SE)
    data_sol = pd.concat([data, descriptors_sol], axis=1)
    scaler = MinMaxScaler()
    scaled_data_sol = scaler.fit_transform(data_sol.iloc[:, 1:])
    scaled_data_sol = pd.DataFrame(scaled_data_sol, columns=data_sol.columns[1:])
    data_sol = pd.concat([data_sol.iloc[:, :1], scaled_data_sol], axis=1)
    print(data_sol.head(5))
    data_sol.to_excel('./Data/end/data_SE.xlsx',index=False)
    
    #print(data_sol.head(5))



