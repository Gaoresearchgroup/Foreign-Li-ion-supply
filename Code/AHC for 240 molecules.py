import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import preprocessing
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import rdkit
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

from rdkit.ML.Cluster import Butina

from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors3D
import matplotlib.ticker as ticker
from rdkit.Chem.Descriptors3D import Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PMI1, PMI2, PMI3, \
    RadiusOfGyration, SpherocityIndex
from PyTool import polt_cluster, plot_distribution
import os

#
plt.rcParams['font.family'] = 'Arial'
figsize_cluster = (10, 2)
figsize_bar = (10, 2)
fontsize = 5
fontsize_x = 3.25
width = 0.25

path_data = 'Data/molecules_2000.xlsx'

path_SE = './Data/end/data_SE.xlsx'
fig_SE_cluster = './Figure/SE/end/'
fig_SE_distribution = './Figure/SE/end/'

path_AL = './Data/end/data_AL.xlsx'
fig_AL_cluster = './Figure/AL/end/'
fig_AL_distribution = './Figure/AL/end/'

data = pd.read_excel(path_data)
data_SE = pd.read_excel(path_SE)
data_AL = pd.read_excel(path_AL)

# 30 features
X_SE = data_SE.iloc[:, 3:]
X_AL = data_AL.iloc[:, 3:]
'''
X_SE['Anode Limit'] = X_SE['Anode Limit'].apply(lambda x: 1.5*x)
X_SE['Solubility Energy'] = X_SE['Solubility Energy'].apply(lambda x: 1.5*x)
X_AL['Anode Limit'] = X_AL['Anode Limit'].apply(lambda x: 1.5*x)
X_AL['Solubility Energy'] = X_AL['Solubility Energy'].apply(lambda x: 1.5*x)
'''

print(X_SE.head(5))
print(X_AL.head(5))

# Solubility Energy AgglomerativeClustering
model_SE = AgglomerativeClustering(distance_threshold=0.5, n_clusters=None)
model_SE = model_SE.fit(X_SE)
polt_cluster(model_SE, len(X_SE), fig_SE_cluster)
plot_distribution(model_SE, data, len(data), fig_SE_distribution, 'Solubility Energy')

# Anode Limit AgglomerativeClustering
model_AL = AgglomerativeClustering(distance_threshold=0.5, n_clusters=None)
model_AL = model_AL.fit(X_AL)
polt_cluster(model_AL, len(X_AL), fig_AL_cluster)
plot_distribution(model_AL, data, len(data), fig_AL_distribution, 'Anode Limit')

# Analyse Solubility Energy AgglomerativeClustering
