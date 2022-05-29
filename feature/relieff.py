import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.txt', sep='\t')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))
print(np.mean(cross_val_score(clf, features, labels)))