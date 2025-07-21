import pandas as pd
import numpy as np

# df[' Label'] = df[' Label'].apply(lambda x: 'Malicious' if x != 'BENIGN' else 'Benign')
"""df_malicious = df[df[' Label'] != 'BENIGN']
print(df_malicious[' Label'].value_counts())
sns.countplot(data=df_malicious, x=' Label')
plt.title('Anzahl bösartiger Verbindungen')
plt.show()"""



"""print(df.isnull().sum().sort_values(ascending=False))
new_df = df.describe()
pd.DataFrame.to_csv(new_df, 'Extremwerte.csv')"""

"""print(df.shape)
print(df.columns)
print(df[' Label'].value_counts())"""

def feature_correlation(dataframe, label):
    df_label = dataframe[dataframe[' Label'] == label]
    df_label = df_label.drop(columns=[' Label'])
    source_features = len(df_label.columns)
    pd.DataFrame.to_csv(df_label, 'TOO.csv')
    corr_threshold = 0.8
    corr_matrix = df_label.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    df_reduced = df_label.drop(columns=to_drop)
    reduced_features = len(df_reduced.columns)
    pd.DataFrame.to_csv(df_reduced, 'FeatureCorrelation_' + label + '.csv')
    return label, source_features, reduced_features

def all_feature_correlation(dataframe):
    values_label = dataframe[' Label'].unique()
    result = []
    for label in values_label:
        result.append(feature_correlation(dataframe, label))
    return result

def target_feature_correlation(dataframe):
    dataframe[' Label'] = dataframe[' Label'].apply(lambda x: 0 if x != 'BENIGN' else 1)
    dataframe = dataframe.drop(columns=[' Bwd PSH Flags',
                                        ' Bwd URG Flags',
                                        'Fwd Avg Bytes/Bulk',
                                        ' Fwd Avg Packets/Bulk',
                                        ' Fwd Avg Bulk Rate',
                                        ' Bwd Avg Bytes/Bulk',
                                        ' Bwd Avg Packets/Bulk',
                                        'Bwd Avg Bulk Rate'])
    correlations = dataframe.corr()[' Label'].sort_values(ascending=False)
    return correlations.head(10), correlations.tail(10)


df = pd.read_csv('CICIDS2017_original.csv')
print(target_feature_correlation(df))
