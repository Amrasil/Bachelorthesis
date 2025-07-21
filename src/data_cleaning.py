import logging
import csv
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame


class DataCleansing:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        logging.basicConfig(
        filename = 'DataCleansing.log',
        filemode ='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initialization successful')

    def drop_nan_columns(self):
        # Drop Columns with only NaN-Values
        nan_columns = self.df.columns[self.df.isna().all()].to_list()
        self.df = self.df.drop(columns = nan_columns)
        self.logger.info(f'Deleted  {len(nan_columns)} NaN Columns: {nan_columns}')
        return self

    def drop_single_value_columns(self):
        # Drop Columns containing the same value in each instance
        single_value_columns = self.df.columns[self.df.nunique(dropna=False) <= 1].to_list()
        self.df = self.df.drop(columns = single_value_columns)
        self.logger.info(f'Deleted  {len(single_value_columns)} single-value Columns: {single_value_columns}')
        return self

    def drop_duplicates(self):
        # Drop duplicate instances
        count = self.df.duplicated(keep='first').sum()
        self.df = self.df.drop_duplicates(keep='first')
        self.logger.info(f'Deleted  {count} duplicate instances')
        return self

    def drop_infinity_instances(self):
        # Drop instances containing infinity values
        mask = self.df.isin([np.inf, -np.inf]).any(axis=1)
        count = mask.sum()
        self.df = self.df[~mask]
        self.logger.info(f'Deleted  {count} instances with infinite values')
        return self

    def drop_nan_instances(self):
        # Drop instances containing NaN values
        count = self.df.isna().any(axis=1).sum()
        self.df = self.df.dropna()
        self.logger.info(f'Deleted  {count} instances with NaN values')
        return self


    def get_label_data(self) -> DataFrame:
        labels = self.df[' Label'].unique()
        data = []
        total_count = self.df.shape[0]
        current_count = 0
        for label in labels:
            count = self.df[self.df[' Label'] == label].shape[0]
            current_count += count
            data.append([label, count, round(count/total_count, 6)])
        return pd.DataFrame(data=np.array(data), columns=['Label', 'Number of instances', 'of total instances'])

    def merge_labels(self):
        # reduce the number of unique labels by merging
        new_labels = {
            'BENIGN': 'BENIGN',
            'DDoS': 'DDoS',
            'PortScan':'PortScan',
            'Bot': 'Bot',
            'Infiltration': 'Infiltration',
            'Web Attack � Brute Force': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Web Attack � Sql Injection': 'Web Attack',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'DoS slowloris': 'Dos/DDos',
            'DoS Slowhttptest': 'Dos/DDos',
            'DoS Hulk': 'Dos/DDos',
            'DoS GoldenEye': 'Dos/DDos',
            'Heartbleed':  'Dos/DDos',
        }
        self.df[' Label'] = self.df[' Label'].map(new_labels)
        return self

    def feature_list(self):
        return self.df.columns

    @classmethod
    def create_unique_csv(cls, df: pd.DataFrame, csv_name: str) -> None:
        df.to_csv(csv_name)

    def create_csv(self):
        # Create new csv file
        self.df.to_csv('CICIDS2017.csv')

    def get_df(self):
        return self.df


def pre_merge():
    df_original = pd.read_csv('CICIDS2017_original.csv')

    cleaner = DataCleansing(df_original)

    df_clean = (
        cleaner
        .drop_nan_columns()
        .drop_single_value_columns()
        .drop_duplicates()
        .drop_infinity_instances()
        .drop_nan_instances()
    )
    cleaner.create_unique_csv(df_clean.get_label_data(),'pre_merge_label_data.csv')

    df_clean.create_csv()

def merge():
    df_pre_merge = pd.read_csv('CICIDS2017.csv')
    cleaner = DataCleansing(df_pre_merge)
    cleaner.merge_labels()
    cleaner.create_unique_csv(cleaner.get_label_data(),'post_merge_label_data.csv')

def post_merge():
    df = pd.read_csv('CICIDS2017.csv')
    cleaner = DataCleansing(df)
    features = cleaner.feature_list()
    with open('features_for_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Features'])  # Kopfzeile
        for feature in features:
            writer.writerow([feature])


post_merge()