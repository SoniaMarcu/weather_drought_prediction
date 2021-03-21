import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sns.set_style('white')

files={}

def save_file_structure():

    for dirname, _, filenames in os.walk('kaggle'):
        for filename in filenames:
            if 'train' in filename:
                files['train'] = os.path.join(dirname, filename)
            if 'valid' in filename:
                files['valid'] = os.path.join(dirname, filename)
            if 'test' in filename:
                files['test'] = os.path.join(dirname, filename)

    files['soil']=os.path.join('kaggle/input','soil_data.csv')


def concatenate_dataframes():

    df=pd.read_csv(files['train'])
    df.append(pd.read_csv(files['valid']))
    df.append(pd.read_csv(files['test']))

    soil_df=pd.read_csv(files['soil'])
    df_complete = pd.merge(df, soil_df, on=['fips'])
    df_complete.fillna(0, inplace=True)

    return df

def eliminate_useless_columns():

    df_complete=concatenate_dataframes()
    plt.figure(figsize=(100, 50))
    train_df_correlation=df_complete.corr()

    corr_matrix = train_df_correlation.abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95 or less than 0.1
    to_drop = [column for column in upper.columns if (any(upper[column] > 0.85) or any(upper[column]) < 0.1)]
    df_complete.drop(to_drop, axis=1, inplace=True)


    heatmap = sns.heatmap(df_complete.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 40}, pad=12);

    plt.savefig('heatmap.png', dpi=400, bbox_inches='tight')
    df_complete.to_pickle("./train_df_complete.pkl")

    return df_complete


def get_train_and_test_data():
    save_file_structure()
    df=eliminate_useless_columns()
    column_names=df.keys()
    print(column_names)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df.fillna(0, inplace=True)

    Y = df.pop('score').values

    X = scaler.fit_transform(df.iloc[:, 2:])
    X_df=pd.DataFrame(X, columns=['PRECTOT', 'PS', 'QV2M', 'T2M_RANGE', 'WS10M',
       'WS10M_MIN', 'WS50M_RANGE'])
    X_df['fips']=df['fips']
    X_df['date']=df['date']

    xTrain=X[0: 13234752, :]
    yTrain=Y[0:13234752]
    xTest=X[13234752:, :]
    yTest=Y[13234752:]

    return xTrain, xTest, yTrain, yTest


