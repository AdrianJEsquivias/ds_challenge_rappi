import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import ast
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


class prediccion:
    def __init__(self, nombre_excel):
        self.df = pd.read_excel(nombre_excel)
        self.df_limpio = self.limpiar_df(self.df)
        self.prediccion = self.predecir(self.df_limpio)
        self.df['Fraude'] = self.prediccion
        self.df[['transaction_id', 'user_id', 'género', 'linea_tc', 'interes_tc',
       'monto', 'fecha', 'hora', 'dispositivo', 'establecimiento', 'ciudad',
       'status_txn', 'is_prime', 'dcto', 'cashback', 'Fraude']].to_excel('prediccion_fraude.xlsx', index=False)
        print('Se guardaron las predicciones en el archivo prediccion_fraude.xlsx')

    def predecir(self, df):
        random_forest_classifier = joblib.load('modelo_clasificador.pkl')
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.fit_transform(df)
        predicciones = random_forest_classifier.predict(X_scaled)
        return predicciones


    def limpiar_df(self, df):
        encoder_genero = load('encoder_genero.joblib')
        encoder_dia = load('encoder_dia.joblib')
        encoder_establecimiento, categories_establecimiento = load('encoder_establecimiento.joblib')
        encoder_ciudad, categories_ciudad = load('encoder_ciudad.joblib')
                
        df_1 = pd.read_excel('pre.xlsx')
        df['mean_monto'] = 0
        df['numero_transacciones'] = 0
        for index, row in df.iterrows():
            aux1 = df_1[(df_1['user_id']==row['user_id'])&(df_1['fecha']<=row['fecha'])].copy()
            df.at[index, 'mean_monto'] = aux1['monto'].mean()
            df.at[index, 'numero_transacciones'] = aux1.shape[0]
        df_ciudad = df_1[['user_id', 'ciudad']].copy().drop_duplicates().sort_values(by='user_id').dropna()
        df['n_ciudad'] = df['user_id'].map(dict(zip(df_ciudad['user_id'], df_ciudad['ciudad'])))
        df['ciudad'] = df['ciudad'].fillna(df['n_ciudad'])
        df['ciudad'] = df['ciudad'].fillna(df_ciudad.ciudad.mode().values[0])
        max_labels = df_1.groupby('user_id')['establecimiento'].value_counts().unstack(fill_value=0).idxmax(axis=1).to_frame().reset_index()
        max_labels.columns = ['user_id', 'establecimiento']
        df['n_establecimiento'] = df['user_id'].map(dict(zip(max_labels['user_id'], max_labels['establecimiento'])))
        df['establecimiento'] = df['establecimiento'].fillna(df['n_establecimiento'])
        df['establecimiento'] = df['establecimiento'].fillna(max_labels.establecimiento.mode().values[0])
        df['dispositivo_n'] = df['dispositivo'].apply(ast.literal_eval)
        df[['año', 'marca', 'proovedor']] = pd.json_normalize(df['dispositivo_n'])
        df['género'] = encoder_genero.transform(df['género'])
        df['dia'] = encoder_dia.transform(df['fecha'].dt.day_name())
        encoded_establecimiento = encoder_establecimiento.transform(df[['establecimiento']])
        encoded_ciudad = encoder_ciudad.transform(df[['ciudad']])
        df['is_prime'] = df['is_prime'].map({True:1, False:0})
        
        df = pd.concat([df, pd.DataFrame(encoded_establecimiento, columns=categories_establecimiento[0]),
                        pd.DataFrame(encoded_ciudad, columns=categories_ciudad[0])], axis=1)
        df.drop(['user_id','transaction_id','dispositivo', 'status_txn','n_ciudad',
            'n_establecimiento', 'dispositivo_n', 'año', 'marca', 'proovedor', 'ciudad', 'establecimiento', 'fecha'], axis=1, inplace=True)
        df.fillna(0, inplace=True)
        return(df)