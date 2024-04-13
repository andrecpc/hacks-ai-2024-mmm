import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from pickle import load
import pickle


def load_model():

    class SalesForecast(nn.Module):
        def __init__(self, input_dim, output_dim, n_future, n_past):
            super(SalesForecast, self).__init__()
            self.layer1 = nn.Linear(input_dim*n_past, 100)
            self.relu = nn.ReLU()
            self.batchnorm1 = nn.BatchNorm1d(100)
            self.layer2 = nn.Linear(100, 50)
            self.batchnorm2 = nn.BatchNorm1d(50)
            self.output_layer = nn.Linear(50, n_future)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.batchnorm1(self.layer1(x)))
            x = self.relu(self.batchnorm2(self.layer2(x)))
            x = self.output_layer(x)
            return x

    input_dim = 144
    output_dim = 29
    n_past = 4
    model = SalesForecast(input_dim, output_dim, output_dim, n_past)
    model.load_state_dict(torch.load('files/best_model.pth'))

    return model


def make_predictions(model, data):

    data['Начало нед'] = pd.to_datetime(data['Начало нед'])
    data = data[data['Начало нед']<pd.to_datetime('2023-09-04')]
    data.set_index('Начало нед', inplace=True)

    data.drop(columns=['год', 'неделя', 'Unnamed: 147'], inplace=True)

    inp = data[-4:].copy()
    # print(inp.head())
    scaler = load(open('files/scaler.pkl', 'rb'))
    # scaler = StandardScaler()
    # features = data.drop(columns=['Продажи, рубли', 'Продажи, упаковки']).columns.to_list()[0:12] + ['Запросы Wordstat']
    features = data.drop(columns=['Продажи, рубли', 'Продажи, упаковки']).columns.to_list()
    features += ['Продажи, рубли', 'Продажи, упаковки']
    # scaler.fit(data[features])
    inp = scaler.transform(inp[features])
    # print(inp)
    out = pd.DataFrame([0]*29, columns=['Продажи, рубли'])

    inp_seq, out_seq = np.array([inp]), np.array([out])

    inp_seq_tensor = torch.tensor(inp_seq, dtype=torch.float32)
    out_seq_tensor = torch.tensor(out_seq, dtype=torch.float32)

    batch_size = 16

    inp_out_dataset = TensorDataset(inp_seq_tensor, out_seq_tensor)
    inp_out_loader = DataLoader(inp_out_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in inp_out_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.view(targets.size(0), -1).numpy())


    target_scaler = load(open('files/target_scaler.pkl', 'rb'))
    predictions_inv = target_scaler.inverse_transform(predictions)

    return predictions_inv.reshape(predictions_inv.shape[1])

model = load_model()

def plot_predictions(predictions, col, size=(10, 6), dpi=150):

    df = pd.read_csv('files/sample_submission.csv')
    plt.figure(figsize=size, dpi=dpi)
    plt.plot(df['week'], predictions)
    plt.title('Predictions')
    plt.xlabel('Index')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    plt.xticks(rotation=90)
    col.pyplot(plt) 

st.title('Neural Network Predictions')

uploaded_file = st.file_uploader("Choose a file", type=['xlsx'])
if uploaded_file:

    data = pd.read_excel(uploaded_file, header=5)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.replace(' ', 0, inplace=True)
    data.fillna(0, inplace=True)
    data.fillna(0, inplace=True)
    data.fillna(0, inplace=True)
    for col in data.columns:
        data.rename(columns={col: col.replace('(', ' ').replace(')', ' ').replace('\n', ' ').replace('.', ' ')}, inplace=True)
    
    data_df = data.copy()

    st.write("Input Data:")
    data = st.dataframe(data)

    predictions = make_predictions(model, data_df)


    col1, col2 = st.columns([1, 3])

    with col1:
        st.write("Predictions:")
        df = pd.read_csv('files/sample_submission.csv')
        st.dataframe(pd.concat([df['week'], pd.DataFrame(predictions)], axis=1))

    with col2:
        plot_predictions(predictions, col2, size=(10, 3))


    st.write("Feature Importances:")
    coef_df = pd.read_csv('files/coef_df.csv')
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df)
