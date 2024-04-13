# -*- coding: utf-8 -*-
# Импорты

import warnings
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import pickle

pd.set_option('display.max_columns', None)
warnings.simplefilter('ignore')
random.seed(42)
np.random.seed(42)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

## Загрузка данных и предобработка

data = pd.read_excel('files/train.xlsx', header=5)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.replace(' ', 0, inplace=True)
data.fillna(0, inplace=True)
data.fillna(0, inplace=True)
data.fillna(0, inplace=True)

for col in data.columns:
    data.rename(columns={col: col.replace('(', ' ').replace(')', ' ').replace('\n', ' ').replace('.', ' ')}, inplace=True)

data['Начало нед'] = pd.to_datetime(data['Начало нед'])
data = data[data['Начало нед']<pd.to_datetime('2023-09-04')]
data.set_index('Начало нед', inplace=True)

data.drop(columns=['год', 'неделя', 'Unnamed: 147'], inplace=True)
# data[['Продажи, рубли', 'Продажи, упаковки']] = 0, 0

split_size = int(data.shape[0] / 4)
train_split, val_split = split_size*2, split_size*3
train = data[0:train_split]
val = data[train_split:val_split]
test = data[val_split:]

data

# Подготовка датасетов

y_train_true = train[['Продажи, рубли']]
y_val_true = val[['Продажи, рубли']]
y_test_true = test[['Продажи, рубли']]

# train['Продажи, рубли'] = 0
# val['Продажи, рубли'] = 0
# test['Продажи, рубли'] = 0

# train['Продажи, упаковки'] = 0
# val['Продажи, упаковки'] = 0
# test['Продажи, упаковки'] = 0

scaler = StandardScaler()
# features = data.drop(columns=['Продажи, рубли', 'Продажи, упаковки']).columns.to_list()[0:12] + ['Запросы Wordstat']
features = data.drop(columns=['Продажи, рубли', 'Продажи, упаковки']).columns.to_list()
features += ['Продажи, рубли', 'Продажи, упаковки']
scaler.fit(data[features])
print(features)

X_train = scaler.transform(train[features])
X_val = scaler.transform(val[features])
X_test = scaler.transform(test[features])
y_train = y_train_true.copy()
y_val = y_val_true.copy()
y_test = y_test_true.copy()

target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train['Продажи, рубли'] = target_scaler.fit_transform(y_train)
y_val['Продажи, рубли'] = target_scaler.transform(y_val)
y_test['Продажи, рубли'] = target_scaler.transform(y_test)

n_past = 4 # Сколько недель прошлого берем
n_future = 29 # Горизонт предикта
n_features = len(features)

# Нарезаем пары по n_past недель прошлого с фичами и по n_future недель с таргетом
def create_sequences(input_data, target_data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(input_data) - n_future +1):
        X.append(input_data[i - n_past:i])
        y.append(target_data[i:i + n_future])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, n_past, n_future)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, n_past, n_future)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, n_past, n_future)

X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_seq_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_seq_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
X_val_seq_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_seq_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

batch_size = 16

train_dataset = TensorDataset(X_train_seq_tensor, y_train_seq_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_seq_tensor, y_val_seq_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_seq_tensor, y_test_seq_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dump(scaler, open('files/scaler.pkl', 'wb'))
dump(target_scaler, open('files/target_scaler.pkl', 'wb'))

# Объявление модели

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

input_dim = len(features)
output_dim = n_future
model = SalesForecast(input_dim, output_dim, output_dim, n_past)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Обучение

num_epochs = 50
best_val_loss = float('inf')
train_losses, val_losses = [], []
scheduler_patience = 3
scheduler_break = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(targets.size(0), -1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, targets.view(targets.size(0), -1))
            total_val_loss += val_loss.item()
    val_loss = total_val_loss / len(val_loader)
    val_losses.append(val_loss)

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'files/best_model.pth')
        counter = 0
    else:
        counter += 1

    if counter >= scheduler_patience:
        model.load_state_dict(torch.load('files/best_model.pth'))
        # model = model.to(device)
        scheduler.step()
        print('new LR:', optimizer.param_groups[0]["lr"])
        counter = 0
        scheduler_break -= 1

    if scheduler_break < 0:
        break

print("Training complete.")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load('files/best_model.pth'))
# model = model.to(device)

# Оценка на валидации

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(targets.view(targets.size(0), -1).numpy())

# Inverse transform the predictions and actuals
predictions_inv = target_scaler.inverse_transform(predictions)
actuals_inv = target_scaler.inverse_transform(actuals)

rmse = np.sqrt(np.mean((predictions_inv - actuals_inv)**2))
mape = np.mean(np.abs((actuals_inv - predictions_inv) / actuals_inv)) * 100

print('RMSE и MAPE val:', rmse, mape)

# Оценка на тесте

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(targets.view(targets.size(0), -1).numpy())

# Inverse transform the predictions and actuals
predictions_inv = target_scaler.inverse_transform(predictions)
actuals_inv = target_scaler.inverse_transform(actuals)

rmse = np.sqrt(np.mean((predictions_inv - actuals_inv)**2))
mape = np.mean(np.abs((actuals_inv - predictions_inv) / actuals_inv)) * 100

print('RMSE и MAPE test:', rmse, mape)

# Пример предикта для как бы новых данных

x = 2 # Просто сдвиг в данных с конца (можно любое число, с которым не вылезаем за пределы теста или валидации)
inp = data[-n_past-n_future-x:-n_future-x]
out = data[-n_future-x:-x][['Продажи, рубли']]

inp

out

out_true = out[['Продажи, рубли']].copy()
inp = scaler.transform(inp[features])
out = out_true.copy()
out['Продажи, рубли'] = target_scaler.transform(out)

inp_seq, out_seq = np.array([inp]), np.array([out])

inp_seq_tensor = torch.tensor(inp_seq, dtype=torch.float32)
out_seq_tensor = torch.tensor(out_seq, dtype=torch.float32)

inp_out_dataset = TensorDataset(inp_seq_tensor, out_seq_tensor)
inp_out_loader = DataLoader(inp_out_dataset, batch_size=batch_size, shuffle=False)

with open('files/features.pkl', 'wb') as f:
    pickle.dump(features, f)
print(len(features))

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in inp_out_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(targets.view(targets.size(0), -1).numpy())

# Inverse transform the predictions and actuals
predictions_inv = target_scaler.inverse_transform(predictions)
actuals_inv = target_scaler.inverse_transform(actuals)

rmse = np.sqrt(np.mean((predictions_inv - actuals_inv)**2))
mape = np.mean(np.abs((actuals_inv - predictions_inv) / actuals_inv)) * 100

print('RMSE и MAPE inp_out:', rmse, mape)

plt.plot(list(range(out_true.shape[0])), out_true['Продажи, рубли'].values, label='true_target')
plt.plot(list(range(out_true.shape[0])), predictions_inv.reshape(predictions_inv.shape[1]), label='pred_target')
plt.legend()
plt.show()







# Предикт таргета для sample submission

inp = data[-n_past:]
out = data[-n_future:][['Продажи, рубли']]
out['Продажи, рубли'] = 0
out.reset_index(drop=True, inplace=True)


out_true = out[['Продажи, рубли']].copy()
inp = scaler.transform(inp[features])
out = out_true.copy()
out['Продажи, рубли'] = target_scaler.transform(out)

inp_seq, out_seq = np.array([inp]), np.array([out])

inp_seq_tensor = torch.tensor(inp_seq, dtype=torch.float32)
out_seq_tensor = torch.tensor(out_seq, dtype=torch.float32)

inp_out_dataset = TensorDataset(inp_seq_tensor, out_seq_tensor)
inp_out_loader = DataLoader(inp_out_dataset, batch_size=batch_size, shuffle=False)

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in inp_out_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(targets.view(targets.size(0), -1).numpy())

# Inverse transform the predictions and actuals
predictions_inv = target_scaler.inverse_transform(predictions)

plt.plot(list(range(out_true.shape[0])), predictions_inv.reshape(predictions_inv.shape[1]), label='pred_target')
plt.legend()
plt.show()

subm = pd.read_csv('files/sample_submission.csv')
subm['revenue'] = predictions_inv.reshape(predictions_inv.shape[1])
subm.to_csv('files/submit.csv', index=False)
subm


#### linreg

from sklearn.linear_model import LinearRegression

x = data.drop(columns=['Продажи, рубли', 'Продажи, упаковки'])
y = data['Продажи, рубли']
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(list(range(len(y))), y, color='blue', label='Original data')
plt.plot(list(range(len(y))), y_pred, color='red', label='Fitted line')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

feature_names = x.columns
coefficients = model.coef_

coef_df = pd.DataFrame(coefficients, index=feature_names, columns=['Coefficient'])

plt.figure(figsize=(10, 6))
coef_df['Coefficient'][0:25].plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(True)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.show()

coef_df.to_csv('files/coef_df.csv', index=True)