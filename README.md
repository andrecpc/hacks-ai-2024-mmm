#### Конкурсный проект команды Feature_88
#### Цифровой прорыв 2024 ЮФО, Моделирование продаж

Проект представляет собой веб приложение Streamlite с предобученной нейронной сетью, которая предсказывает продажи на 29 недель вперед.

###### Описание файлов:

 - training_final -- обучение модели с валидацией и предиктом.
 - requirements.txt -- зависимости окружения.
 - commands.txt -- все шаги команд по установке библиотек и запуску приложения.
 - app.py -- файл приложения (для запуска выполнить команду streamlit run app.py).
 - files -- различные вспомогательные файлы, сырые данные и файлы модели.
 - MIX_(2).ipynb -- Решение нейронной сетью на LSTM, но только на 1 признаке -- на истории продаж.

 ###### Суть подхода

Обучаем нейронную сеть на PyTorch на всех доступных признаках. На вход подаем признаки 4 недель, на выходе получаем последовательное предсказание продаж на 29 недель.
Модель простая, но и данных не много:
```
SalesForecast(
  (layer1): Linear(in_features=576, out_features=100, bias=True)
  (relu): ReLU()
  (batchnorm1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer2): Linear(in_features=100, out_features=50, bias=True)
  (batchnorm2): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (output_layer): Linear(in_features=50, out_features=29, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

В созданном приложении загружаем имеющиеся данные и получаем предсказание на 29 недель приватных данных.
Также с помощью линейной регресии находим коэффициенты признаков для оценки их флияния на целевую метрику.

[Видео демонстрация](https://drive.google.com/file/d/1q2_rvG7_FH5F0aHCG2S1r0F3iFj-C6tt/view?usp=sharing)

Скриншот приложения
![Скриншот приложения](https://github.com/andrecpc/hacks-ai-2024-mmm/blob/main/demo.png)
