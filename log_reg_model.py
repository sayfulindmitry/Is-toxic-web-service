#Импортируем все нужные библиотеки

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.externals import joblib


#Сохраняем данные, заполняем пропущенные значения

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

df = train_df.copy()
df.dropna(axis=1, inplace=True)


#Векторизуем комментарии, поскольку наша модель будет работать с числами.
#Удобной статистикой для этого является Tf-idf

Vectorize = TfidfVectorizer()

X = Vectorize.fit_transform(df["comment_text"])

test_X = Vectorize.transform(test_df["comment_text"])

#Объявим целевую переменную через следующий критерий:

y = np.where(train_df['target'] >= 0.5, 1, 0)

#Для обучения модели нам нужно разделить датасет (2/3 и 1/3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

#Теперь мы готовы для создания и обучения модели (будем использовать логистическую регрессию)
lr = LogisticRegression(C=5, random_state=42, solver='sag', max_iter=1000, n_jobs=-1)
lr.fit(X_train, y_train)

#Проверим score на кросс-валидации
cv_accuracy = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')

print(cv_accuracy) #[0.94607755 0.94436349 0.94531875 0.94375522 0.94326619]
print(cv_accuracy.mean()) #0.9445562409622392

#Модель обучилась отлично!

#Осталось сохранить векторизация и саму модель для внедрения ее в веб-сервис

joblib.dump(lr, 'log_reg_model.pkl')
joblib.dump(Vectorize, 'vectorize.pkl')

