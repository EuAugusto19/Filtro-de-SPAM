import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

X = df['sms_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Número de linhas em X_train: {X_train.shape[0]}")
print(f"Número de linhas em X_test: {X_test.shape[0]}")
print(f"Número de linhas em y_train: {y_train.shape[0]}")
print(f"Número de linhas em y_test: {y_test.shape[0]}")

