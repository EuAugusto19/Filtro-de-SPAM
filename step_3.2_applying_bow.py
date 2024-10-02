import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

X = df['sms_message'] 
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

count_vector = CountVectorizer()
X_train_matrix = count_vector.fit_transform(X_train)
X_test_matrix = count_vector.transform(X_test)

print(f"Matriz de treino (X_train_matrix): {X_train_matrix.shape}")
print(f"Matriz de teste (X_test_matrix): {X_test_matrix.shape}")
