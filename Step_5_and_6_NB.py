import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

df['sms_message'] = df['sms_message'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

X = df['sms_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

count_vector = CountVectorizer()

X_train_matrix = count_vector.fit_transform(X_train)
X_test_matrix = count_vector.transform(X_test)

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_matrix, y_train)

y_pred = naive_bayes.predict(X_test_matrix)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
