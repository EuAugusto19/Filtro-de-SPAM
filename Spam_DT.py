import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

df['sms_message'] = df['sms_message'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

X = df['sms_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

count_vector = CountVectorizer()

X_train_cv = count_vector.fit_transform(X_train)
X_test_cv = count_vector.transform(X_test)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train_cv, y_train)

y_pred_dt = dt_classifier.predict(X_test_cv)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f"Acurácia (Árvore de Decisão): {accuracy_dt:.2f}")
print(f"Precisão (Árvore de Decisão): {precision_dt:.2f}")
print(f"Recall (Árvore de Decisão): {recall_dt:.2f}")
print(f"F1-score (Árvore de Decisão): {f1_dt:.2f}")