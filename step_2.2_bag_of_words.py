import pandas as pd
import re
from collections import Counter

df =  pd.read_csv('SMSSpamCollection.csv', sep= '\t', header=None, names=['label', 'message'])
df['label'] = df['label'].replace({'ham': 0, 'spam': 1})
df['message'] = df['message'].str.lower()
df['message'] = df['message'].str.replace('.', '', regex=False)
preprocessed_documents = df['message'].apply(lambda x: x.split()).tolist()
frequency_list = [Counter(doc) for doc in preprocessed_documents]
for i in range(5):
  print(f"Document {i+1}: {frequency_list[i]}")



