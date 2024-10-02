import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df =  pd.read_csv('SMSSpamCollection.csv', sep= '\t', header=None, names=['label', 'message'])

df['message'] = df['message'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

corpus = df['message'].tolist()
count_vector = CountVectorizer()
count_vector.fit(corpus)
doc_array = count_vector.transform(corpus).toarray()
features = count_vector.get_feature_names_out()
frequency_matrix = pd.DataFrame(doc_array, columns=features)

