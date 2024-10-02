import pandas as pd
df = pd.read_csv('SMSSpamCollection.csv', sep= '\t', header=None, names=['label', 'message'])
print(df.head())