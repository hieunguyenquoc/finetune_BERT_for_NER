import pandas as pd

def load_data_NER():
    df = pd.read_csv("ner_datasetreference.csv",encoding="unicode_escape")

    df = df[:10000]
    df = df.fillna(method="ffill")
    # let's create a new column called "sentence" which groups the words by sentence
    df['sentence'] = df[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence
    df['word_labels'] = df[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
    
    # label2id = {k: v for v, k in enumerate(df["Tag"].unique())}
    id2label = {v: k for v, k in enumerate(df["Tag"].unique())}

    return id2label