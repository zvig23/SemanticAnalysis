import re
import string
from typing import List

import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from thinc.util import to_categorical
from torch import nn


def make_classification_report(X_valid: np.ndarray, y_valid: np.ndarray, model: nn.Module):
    """
    Makes a classification report.

    Args:     X_valid, NumPy array: validation features
              Y_valid, NumPy array: validation target
              checkpoint_filepath:  file path to save epoch with max validation accuracy

    Returns:  classification report
    """

    label_names = ["anger", "fear", "joy", "love", "sadness"]

    y_prob = model.predict(X_valid)
    prediction_ints = np.zeros_like(y_prob)
    prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1

    print(classification_report(y_valid, prediction_ints, target_names=label_names, digits=4))


def remove_stopwords(input_text: str) -> str:
    """
    Removes stopwords from tweets based on Indonesian stop word list

    Args:     input_text, string:  tweet

    Returns:  string: cleaned tweet
    """

    stopwords_list: List[str] = stopwords.words('indonesian')

    words: List[str] = input_text.split()
    clean_words: List[str] = [word for word in words if word not in stopwords_list]

    return " ".join(clean_words)


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def custom_tokenizer(df, tk):
    """
    Tokenizes tweets using Keras' Tokenizer after stop word removal

    Args:     df, Pandas DataFrame: Contains all tweets and labels
              tk, Keras Tokenizer Object:  Creates tokens via fit_on_texts fn

    Returns:  df, Pandas DataFrame:  Updated DataFrame with tokenized tweets in new column
    """

    tokenizeList = []

    # remove stopwords
    df.tweet = df.tweet.apply(remove_stopwords)

    tweetList = df['tweet'].tolist()
    tk.fit_on_texts(df['tweet'])
    inv_map = {v: k for k, v in tk.word_index.items()}

    for sentence in tweetList:
        tweet = re.split('\s+', sentence)
        processed_seq = tk.texts_to_sequences(tweet)
        tokens = [inv_map[tok] for seq in processed_seq for tok in seq]
        tokenizeList.append(tokens)

    df['tokens'] = tokenizeList

    return df


def encode_labels(df):
    """
    One-hot encodes the emotion labels.

    Args:       df, Pandas DataFrame:  Contains all tweets and labels

    Returns:    y_oh, NumPy array:  one-hot encoded emotion labels
    """

    le = LabelEncoder()

    y = le.fit_transform(df['label'])

    return to_categorical(y)
