import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import transformers
from transformers import AutoTokenizer, TFBertModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

import torch

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

from openpyxl import load_workbook
import xlsxwriter

import shutil

import keras
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import requests
import os
import time

import keras.backend as K

def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
def download_files(id_list, destination_list):
    for (a, b) in zip(id_list, destination_list):
        download_file_from_google_drive(b, a)
def make_classification_report(df_test, x_test, checkpoint_fpath, five_emotions=True):
    """
    Makes a classification report.

    Args:     X_valid, NumPy array: validation features
              Y_valid, NumPy array: validation target
              checkpoint_fpath:  file path to save epoch with max validation accuracy

    Returns:  classification report
    """
    # checkpoint_fpath = '/content/drive/MyDrive/Thesis_V2/DOE'

    model.load_weights(checkpoint_fpath)

    if five_emotions == True:
        # INDEX2LABEL = {0: 'love', 1: 'anger', 2: 'sadness', 3: 'happy', 4: 'fear'} # Index to label string
        label_names = ['love', 'anger', 'sadness', 'happy', 'fear']
    else:
        label_names = ['positive', 'negative']

    predicted_raw = model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})

    y_predicted = np.argmax(predicted_raw, axis=1)
    y_true = df_test.Sentiment

    # y_prob = model.predict(X_valid)
    # prediction_ints = np.zeros_like(y_prob)
    # prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    # prediction = np.where(prediction_ints==1)[1]

    report = classification_report(y_true, y_predicted, target_names=label_names, digits=4)
    print(report)
    return report
def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index=False)


def load_dataset(tweets_filepath):
    LABEL2INDEX = {'Love': 0, 'Anger': 1, 'Sadness': 2, 'Joy': 3, 'Fear': 4}  # Label string to index
    INDEX2LABEL = {0: 'Love', 1: 'Anger', 2: 'Sadness', 3: 'Joy', 4: 'Fear'}  # Index to label string
    NUM_LABELS = 5  # Number of label
    df = pd.read_csv(tweets_filepath)  # Read tsv file with pandas
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(['ReachSize'], axis=1, errors='ignore')
    print("____HERE_____")
    print(df.head())
    print("____HERE_____")
    # df.columns = ['text','sentiment'] # Rename the columns
    df.columns = ['Input', 'Sentiment']  # Rename the columns
    print("____HERE2_____")
    print(df.head())
    print("____HERE2_____")
    df['Sentiment'] = df['Sentiment'].apply(lambda lab: LABEL2INDEX[lab])  # Convert string label into index
    return df

def load_dataset_original_5(dfPath):
  # Turn labels into numbers for data set with 5 emotions
  LABEL2INDEX = {'love': 0, 'anger': 1, 'sadness': 2, 'happy': 3, 'fear': 4} # Label string to index
  INDEX2LABEL = {0: 'love', 1: 'anger', 2: 'sadness', 3: 'happy', 4: 'fear'} # Index to label string
  NUM_LABELS = 5 # Number of label
  df = pd.read_csv(dfPath) # Read tsv file with pandas
  #df.columns = ['text','sentiment'] # Rename the columns
  df.columns = ['Sentiment','Input'] # Rename the columns
  df['Sentiment'] = df['Sentiment'].apply(lambda lab: LABEL2INDEX[lab]) # Convert string label into index
  return df

def train_test_valid_SPLIT(df):
    train_70 = df.sample(frac=0.7)

    df_30 = df.drop(train_70.index)

    valid_15 = df_30.sample(frac=0.5)

    test_15 = df_30.drop(valid_15.index)

    return train_70, valid_15, test_15


def model_full(df_data_path, dense_1, dense_2, dropout, learning_rate, epoch, batch_size, DECAY, checkpoint_fpath,
               name_of_excel_classification_report, year):


    train = load_dataset('/remote_home/SSEmoji/BalancedFinal'+ str(year) + '.csv')
    test = load_dataset_original_5('Twitter_Emotion_Dataset.csv')

    y_train = to_categorical(train.Sentiment)

    y_test =  to_categorical(test.Sentiment)
    
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    bert = TFBertModel.from_pretrained("indolem/indobertweet-base-uncased", from_pt=True)
    
    # Tokenize the input (takes some time)
    # here tokenizer using from bert-base-cased
    x_train = tokenizer(
        text=train.Input.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    x_test = tokenizer(
        text=test.Input.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)

    input_ids = x_train['input_ids']
    attention_mask = x_train['attention_mask']

    max_len = 70
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    embeddings = bert(input_ids, attention_mask=input_mask)[0]
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(dense_1, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = Dense(dense_2, activation='relu')(out)
    y = Dense(5, activation='sigmoid')(out)
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True

    LEARNING_RATE = learning_rate

    # 3e-6
    NB_START_EPOCHS = epoch
    BATCH_SIZE = batch_size

    optimizer = Adam(
        learning_rate=learning_rate,  # this learning rate is for bert model , taken from huggingface website
        epsilon=1e-08,
        decay=DECAY,
        clipnorm=1.0)

    # Set loss and metrics
    loss = CategoricalCrossentropy(from_logits=True)
    metric = CategoricalAccuracy('balanced_accuracy')
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metric)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_fpath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])

    history = model.fit({'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']},
                        y_train,
                        epochs=NB_START_EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.2,
                        callbacks=[model_checkpoint_callback],
                        shuffle=True)

    model.load_weights(checkpoint_fpath)
    # INDEX2LABEL = {0: 'love', 1: 'anger', 2: 'sadness', 3: 'happy', 4: 'fear'} # Index to label string
    label_names = ['love', 'anger', 'sadness', 'happy', 'fear']

    predicted_raw = model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})
    y_predicted = np.argmax(predicted_raw, axis=1)
    # y_true = y_test.Sentiment
    # y_true = y_test
    y_true = test.Sentiment
    # y_prob = model.predict(X_valid)
    # prediction_ints = np.zeros_like(y_prob)
    # prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    # prediction = np.where(prediction_ints==1)[1]

    report = classification_report(y_true, y_predicted, target_names=label_names, digits=4, output_dict=True)

    print(classification_report(y_true, y_predicted, target_names=label_names, digits=4))

    df = pd.DataFrame(report).transpose()
    
    df.to_csv(name_of_excel_classification_report)

    return history


def delete_saved_model(folder_list_path, file_list_path):
    # folder_list_path = ['/assets', '/variables']
    # file_list_path = ['keras_metadata.pb', 'saved_model.pb']

    time.sleep(15)

    for i in folder_list_path:
        #os.rmdir(i)
        shutil.rmtree(i)
        try:
            #os.rmdir(i)
            #os.unlike(i)
            shutil.rmtree(i)
            # % rm - rf '/content/drive/MyDrive/Thesis_V2/DOE/assets'
        except:
            print("FOLDER NOT FOUND: ", i)

        time.sleep(15)

    time.sleep(15)

    for j in file_list_path:
        try:
            # % rm - rf '/content/drive/MyDrive/Thesis_V2/DOE/variables'
            os.remove(j)
        except:
            print("FILE NOT FOUND: ", j)

    time.sleep(15)
def make_confusion_matrix(X_valid, y_valid, checkpoint_fpath):
    """
    Makes a confusion matrix.

    Args:     X_valid, NumPy array: validation features
              Y_valid, NumPy array: validation target
              checkpoint_fpath:  file path to save epoch with max validation accuracy

    Returns:  confusion matrix
    """

    model.load_weights(checkpoint_fpath)

    label_names = ['anger', 'fear', 'joy', 'love', 'sadness']

    y_prob = model.predict(X_valid)
    prediction_ints = np.zeros_like(y_prob)
    prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    prediction = np.where(prediction_ints == 1)[1]

    y_cat_valid_emb = np.where(y_valid == 1)[1]

    cf_matrix = confusion_matrix(prediction, y_cat_valid_emb)

    cf_matrix_norm = cf_matrix / cf_matrix.astype(np.float).sum(axis=1, keepdims=True)

    cf_matrix_norm_round = np.around(cf_matrix_norm, decimals=2)

    df_cm = pd.DataFrame(cf_matrix_norm_round, columns=label_names, index=label_names)

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sns.heatmap(df_cm, cmap='Blues', annot=True)
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")
    plt.show()

def DOE(DOE_RunNumber, DOE_dense_1, DOE_dense_2, DOE_dropout, DOE_learning_rate, DOE_EPOCH, DOE_batch_size, DOE_DECAY):
    df_data_path = 'Twitter_Emotion_Dataset.csv'

    size_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]

    for size in size_list:
        for split in range(1,5+1):
            for iteration in range(0,10):
                # Create Checkpoint Path
                checkpoint_file_path = '/remote_home/SIZE_SPLIT/RESULTS/TestSize_' + str(size) + '_Split_' + str(split) + '_Iteration_' + str(iteration)

                # Create Model Directory
                try:
                    os.mkdir(checkpoint_file_path)
                except:
                    print("Model Directory already there!!!!")

                csv_name = '/remote_home/SIZE_SPLIT/CSV/TestSize_' + str(size) + '_Split_' + str(split) + '_Iteration_' + str(iteration) + '.csv'

                m_history = model_full(df_data_path, dense_1=DOE_dense_1, dense_2=DOE_dense_2, dropout=DOE_dropout,
                                       learning_rate=DOE_learning_rate, epoch=DOE_EPOCH,
                                       batch_size=DOE_batch_size, DECAY=DOE_DECAY,
                                       checkpoint_fpath=checkpoint_file_path,
                                       name_of_excel_classification_report=csv_name, SizeSplit = size, DataSplit = split)

def EMOJI(DataSet, Year, DOE_dense_1, DOE_dense_2, DOE_dropout, DOE_learning_rate, DOE_EPOCH, DOE_batch_size, DOE_DECAY):
    df_data_path = 'Twitter_Emotion_Dataset.csv'

    for i in range(0,30):
        checkpoint_file_path = '/remote_home/SSEmoji/RESULTS/Year' + str(Year) + '_Iteration_' + str(i)
        try:
            os.mkdir(checkpoint_file_path)
        except:
            print("Model Directory already there!!!!")

        csv_name = '/remote_home/SSEmoji/CSV/Year' + str(Year) + '_Iteration_' + str(i) + '.csv'

        m_history = model_full(df_data_path, dense_1=DOE_dense_1, dense_2=DOE_dense_2, dropout=DOE_dropout,
                               learning_rate=DOE_learning_rate, epoch=DOE_EPOCH,
                               batch_size=DOE_batch_size, DECAY=DOE_DECAY,
                               checkpoint_fpath=checkpoint_file_path,
                               name_of_excel_classification_report=csv_name, year=Year)


run_code = True

if run_code == True:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Download Desired Data from Google Drive
    destination_list = ['1O7BsNT792ZlzX4NFzQauiahqL-vuqJb-']
    id_list = ['Twitter_Emotion_Dataset.csv']
    download_files(id_list, destination_list)

    #EMOJI RUN - 2018
    Data2018 = '/remote_home/SSEmoji/BalancedFinal2018.csv'
    EMOJI(DataSet = Data2018, Year = 2018, DOE_dense_1=64, DOE_dense_2=15, DOE_dropout=.05, DOE_learning_rate=3e-5, DOE_EPOCH=25,
        DOE_batch_size=40, DOE_DECAY=.005)

    #EMOJI RUN - 2022
    Data2022 = '/remote_home/SSEmoji/BalancedFinal2022.csv'
    EMOJI(DataSet = Data2022, Year = 2022, DOE_dense_1=64, DOE_dense_2=15, DOE_dropout=.05, DOE_learning_rate=3e-5, DOE_EPOCH=25,
        DOE_batch_size=40, DOE_DECAY=.005)