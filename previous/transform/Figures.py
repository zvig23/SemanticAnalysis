import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import transformers
from transformers import AutoTokenizer, TFBertModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

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
    LABEL2INDEX = {'love': 0, 'anger': 1, 'sadness': 2, 'happy': 3, 'fear': 4}  # Label string to index
    INDEX2LABEL = {0: 'love', 1: 'anger', 2: 'sadness', 3: 'happy', 4: 'fear'}  # Index to label string
    NUM_LABELS = 5  # Number of label
    df = pd.read_csv(tweets_filepath)  # Read tsv file with pandas
    # df.columns = ['text','sentiment'] # Rename the columns
    df.columns = ['Sentiment', 'Input']  # Rename the columns
    df['Sentiment'] = df['Sentiment'].apply(lambda lab: LABEL2INDEX[lab])  # Convert string label into index
    return df


def train_test_valid_SPLIT(df):
    train_70 = df.sample(frac=0.7)

    df_30 = df.drop(train_70.index)

    valid_15 = df_30.sample(frac=0.5)

    test_15 = df_30.drop(valid_15.index)

    return train_70, valid_15, test_15


def model_full(df_data_path, dense_1, dense_2, dropout, learning_rate, epoch, batch_size, DECAY, checkpoint_fpath,
               name_of_excel_classification_report, iteration_number):
    df = load_dataset(df_data_path)
    train_70, valid_15, test_15 = train_test_valid_SPLIT(df)

    y_train = to_categorical(train_70.Sentiment)
    y_valid = to_categorical(valid_15.Sentiment)
    y_test = to_categorical(test_15.Sentiment)

    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    bert = TFBertModel.from_pretrained("indolem/indobertweet-base-uncased", from_pt=True)

    # Tokenize the input (takes some time)
    # here tokenizer using from bert-base-cased
    x_train = tokenizer(
        text=train_70.Input.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    x_valid = tokenizer(
        text=valid_15.Input.tolist(),
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True)
    x_test = tokenizer(
        text=test_15.Input.tolist(),
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

    print(model.summary())
'''
    history = model.fit({'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']},
                        y_train,
                        epochs=NB_START_EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(
                            {'input_ids': x_valid['input_ids'], 'attention_mask': x_valid['attention_mask']}, y_valid),
                        callbacks=[model_checkpoint_callback],
                        shuffle=True)

    model.load_weights(checkpoint_fpath)
    # INDEX2LABEL = {0: 'love', 1: 'anger', 2: 'sadness', 3: 'happy', 4: 'fear'} # Index to label string
    label_names = ['love', 'anger', 'sadness', 'happy', 'fear']

    predicted_raw = model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})
    y_predicted = np.argmax(predicted_raw, axis=1)
    # y_true = y_test.Sentiment
    # y_true = y_test
    y_true = test_15.Sentiment
    # y_prob = model.predict(X_valid)
    # prediction_ints = np.zeros_like(y_prob)
    # prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    # prediction = np.where(prediction_ints==1)[1]

    report = classification_report(y_true, y_predicted, target_names=label_names, digits=4, output_dict=True)

    print(classification_report(y_true, y_predicted, target_names=label_names, digits=4))

    df = pd.DataFrame(report).transpose()

    df.to_csv(name_of_excel_classification_report)

    make_confusion_matrix(x_test, y_test, checkpoint_filepath,
                          '/remote_home/FIGURES/CONFUSION/Confusion' + '_Iteration' + str(iteration_number) + '.png')

    image_path_acc = '/remote_home/FIGURES/ACCURACY/Accuracy' + '_Iteration' + str(iteration_number) + '.png'
    eval_metric(history, 'accuracy', image_path_acc)
    image_path_loss = '/remote_home/FIGURES/LOSS/Loss' + '_Iteration' + str(iteration_number) + '.png'
    eval_metric(history, 'loss', image_path_loss)'''

    return history


def delete_saved_model(folder_list_path, file_list_path):
    # folder_list_path = ['/assets', '/variables']
    # file_list_path = ['keras_metadata.pb', 'saved_model.pb']

    time.sleep(15)

    for i in folder_list_path:
        # os.rmdir(i)
        shutil.rmtree(i)
        try:
            # os.rmdir(i)
            # os.unlike(i)
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


# def save_model_results():

def make_confusion_matrix(X_valid, y_valid, checkpoint_filepath, image_path):
    """
    Makes a confusion matrix.

    Args:     X_valid, NumPy array: validation features
              Y_valid, NumPy array: validation target
              checkpoint_filepath:  file path to save epoch with max validation accuracy

    Returns:  confusion matrix
    """

    model.load_weights(checkpoint_filepath)

    label_names = ['anger', 'fear', 'joy', 'love', 'sadness']

    y_prob = model.predict({'input_ids': X_valid['input_ids'], 'attention_mask': X_valid['attention_mask']})
    prediction_ints = np.zeros_like(y_prob)
    prediction_ints[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    prediction = np.where(prediction_ints == 1)[1]

    y_true = df_test.Sentiment

    # y_cat_valid_emb = np.where(y_true==1)[1]

    cf_matrix = confusion_matrix(prediction, y_true)

    cf_matrix_norm = cf_matrix / cf_matrix.astype(np.float).sum(axis=1, keepdims=True)

    cf_matrix_norm_round = np.around(cf_matrix_norm, decimals=2)

    df_cm = pd.DataFrame(cf_matrix_norm_round, columns=label_names, index=label_names)

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sns.heatmap(df_cm, cmap='Blues', annot=True)
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")
    plt.savefig(image_path)
    plt.show()


def eval_metric(history, metric_name, image_path):
    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Args:     history, Keras object: model training history
              metric_name, string: loss or accuracy

    Returns:  matplotlib chart:  line chart with epochs of x-axis and metric on y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, label='Train ' + metric_name)
    plt.plot(e, val_metric, label='Validation ' + metric_name)
    plt.title('IndoBERT TWEET ' + metric_name)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.xticks(np.arange(min(e) + 1, max(e) + 1, 2.0))
    plt.legend()
    plt.savefig(image_path)
    plt.show()

def DOE(DOE_RunNumber, DOE_dense_1, DOE_dense_2, DOE_dropout, DOE_learning_rate, DOE_EPOCH, DOE_batch_size, DOE_DECAY):
    df_data_path = 'Twitter_Emotion_Dataset.csv'


    for i in range(0, 10):

        checkpoint_file_path = '/remote_home/FIGURES/Model_' + str(DOE_RunNumber) + '_' + str(i)

        try:
            # count += 1
            os.mkdir(checkpoint_file_path)
        except:
            print("Model Directory already there!!!!")

        csv_name = '/remote_home/FIGURES/Run' + str(DOE_RunNumber) + '_Iteration' + str(i) + '.csv'

        start_time = time.time()
        m_history = model_full(df_data_path, dense_1=DOE_dense_1, dense_2=DOE_dense_2, dropout=DOE_dropout,
                               learning_rate=DOE_learning_rate, epoch=DOE_EPOCH,
                               batch_size=DOE_batch_size, DECAY=DOE_DECAY, checkpoint_fpath=checkpoint_file_path,
                               name_of_excel_classification_report=csv_name, iteration_number=i)

        make_confusion_matrix(x_test, y_test, checkpoint_filepath,'/remote_home/FIGURES/CONFUSION/Confusion' + '_Iteration' + str(i) + '.png')

        time_list.append((time.time() - start_time))

        delete_saved_model(folder_list_path, file_list_path)


if __name__ == '__main__':
    # Download Desired Data from Google Drive
    destination_list = ['1O7BsNT792ZlzX4NFzQauiahqL-vuqJb-']
    id_list = ['Twitter_Emotion_Dataset.csv']
    download_files(id_list, destination_list)


    # delete_saved_model(folder_list_path,file_list_path)

    count = 1
    time_list = []



    # Run 1
    DOE(DOE_RunNumber=1, DOE_dense_1=64, DOE_dense_2=15, DOE_dropout=.05, DOE_learning_rate=3e-5, DOE_EPOCH=20,
        DOE_batch_size=64, DOE_DECAY=.005)



