# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk import SnowballStemmer


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
# Scikit-learn metrics
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report

#read dataset
real_data = pd.read_csv('/content/drive/MyDrive/projects/True.csv')
fake_data = pd.read_csv('/content/drive/MyDrive/projects/Fake.csv')

real_data.head()
fake_data.head()

#add column
real_data['target'] = 1
fake_data['target'] = 0

real_data.tail()
fake_data.tail()


#Merging the 2 datasets
data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
data.head()

#checking missing values
data.isnull().sum()

#count the subjects
print(data.subject.value_counts())

#merging
#deleting unwanted columns
data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']
data.head()

# download some modules from nltk library
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#list the stopword into a object
stop_words_list = stopwords.words("english")

#text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r'""'," ",text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://S+|www.\.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if not word in stop_words_list]
    words = [re.sub(r"(.)\1{1,}", r"\1\1", word) for word in words]
    words = [word.strip() for word in words if len(word.strip()) > 1]
    text = " ".join(words)
    return text

data["text"] = data["text"].apply(preprocess_text)

#Stemming
stemmer = SnowballStemmer("english")

def stemming(text):
    stemmed_text = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemmed_text += stem
        stemmed_text += " "

    stemmed_text = stemmed_text.strip()
    return stemmed_text

data["text"] = data["text"].apply(stemming)

#lemmatization
def lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)

    for word in text:

         lemma = nltk.WordNetLemmatizer()
         word = lemma.lemmatize(word)
         final_text.append(word)
    return " ".join(final_text)

data['text']=data['text'].apply(lemmatization)

data.head()

# texts = ' '.join(data['text'])

# string = texts.split(" ")

#split traning set and testing set
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state=0)

max_features = 10000
maxlen = 300

tf.keras.utils.pad_sequences

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = tf.keras.utils.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = tf.keras.utils.pad_sequences(tokenized_test, maxlen=maxlen)

batch_size = 64
embed_size = 100

model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
#LSTM
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25,kernel_regularizer=l2(0.01)))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 64 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)# Monitor the validation loss, and stop training if the validation loss stops improving

model.summary()

history = model.fit(X_train, y_train,
                     validation_split=0.2, epochs=50,
                       batch_size=batch_size, shuffle=True,
                       verbose = 1,callbacks=[early_stopping]  # EarlyStopping callback
                       )

print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

model.save_weights('basic_model.h5')

model.load_weights('basic_model.h5')

#Predict
y_prediction = model.predict(X_test)

# labelling (0 - 0.5) to be 0 and (0.6 - 1) to be 1
y_predictions_binary = [1 if pred > 0.5 else 0 for pred in y_prediction]

y_predictions_binary

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test and y_prediction are your true labels and predicted labels, respectively
y_test = np.array(y_test)
# y_prediction = np.argmax(y_prediction, axis=1)

# Convert to integer labels
y_test = y_test.astype(int)
# y_prediction = y_prediction.astype(int)

# Check unique values
print(np.unique(y_test), np.unique(y_prediction))

# Create and plot confusion matrix
conf_mat = confusion_matrix(y_predictions_binary,y_test)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(conf_mat)

target_names=["class_0","class_1"]
print(classification_report(y_test, y_predictions_binary, target_names=target_names))

svm_class= svm.SVC()
svm_class.fit(X_train,y_train)
y_pred=svm_class.predict(X_test)
Accuracy_Svm=round((metrics.accuracy_score(y_test, y_pred)*100),2)
print('Accuracy: ',Accuracy_Svm,"%")

conf_mat = confusion_matrix(y_pred,y_test)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(conf_mat)

target_names=["class_0","class_1"]
print(classification_report(y_test, y_pred, target_names=target_names))

x=['LSTM',"SVM"]
y=[88.29,73.18]
plt.bar(x,y,color='yellow')
plt.xlabel("Model")
plt.ylabel("Accuracy in %")
plt.title('Model Accuracy Comparison')
plt.show()