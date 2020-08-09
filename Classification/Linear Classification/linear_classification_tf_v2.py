# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# Tensflow version == 2.3.0
## Define path data
COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']
TRAIN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(TRAIN_DATA_URL, skipinitialspace=True, names = COLUMNS, index_col=False)
df_test = pd.read_csv(TEST_DATA_URL,skiprows = 1, skipinitialspace=True, names = COLUMNS, index_col=False)

print(df_train.head())
print('********************************')
print(df_test.head())
print('********************************')
label = {'<=50K': 0,'>50K': 1}
df_train.label = [label[item] for item in df_train.label]
label_t = {'<=50K.': 0,'>50K.': 1}
df_test.label = [label_t[item] for item in df_test.label]

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

df_train_1 = np.asarray(df_train)
model.fit(df_train_1, df_train.label, epochs=10)

test_loss, test_acc = model.evaluate(df_test,  df_test.label, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(df_test)