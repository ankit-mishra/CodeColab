import tensorflow as tf
import pandas as pd

# Tensflow version == 2.3.0
## Define path data
COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
PATH_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(PATH, skipinitialspace=True, names = COLUMNS, index_col=False)
df_test = pd.read_csv(PATH_test,skiprows = 1, skipinitialspace=True, names = COLUMNS, index_col=False)

print(df_train.shape, df_test.shape)
print(df_train.dtypes) 

label = {'<=50K': 0,'>50K': 1}
df_train.label = [label[item] for item in df_train.label]
label_t = {'<=50K.': 0,'>50K.': 1}
df_test.label = [label_t[item] for item in df_test.label]

print(df_train["label"].value_counts())
### The model will be correct in atleast 70% of the case
print(df_test["label"].value_counts())
## Unbalanced label
print(df_train.dtypes)

## Add features to the bucket: 
### Define continuous list
CONTI_FEATURES  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']
### Define the categorical list
CATE_FEATURES = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native_country']

continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]

def print_transformation(feature = "age", continuous = True, size = 2): 
    #X = fc.numeric_column(feature)
    ## Create feature name
    feature_names = [
    feature]

    ## Create dict with the data
    d = dict(zip(feature_names, [df_train[feature]]))

    ## Convert age
    if continuous == True:
        c = tf.feature_column.numeric_column(feature)
        feature_columns = [c]
    else: 
        c = tf.feature_column.categorical_column_with_hash_bucket(feature, hash_bucket_size=size) 
        c_indicator = tf.feature_column.indicator_column(c)
        feature_columns = [c_indicator]
    
## Use input_layer to print the value
    #feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    #input_layer = tf.keras.layers.Input(tensor=feature_layer, name='features', dtype=tf.float32)
    input_layer = tf.compat.v1.feature_column.input_layer(
        features=d,
        feature_columns=feature_columns
        )

    ## Create lookup table
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(input_layer, zero)
    ## Return lookup tble
    indices = tf.where(where)
    values = tf.gather_nd(input_layer, indices)
    ## Initiate graph
    #sess = tf.compat.v1.Session()
    # TF 2.0 supports eager execution which means you don't have to explicitly create a session 
    
    ## Print value
    print(input_layer)

print_transformation(feature = "sex", continuous = False, size = 2)

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in CATE_FEATURES]

model = tf.estimator.LinearClassifier(
    n_classes = 2,
    model_dir="ongoing/train", 
    feature_columns = categorical_features + continuous_features)


FEATURES = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country']
LABEL= 'label'
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
       x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
       y = pd.Series(data_set[LABEL].values),
       batch_size=n_batch,   
       num_epochs=num_epochs,
       shuffle=shuffle)

model.train(input_fn=get_input_fn(df_train, 
                                      num_epochs=None,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)

model.evaluate(input_fn=get_input_fn(df_test, 
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)