# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:51:26 2024

@author: eboscolo
"""


#import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional, Attention,GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers #, regularizers, constraints
#from gensim.models import KeyedVectors
#import gensim.downloader
#from keras.layers import   SpatialDropout1D, Bidirectional, Attention,GlobalAveragePooling1D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#from tensorflow.keras.utils import get_custom_objects
import pickle
import sys
#print(list(gensim.downloader.info()['models'].keys()))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# For reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

###############################################################################
########### INIT VARS !! ######################################################
choice = 'eval_performance' # enable_train' # 'eval_performance' 
model_type = 'multi_h_attention' #'single_h_attention'#'no_attention' #'simple_attention' ' bahdanau_attention
n_head = 3 # head for multi head attention
embedding_size = 300 ## depends on glove embedding model
target_count= 6 # number of classes
learning_rate = 0.001  # lr
ep = 10 ## epochs  
path = "C:/Users/Enrico/Desktop/Progetti/21 ATTENTION iNVESTIGATION/"
### embedding file
#EMBEDDING_FILE =  '.../glove.6B.200d.txt'
EMBEDDING_FILE =  'glove.840B.300d.txt' ## load this from glove site
###########################################################################

################################################################################

# Learning Rate Scheduler ###
def lr_schedule(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)  # Reduce the learning rate
# 'Clean' Sentences #### 
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

### Attention classes #########################################################
#### this is a simple dot implementation of Attention #########################
class Simple_DotProductAttention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(Simple_DotProductAttention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # Initialize weights using glorot uniform initializer
        self.initializer = initializers.GlorotUniform()
        
        # Shape of weights should match the last dimension of the input this is the embedding dim
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.initializer,
            trainable=True,
            name='attention_weights'
        )
        
        # Shape of bias should match the sequence length (SL)
        if input_shape[1] is not None:  # Ensuring that sequence length is defined
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer=self.initializer,
                trainable=True,
                name='attention_bias'
            )
        else:
            self.b = None

    def call(self, inputs):
        # Compute attention weights
        x = inputs
        x = tf.cast(x, dtype=tf.float32)  # Ensuring tensor type is float32
        eij = tf.squeeze(tf.matmul(x, tf.expand_dims(self.W, axis=-1)), axis=-1)  # [N, SL, 1]
        
        if self.b is not None:
            eij += self.b
        
        eij = tf.tanh(eij)
        
        # Compute attention scores using softmax
        a = tf.nn.softmax(eij, axis=1)  # [N, SL]
        a = tf.expand_dims(a, axis=-1)  # [N, SL, 1]
        
        # Compute weighted sum
        weighted_input = x * a  # Broadcasting Mechanism: [N, SL, EM] * [N, SL, 1] = [N, SL, EM]
        
        if self.return_sequences:
            return weighted_input
        
        weighted_sum = tf.reduce_sum(weighted_input, axis=1)  # sum over SL, result [N, EM]
        
        return weighted_sum

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(Simple_DotProductAttention, self).get_config()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config
###############################################################################
### Additive Attention Implementation
class BahdanauAttention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # Shape of weights should match the last dimension of the input
        self.Wa = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weights'
        )
        
        # Shape of bias should match the last dimension of the input
        self.ba = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        
        # Initialize the context vector
        self.Wc = self.add_weight(
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True,
            name='context_vector'
        )

    def call(self, inputs):
        x = inputs
        
        # Compute score using additive attention
        # Shape of x: [batch_size, seq_len, dim]
        # Shape of Wa: [dim, dim]
        # Shape of ba: [dim]
        
        # Apply the linear transformation and bias
        score = tf.tanh(tf.matmul(x, self.Wa) + self.ba)  # [batch_size, seq_len, dim]
        
        # Compute attention scores
        # Shape of Wc: [dim]
        attention_scores = tf.reduce_sum(score * self.Wc, axis=-1)  # [batch_size, seq_len]
        
        # Compute attention weights using softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch_size, seq_len]
        
        # Reshape attention weights
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # [batch_size, seq_len, 1]
        
        # Compute weighted sum
        weighted_input = x * attention_weights  # [batch_size, seq_len, dim]
        
        if self.return_sequences:
            return weighted_input
        
        weighted_sum = tf.reduce_sum(weighted_input, axis=1)  # sum over seq_len, result [batch_size, dim]
        
        return weighted_sum

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config

### Attention class with query , key, value (transformar approach)
### Single Head
class single_head_Attention(Layer):
    def __init__(self,return_sequences=True, **kwargs ):
        super(single_head_Attention, self).__init__(**kwargs) ##
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.embedding_dim = input_shape[-1]
        self.W_query = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_query')
        self.W_key = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_key')
        self.W_value = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_value')
        super(single_head_Attention, self).build(input_shape)

    def call(self, inputs):
        queries =tf.linalg.matmul(inputs, self.W_query)# K.dot(inputs, self.W_query)## dim [N, SL, EmbDim]
        keys =tf.linalg.matmul(inputs, self.W_key)# K.dot(inputs, self.W_key)
        values = tf.linalg.matmul(inputs, self.W_value)# K.dot(inputs, self.W_value)
        
        #scores = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1))) ## dim [N, SL,SL] scale for stability
        scores = tf.matmul(queries, keys, transpose_b=True) 
        scores = scores / K.sqrt(K.cast(self.embedding_dim, 'float32')) ## probability !
        attention_weights = K.softmax(scores)
        
        #weighted_sum = K.batch_dot(attention_weights, values)
        # Compute weighted sum
        weighted_sum = tf.matmul(attention_weights, values) # [N,SL,SL] + [N, SL,SL] batch-wise matrix multiplication
        if self.return_sequences:
          return weighted_sum
        
        return tf.reduce_mean(weighted_sum, axis=1) #K.mean(weighted_sum, axis=1)#  [batch_size, embedding_dim] resulting cintext vector
    def get_config(self):
       config = super(single_head_Attention, self).get_config()
       config.update({
           'return_sequences': self.return_sequences
       })
       return config

### multi head attention
class multi_Head_Attention(Layer):
    def __init__(self, num_heads, return_sequences=True, **kwargs):
        super(multi_Head_Attention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.embedding_dim = input_shape[-1]
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("The embedding dimension must be divisible by the number of heads.")
        self.depth = self.embedding_dim // self.num_heads

        self.W_query = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_query')
        self.W_key = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_key')
        self.W_value = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_value')
        self.W_output = self.add_weight(shape=(self.embedding_dim, self.embedding_dim), initializer='glorot_uniform', name='W_output')

        super(multi_Head_Attention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (batch_size, seq_len, num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        queries = tf.matmul(inputs, self.W_query)  # (batch_size, seq_len, embedding_dim)
        keys = tf.matmul(inputs, self.W_key)      # (batch_size, seq_len, embedding_dim)
        values = tf.matmul(inputs, self.W_value)  # (batch_size, seq_len, embedding_dim)

        # Split and transpose for multi-head attention
        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len, depth)
        keys = self.split_heads(keys, batch_size)        # (batch_size, num_heads, seq_len, depth)
        values = self.split_heads(values, batch_size)    # (batch_size, num_heads, seq_len, depth)

        # Scaled dot-product attention
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(self.depth, tf.float32))  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)
        weighted_sum = tf.matmul(attention_weights, values)  # (batch_size, num_heads, seq_len, depth)

        # Concatenate heads
        weighted_sum = tf.transpose(weighted_sum, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        concat_attention = tf.reshape(weighted_sum, (batch_size, -1, self.embedding_dim))  # (batch_size, seq_len, embedding_dim)

        # Output projection step: Combining Information from Multiple Heads
        output = tf.matmul(concat_attention, self.W_output)  # (batch_size, seq_len, embedding_dim)

        if self.return_sequences:
            return output  # (batch_size, seq_len, embedding_dim)

        return tf.reduce_mean(output, axis=1)  # (batch_size, embedding_dim)

    def get_config(self):
        config = super(multi_Head_Attention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'return_sequences': self.return_sequences
        })
        return config

###############################################################################
### Model Definition ##########################################################
def build_model(maxlen, vocab_size, embedding_size, embedding_matrix, target_count,model_type):
    if(model_type == 'no_attention'):
        print('no attention enabled')
        # Input layer for words
        input_words = Input(shape=(maxlen,))
        
        # Embedding layer with pre-trained word embeddings
        x_words = Embedding(input_dim=vocab_size,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            mask_zero=True,
                            trainable=False)(input_words) ### traianble false because i have pre trained matrix
        
        # Apply Spatial Dropout
        #x_words = SpatialDropout1D(0.2)(x_words)
        
        # Bidirectional LSTM layer
        x = Bidirectional(LSTM(50, return_sequences=False))(x_words) ##x_words = LSTM(50, return_sequences=True)(x_words)
        ###########################################################################
        # Dense layer with ReLU activation
        x = Dense(25, activation='relu')(x)
        # Output layer with softmax activation
        pred = Dense(target_count, activation='sigmoid')(x) ## calc with softmax 
        # Define and compile the model
        model = Model(inputs=input_words, outputs=pred)
        
    elif(model_type =='simple_attention'):
        print('simple_attention enabled')
        # Input layer for words
        input_words = Input(shape=(maxlen,))
        
        # Embedding layer with pre-trained word embeddings
        x_words = Embedding(input_dim=vocab_size,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            mask_zero=True,
                            trainable=False)(input_words) ### traianble false because i have pre trained matrix
        
        # Apply Spatial Dropout
        x_words = SpatialDropout1D(0.3)(x_words)
        # Bidirectional LSTM layer
        x_words = Bidirectional(LSTM(50, return_sequences=True))(x_words) ##x_words = LSTM(50, return_sequences=True)(x_words)
        #x_words = LSTM(150, return_sequences=True)(x_words)
        # Apply Attention mechanism
        x = Simple_DotProductAttention(return_sequences=True)(x_words)
        #x = Attention()([x_words, x_words]) 
        ###########################################################################
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Global Average Pooling layer to reduce dimensions and summarize sequence information
        x = GlobalAveragePooling1D()(x)
        # Dense layer with ReLU activation
        x = Dense(50, activation='relu')(x)
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Output layer with softmax activation
        pred = Dense(target_count, activation='sigmoid')(x) ## calc with sigm. (softmax?)  
        # Define and compile the model
        model = Model(inputs=input_words, outputs=pred)

    elif(model_type == 'bahdanau_attention'):
        print('bahdanau attention enabled')
        input_words = Input(shape=(maxlen,))
        
        # Embedding layer with pre-trained word embeddings
        x_words = Embedding(input_dim=vocab_size,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            mask_zero=True,
                            trainable=False)(input_words) ### traianble false because i have pre trained matrix
        
        # Apply Spatial Dropout
        x_words = SpatialDropout1D(0.3)(x_words)
        # Bidirectional LSTM layer
        x_words = Bidirectional(LSTM(150, return_sequences=True))(x_words) ##x_words = LSTM(50, return_sequences=True)(x_words)
        #x_words = LSTM(150, return_sequences=True)(x_words)
        # Apply bahadanau Attention mechanism
        x = BahdanauAttention(return_sequences=True)(x_words)
        #x = Attention()([x_words, x_words]) 
        ###########################################################################
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Global Average Pooling layer to reduce dimensions and summarize sequence information
        x = GlobalAveragePooling1D()(x)
        # Dense layer with ReLU activation
        x = Dense(150, activation='relu')(x)
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Output layer with softmax activation
        pred = Dense(target_count, activation='sigmoid')(x) ## calc with sigm. (softmax?)  
        # Define and compile the model
        model = Model(inputs=input_words, outputs=pred)
        
    elif(model_type == 'single_h_attention'):
        print('single head attention enabled')
        input_words = Input(shape=(maxlen,))
        
        # Embedding layer with pre-trained word embeddings
        x_words = Embedding(input_dim=vocab_size,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            mask_zero=True,
                            trainable=False)(input_words) ### traianble false because i have pre trained matrix
        
        # Apply Spatial Dropout
        x_words = SpatialDropout1D(0.3)(x_words)
        # Bidirectional LSTM layer
        x_words = Bidirectional(LSTM(150, return_sequences=True))(x_words) ##x_words = LSTM(50, return_sequences=True)(x_words)
        #x_words = LSTM(150, return_sequences=True)(x_words)
        # Apply qkv Attention mechanism
        x = single_head_Attention(return_sequences=True)(x_words)
        #x = Attention()([x_words, x_words]) 
        ###########################################################################
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Global Average Pooling layer to reduce dimensions and summarize sequence information
        x = GlobalAveragePooling1D()(x)
        # Dense layer with ReLU activation
        x = Dense(150, activation='relu')(x)
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Output layer with softmax activation
        pred = Dense(target_count, activation='sigmoid')(x) ## calc with sigm. (softmax?) 
        # Define and compile the model
        model = Model(inputs=input_words, outputs=pred)
    
    elif(model_type == 'multi_h_attention'):
        print('multi head attention enabled')
        input_words = Input(shape=(maxlen,))
        
        # Embedding layer with pre-trained word embeddings
        x_words = Embedding(input_dim=vocab_size,
                            output_dim=embedding_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            mask_zero=True,
                            trainable=False)(input_words) ### traianble false because i have pre trained matrix
        
        # Apply Spatial Dropout
        x_words = SpatialDropout1D(0.3)(x_words)
        # Bidirectional LSTM layer
        x_words = Bidirectional(LSTM(150, return_sequences=True))(x_words) ##x_words = LSTM(50, return_sequences=True)(x_words)
        #x_words = LSTM(150, return_sequences=True)(x_words)
        # Apply multi head Attention mechanism

        x = multi_Head_Attention(n_head,return_sequences=True)(x_words)
        #x = Attention()([x_words, x_words]) 
        ###########################################################################
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Global Average Pooling layer to reduce dimensions and summarize sequence information
        x = GlobalAveragePooling1D()(x)
        # Dense layer with ReLU activation
        x = Dense(150, activation='relu')(x)
        # Apply Dropout
        x = Dropout(0.2)(x)
        # Output layer with softmax activation
        pred = Dense(target_count, activation='sigmoid')(x) ## calc with sigm. (softmax?) 
        # Define and compile the model
        model = Model(inputs=input_words, outputs=pred)
    
    return model

###############################################################################

if __name__ == "__main__":

    ###load data: #############################################################
    print("Loading data...")
    try:
        toxic_comments = pd.read_csv(path+"train.csv")
    except:
        print("No Train Data available..")
        sys.exit()
    try:
        toxic_comments_test = pd.read_csv(path +"test.csv")
    except:
        print("No Test Data available..")
        sys.exit()
    
    print("Train shape:", toxic_comments.shape)
    print("Test shape:", toxic_comments_test.shape)

    toxic_comments_cleaned = toxic_comments.loc[toxic_comments['comment_text'].notnull()]
    toxic_comments_cleaned.dropna(inplace=True)
    toxic_comments_labels = toxic_comments_cleaned[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    ### test data
    toxic_comments_test_cleaned = toxic_comments_test.loc[toxic_comments['comment_text'].notnull()]

    ######### datainvestigate ##########################################
    # Count the number of 1s in each column
    # Count the number of 1s in each column
    count_ones = toxic_comments_labels .apply(lambda x: (x == 1).sum())
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    count_ones.plot(kind='bar', color='skyblue')
    plt.title('Count of 1s in Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Count of 1s')
    plt.xticks(rotation=0)  # Rotate x-axis labels to horizontal
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Check for multi-labeled instances
    multi_labeled_instances_count = (toxic_comments_labels.sum(axis=1) > 1).sum()
    # Print the count of multi-labeled instances
    print(f"Number of multi-labeled instances: {multi_labeled_instances_count}")
    # Identify multi-labeled rows (rows with more than one label)
    multi_labelled_rows = toxic_comments_labels[toxic_comments_labels.sum(axis=1) > 1]
    # Drop the 'multi_label_count' column for counting class occurrences
    class_counts = multi_labelled_rows.sum()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Count of Multi-Labeled Class')
    plt.xticks(rotation=45)
    plt.show()
    ###############################################################################
    

    ################################################################################
    if(choice == 'enable_train'):

    
        def load_embeddings(filename):
            embeddings = {}
            try:
                with open(filename, encoding='utf-8') as f:
                    for line in f:
                        values = line.rstrip().split(' ')
                        word = values[0]
                        vector = np.asarray(values[1:], dtype='float32')
                        embeddings[word] = vector
            except FileNotFoundError:
                print(f"File {filename} not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
            return embeddings

        embeddings = load_embeddings(EMBEDDING_FILE)
        ###############################################################################
        X = []
        sentences = list(toxic_comments_cleaned["comment_text"])
        for sen in sentences:
            X.append(preprocess_text(sen))
    
        X_test = []
        sentences_test = list(toxic_comments_test_cleaned["comment_text"])
        for sen in sentences_test:
            X_test.append(preprocess_text(sen))
        #toxic_comments_cleaned ['comment_text'] = toxic_comments_cleaned['comment_text'].str.replace("n't", 'not')
        #toxic_comments_cleaned['comment_text'] = toxic_comments_cleaned['comment_text'].apply(lambda x: re.sub(r'[0-9]+', '0', x))
        ##############################################################################    
        y_train = toxic_comments_labels.values

        ########### start tokenizer and cleaning the sentences ########################
        ## add together train and test
        X_all = X+X_test
    
        tokenizer = Tokenizer(lower=True, filters='\n\t')
        tokenizer.fit_on_texts(X_all) # This method updates the internal vocabulary based on the list of texts x
    
        
        x_train = tokenizer.texts_to_sequences(X)
        x_test  = tokenizer.texts_to_sequences(X_test)
        vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.
        print('vocabulary size: {}'.format(vocab_size))
    
        #### padding
        all_sequences = x_train+ x_test
        # Find the maximum length of the sequences
        #maxlen = len(max(all_sequences, key=len))
        ###
        # Calculate the lengths of all sequences
        sequence_lengths = [len(seq) for seq in all_sequences]
        # Calculate the 95th percentile length
        maxlen = int(np.percentile(sequence_lengths, 95))
    
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
        print('maxlen: {}'.format(maxlen))
        print(x_train.shape)
        print(x_test.shape)
    
    
        # Split training data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        #The filter_embeddings function is essential for creating an embedding matrix from pre-trained word embeddings
        #ensuring that only embeddings for words present in your specific vocabulary (word_index) are included.
        # This matrix can then be used to initialize an embedding layer in a neural network, 
        #facilitating the use of pre-trained embeddings in natural language processing tasks effectively.
        def filter_embeddings(embeddings, word_index, vocab_size, dim):
            # Initialize the embedding matrix with zeros
           embedding_matrix = np.zeros((vocab_size, dim))
           
           # Iterate over the word_index dictionary
           for word, i in word_index.items():
               # Skip words with indices out of bounds
               if i >= vocab_size:
                   continue
               
               # Get the embedding vector for the word
               vector = embeddings.get(word)
               
               # If the embedding vector exists, assign it to the embedding matrix
               if vector is not None:
                   # Ensure the vector has the correct dimension
                   if len(vector) == dim:
                       embedding_matrix[i] = vector
                   else:
                       print(f"Dimension mismatch for word '{word}': expected {dim}, got {len(vector)}")
               else:
                   print(f"Word '{word}' not found in embeddings")
           
           return embedding_matrix
        #breakpoint()

        embedding_matrix = filter_embeddings(embeddings, tokenizer.word_index,
                                             vocab_size, embedding_size)
        #how many words in your tokenizer's vocabulary are out-of-vocabulary with respect to your pre-trained embeddings.
        print('OOV: {}'.format(len(set(tokenizer.word_index) - set(embeddings))))
        #######
        print('Data loaded..')
    ###########################################################################
    
        ### save the tokenizer
        with open('tokenizer_'+model_type+'.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #######################################################################
        print('Start Training Step..')
        
        model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix,target_count, model_type)

        # Create the Nadam optimizer with the specified learning rate
        # Register the custom layer
        #get_custom_objects()['DotProductAttention'] = DotProductAttention #Register the custom layer 
        #get_custom_objects()['Simple_DotProductAttention'] = Simple_DotProductAttention #Register the custom layer 
        # Load model with custom objects
        #model = load_model('best_model_'+model_type+ '.h5', custom_objects={'Simple_DotProductAttention': Simple_DotProductAttention})
        optimizer = Nadam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy sparse_categorical_crossentropy
        model.summary()
        
        # Define callbacks
        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model_'+model_type+'.h5', save_best_only=True, monitor='val_loss', mode='min')
        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=ep, verbose=1,
                            batch_size=128, shuffle=True,
                            # callbacks=[early_stopping, model_checkpoint])
                            callbacks=[early_stopping, model_checkpoint,lr_scheduler])
        
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(model_type + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
       # Adjust layout to prevent overlap
        plt.tight_layout()
    
       # Save the combined plot
        plt.savefig(model_type + '_accuracyloss__plot.png')
    
      # Show the combined plot
        plt.show()

        score = model.evaluate(x_val, y_val, verbose=1)
        ###############################################################################
        #breakpoint()
        #y_pred = []
        # y_pred = model.predict(x_test)
        #y_val_pred = model.predict(x_val)
    ##########################################################################
    elif(choice=='eval_performance'):
        print('Evaluation performence..')
        #model_type = 'simple_attention'
        # Load the tokenizer
        with open('tokenizer_'+model_type+'.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        if(model_type == 'no_attention'):
            try:
                model = load_model('best_model_'+model_type+ '.h5')
            except:
                print('no model found..')
                sys.exit()  # Exit the script
        
        elif(model_type == 'simple_attention'):
            try:
                model = load_model('best_model_'+model_type+ '.h5', custom_objects={'Simple_DotProductAttention': Simple_DotProductAttention})
            except:
                print('no model found..')
                sys.exit()  # Exit the script
        
        elif(model_type == 'bahdanau_attention'):
            try:
                model = load_model('best_model_'+model_type+ '.h5', custom_objects={'BahdanauAttention': BahdanauAttention})
            except:
                print('no model found..')
                sys.exit()  # Exit the script
        
        elif(model_type == 'single_h_attention'):#qkv_attention
            try:
                model = load_model('best_model_'+model_type+ '.h5', custom_objects={'single_head_Attention': single_head_Attention})
            except:
                print('no model found..')
                sys.exit()  # Exit the script
                
        elif(model_type == 'multi_h_attention'):#qkv_attention
            try:
                model = load_model('best_model_'+model_type+ '.h5', custom_objects={'multi_Head_Attention': multi_Head_Attention})
            except:
                print('no model found..')
                sys.exit()  # Exit the script

        X = []
        sentences = list(toxic_comments_cleaned["comment_text"])
        for sen in sentences:
            X.append(preprocess_text(sen))
    
        X_test = []
        sentences_test = list(toxic_comments_test_cleaned["comment_text"])
        for sen in sentences_test:
            X_test.append(preprocess_text(sen))
        #toxic_comments_cleaned ['comment_text'] = toxic_comments_cleaned['comment_text'].str.replace("n't", 'not')
        #toxic_comments_cleaned['comment_text'] = toxic_comments_cleaned['comment_text'].apply(lambda x: re.sub(r'[0-9]+', '0', x))
        ##############################################################################    
        y_train = toxic_comments_labels.values
        #new_test_sentences_cleaned = [preprocess_text(sen) for sen in new_test_sentences]
        x_train = tokenizer.texts_to_sequences(X)
        #x_test  = tokenizer.texts_to_sequences(X_test)
        
        
        # Infer the maxlen for padding from the model's input shape
        input_shape = model.layers[0].input_shape
        #breakpoint()
        # The input_shape is typically in the format list-touple(batch_size, sequence_length)
        # We take the second element (sequence_length)
        maxlen = input_shape[0][1]
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        # x_train = x_train.astype(np.float32)
        # x_val = x_val.astype(np.float32)
        # y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        #y_val_single_label = np.argmax(y_val, axis=1)        
        y_val_pred = model.predict(x_val)
        # Convert probabilities to binary predictions with a threshold of 0.5
        predictions = (y_val_pred >= 0.5).astype(np.float32)

        # Compute confusion matrices for each label
        n_labels = y_val.shape[1]
        # Save raw counts confusion matrices
        plt.figure(figsize=(15, 5))
        for i in range(n_labels):
            cm = confusion_matrix(y_val[:, i], predictions[:, i])
            plt.subplot(1, n_labels, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Present', 'Present'],
                        yticklabels=['Not Present', 'Present'])
            plt.xlabel(f'Predicted Label for Class {i}')
            plt.ylabel(f'True Label for Class {i}')
            plt.title(f'Confusion Matrix for Class {i} (Counts)')
        plt.tight_layout()
        plt.savefig(model_type + '_confusion_matrix_counts.png', dpi=300)
        plt.close()  # Close the figure to free memory
        
        # Save percentage confusion matrices
        plt.figure(figsize=(15, 5))
        for i in range(n_labels):
            cm = confusion_matrix(y_val[:, i], predictions[:, i])
            cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage
            plt.subplot(1, n_labels, i + 1)
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Not Present', 'Present'],
                        yticklabels=['Not Present', 'Present'])
            plt.xlabel(f'Predicted Label for Class {i}')
            plt.ylabel(f'True Label for Class {i}')
            plt.title(f'Confusion Matrix for Class {i}')
        plt.tight_layout()
        plt.savefig(model_type+ '_confusion_matrix_percentages.png', dpi=300)
        plt.close()  # Close the figure to free memory
        
        
        
        
