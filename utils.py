import re
from typing import Union, Dict, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from textblob import Word
from wordcloud import WordCloud
from gensim.models import Word2Vec

import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Layer,  BatchNormalization, Bidirectional, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Text Cleaning
def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing HTML tags, punctuation, numbers, and extra whitespace.

    Parameters
    ----------
    text : str
        The input text string to be cleaned.

    Returns
    -------
    str
        The cleaned and normalized text string.
    
    Notes
    -----
    This function performs the following operations:
        - Converts the text to lowercase.
        - Removes punctuation.
        - Removes HTML tags, including <br> tags.
        - Removes numbers.
        - Collapses multiple spaces into a single space and trims leading/trailing spaces.
    """
    text = text.lower() 
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s*br\s*/?\s*", "", text)
    text = re.sub(r"<.*?>", "", text) 
    text = re.sub(r"\d", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

################################################################################################

# Text Preprocessing
def text_preprocessing(df: pd.DataFrame, sentiment_feature: str, remove_rare_words:bool=False) -> pd.DataFrame:
    """Preprocess text data for sentiment analysis.

    This function performs several preprocessing steps on the specified text column
    of a pandas DataFrame, including cleaning the text, removing duplicates, 
    filtering out stopwords, eliminating rare words, and lemmatizing the text.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the text data to be preprocessed.
    sentiment_feature : str
        The name of the column in the DataFrame that contains the text data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the preprocessed text data in the specified sentiment feature column.
    
    Notes
    -----
    The function checks if the 'wordnet' resource is downloaded; if not, it downloads it.
    """
    
    df[sentiment_feature] = df[sentiment_feature].apply(clean_text)
    
    df.drop_duplicates(inplace=True)
    
    sw = stopwords.words("english")
    df[sentiment_feature] = df[sentiment_feature].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    
    if remove_rare_words:
            rare_words = pd.Series(" ".join(df[sentiment_feature]).split()).value_counts()[-1000:]
            df[sentiment_feature] = df[sentiment_feature].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))
    
    # Check if 'wordnet' is already downloaded
    try:
        nltk.data.find("corpora/wordnet")
        print("WordNet is already downloaded.")
    except LookupError:
        print("WordNet is not downloaded. Downloading now...")
        nltk.download("wordnet")
    
    df[sentiment_feature] = df[sentiment_feature].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
    
    return df


# Word Cloud Generator
def generate_word_cloud(df: pd.DataFrame, 
                        column: str,
                        max_font_size: int = 50,
                        max_words: int = 100,
                        background_color: str = "black",
                        interpolation: str = "bilinear") -> None:
    """
    Generate and display a word cloud from the text in a specified column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the text data.
    column : str
        The name of the column from which the text data will be used to generate the word cloud.
    max_font_size : int, optional
        The maximum font size for the largest word in the word cloud (default is 50).
    max_words : int, optional
        The maximum number of words to be included in the word cloud (default is 100).
    background_color : str, optional
        The background color of the word cloud (default is "black").
    interpolation : str, optional
        The interpolation method used for displaying the word cloud (default is "bilinear").

    Returns
    -------
    None
        Displays the word cloud as a plot.

    """
    text = " ".join(text for text in df[column])
    wordcloud = WordCloud(max_font_size=max_font_size,
                          max_words=max_words,
                          background_color=background_color).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation=interpolation)
    plt.axis("off")
    plt.show()

################################################################################################

# Machine Learning Train and Evaluation 

def train_and_evaluate_model(
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str,
    scoring_func: callable,
    cv_type: int
) -> BaseEstimator:
    """Train and evaluate a machine learning model.

    This function fits the model on the training data, performs cross-validation
    to evaluate its performance, makes predictions on the test data, calculates
    the final score using a specified scoring function, and visualizes the
    confusion matrix.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to be trained.
    x_train : pd.DataFrame
        Feature data for training.
    y_train : pd.Series
        Target data for training.
    x_test : pd.DataFrame
        Feature data for testing.
    y_test : pd.Series
        Target data for testing.
    scoring : str
        The scoring metric to evaluate the model during cross-validation.
    scoring_func : callable
        A function to calculate the final score based on true and predicted labels.
    cv_type : int
        The number of cross-validation folds.

    Returns
    -------
    BaseEstimator
        The trained machine learning model.
    """

    trained_model = model.fit(x_train, y_train)
    result_cv_score = cross_val_score(trained_model, x_train, y_train, scoring=scoring, cv=cv_type).mean()
    y_pred = trained_model.predict(x_test)
    result_final_score = scoring_func(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Train cross validation score: {result_cv_score:.2f}")
    print(f"Final test score: {result_final_score:.2f}")

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    
    return model

# Machine learning Prediction

def predict_sentiment(
    model: BaseEstimator,
    vectorizer: Union[TfidfVectorizer, CountVectorizer],
    review: str
) -> None:
    """Predict the sentiment of a given review using the specified model and vectorizer.

    This function transforms the input review using the provided vectorizer, 
    makes a prediction with the given model, and prints the sentiment label.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model used for prediction.
    vectorizer : Union[TfidfVectorizer, CountVectorizer]
        The vectorizer used to transform the input review into feature space.
    review : str
        The review text to be analyzed.

    Returns
    -------
    None
        This function prints the input review and the predicted sentiment label.
    """
    if isinstance(review, str):
        review = [review]
        
    new_review = vectorizer.transform(review)
    pred = model.predict(new_review)
    label = "positive" if pred[0] == 1 else "negative"
    print(f"Review:  \n{review[0]} \n\n Prediction: {label}")

################################################################################################

# Deep Learning Functions



# Train word2vec with sentiment data and get embeddings matrix
def train_word2vec_and_get_embedding_matrix(
    X_train: List[List[str]], 
    vector_size: int, 
    total_vocabulary_size: int,
    tokenizer: Tokenizer 
) -> np.ndarray:
    """
    Trains a Word2Vec model on the training data and constructs an embedding matrix.

    Parameters
    ----------
    X_train : List[List[str]]
        The training data, a list of tokenized sentences, where each sentence is represented as a list of words.
    
    vector_size : int
        The size of the word vectors to be generated by the Word2Vec model.
    
    total_word_size : int
        The total number of unique words in the vocabulary.

    tokenizer : Tokenizer
        A Keras Tokenizer instance used to map words to their indices.
    
    Returns
    -------
    np.ndarray
        An embedding matrix where each row corresponds to a word vector for the respective word index.
    """

    model = Word2Vec(X_train, vector_size=vector_size, window=5, min_count=1, workers=4)
    
    embedding_matrix = np.zeros((total_vocabulary_size + 1, vector_size))

    for word, i in tokenizer.word_index.items():
        if word in model.wv:
            embedding_matrix[i] = model.wv[word]

    return embedding_matrix



# Processing
def preprocess_sentiment_data(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    most_freq_words: int, 
    max_padding_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses text data for use in CNN-LSTM models by tokenizing and padding the sequences.

    Parameters
    ----------
    X_train : np.ndarray
        An array of training text data (reviews).
    
    X_test : np.ndarray
        An array of testing text data (reviews).
    
    most_freq_words : int
        The maximum number of words to keep, based on word frequency. 
        Only the most frequent `most_freq_words` will be kept.
    
    max_padding_length : int
        The maximum length of the sequences after padding. 
        Sequences longer than this will be truncated.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
            - processed_train: Padded sequences for the training data.
            - processed_test: Padded sequences for the testing data.
    """
    
    tokenizer = Tokenizer(num_words=most_freq_words, oov_token="<oov>")
    tokenizer.fit_on_texts(X_train)
    train_review_sequences = tokenizer.texts_to_sequences(X_train)
    test_review_sequences = tokenizer.texts_to_sequences(X_test)
    processed_train = pad_sequences(train_review_sequences, truncating='post', padding='pre', maxlen=max_padding_length)
    processed_test = pad_sequences(test_review_sequences, truncating='post', padding='pre', maxlen=max_padding_length)
    
    return processed_train, processed_test
  

# Model and training

def self_attention_block(
    inputs: np.ndarray, 
    num_heads: int, 
    ff_dim: int
) -> np.ndarray:
    """
    Applies a self-attention mechanism to the input tensor.

    Parameters
    ----------
    inputs : np.ndarray
        The input tensor to which self-attention will be applied.
    
    num_heads : int
        The number of attention heads to use in the multi-head attention layer.
    
    ff_dim : int
        The dimensionality of the feed-forward layer.

    Returns
    -------
    np.ndarray
        The output tensor after applying self-attention and feed-forward layers.
    """
    # Multi-Head Attention 
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = Dropout(0.1)(attention_output)
    
    # Layer Normalization
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward Layer
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    
    # Residual Connection
    return Add()([ffn_output, inputs])


def sentiment_cnn_lstm_att_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    total_word_size: int, 
    embedding_dim: int, 
    maxlen: int, 
    batch_size: int, 
    validation_split: float, 
    epochs: int, 
    embedding_matrix: np.ndarray
) -> Tuple[np.ndarray, Model]:
    """
    Builds and trains a CNN-LSTM model with a self-attention mechanism for sentiment analysis.

    Parameters
    ----------
    X_train : np.ndarray
        The training input data (text sequences).
    
    y_train : np.ndarray
        The target output labels for the training data.
    
    total_word_size : int
        The total number of unique words in the vocabulary.
    
    embedding_dim : int
        The dimension of the word embeddings.
    
    maxlen : int
        The maximum length of input sequences after padding.
    
    batch_size : int
        The number of samples per gradient update.
    
    validation_split : float
        The fraction of the training data to be used as validation data.
    
    epochs : int
        The number of epochs to train the model.
    
    embedding_matrix : np.ndarray
        A matrix of pre-trained embeddings for initializing the embedding layer.

    Returns
    -------
    Tuple[np.ndarray, Model]
        A tuple containing:
            - result: Training history of the model.
            - model: The trained Keras model instance.
    """
    
    inputs = Input(shape=(maxlen,))
    
    x = Embedding(total_word_size + 1, 
                  embedding_dim, 
                  input_length=maxlen,
                  weights=[embedding_matrix])(inputs)
    x = BatchNormalization()(x)
    
    # Conv1D + MaxPooling (Extracting local features)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Self Attention Layer
    x = self_attention_block(x, num_heads=4, ff_dim=512)
    
    # Bidirectional layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    
 
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    checkpoint_callback = ModelCheckpoint(
        filepath="./model/best_model.keras",  
        monitor="val_loss",
        save_best_only=True,  
        save_weights_only=False,
        mode="min",
        verbose=1
    )
        
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,  
        verbose=1
    )

    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=2,
        min_lr=0.00001,
        verbose=1
    ) 
    
    result = model.fit(X_train, y_train, 
                       validation_split=validation_split,
                       epochs=epochs, 
                       batch_size=batch_size,
                       callbacks=[checkpoint_callback, early_stopping_callback, lr_callback])
    
    return result, model


# Plotting
def plot_training_history(result: Tuple[np.ndarray, Model]) -> None:  # Giriş tipi burada değişiyor
    """
    Plots the training and validation loss and accuracy graphs from the model's training history.

    Parameters:
    -----------
    result : Tuple[np.ndarray, Model]
        A tuple containing:
            - history: Training history of the model.
            - model: The trained Keras model instance.

    Returns:
    --------
    None
        This function does not return any value, it just displays the plots.
    """
    
   
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(result.history['loss'], label='Train Loss')
    plt.plot(result.history['val_loss'], label='Validation Loss')
    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(result.history['accuracy'], label='Train Accuracy')
    plt.plot(result.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Evaluate   
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    
    """
    Evaluates the performance of the given model on the training and testing data.
    
    Parameters
    ----------
    model : keras.Model
        The trained Keras model to evaluate.
        
    X_train : np.ndarray
        The training input data.
        
    y_train : np.ndarray
        The true labels for the training data.
        
    X_test : np.ndarray
        The testing input data.
        
    y_test : np.ndarray
        The true labels for the testing data.
        
    Returns
    -------
    None
    """
    
   
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    cf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Prediction
    
def sentiment_cnn_lstm_att_predict(text: str, model: Model, tokenizer: Tokenizer, max_sequence_length: int) -> None:
    """
    Predicts the sentiment of a given text using the provided model and tokenizer.

    Parameters
    ----------
    text : str
        The input text for which sentiment prediction is to be made.
    model : keras.Model
        The pre-trained Keras model used for making predictions.
    tokenizer : Tokenizer
        The tokenizer used to preprocess the input text.
    max_sequence_length : int
        The maximum length of the input sequences after padding.

    Returns
    -------
    None
        This function prints the original text and its predicted sentiment.

    """
    
    text_clean = clean_text(text)
    sequences = tokenizer.texts_to_sequences([text_clean])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)
    prediction = (prediction > 0.5).astype(int)
    
    result = "Positive" if prediction == 1 else "Negative"
    print(f"Review:\n\n{text}\n\n{result}")
    
 ################################################################################################   
 
# Bert Model



# Model and training

@register_keras_serializable(package="DistilBert", name="DistilBert")
class CustomDistilBertLayer(Layer):
    """
    Custom layer that utilizes a DistilBERT transformer model to extract features.

    This layer takes input IDs and attention masks, processes them through the 
    DistilBERT transformer, and returns the hidden state corresponding to the 
    [CLS] token.

    Attributes:
        transformer: A pre-trained transformer model (e.g., DistilBERT).

    Methods:
        call(inputs): Processes the input IDs and attention mask through the transformer.
    """

    def __init__(self, transformer: Model, **kwargs) -> None:
        """
        Initializes the CustomDistilBertLayer with a given transformer model.

        Args:
            transformer (tf.keras.Model): The transformer model to be used.
            **kwargs: Additional keyword arguments for the parent Layer class.
        """
        super(CustomDistilBertLayer, self).__init__(**kwargs)
        self.transformer = transformer

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Processes the input IDs and attention masks through the transformer.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): A tuple containing input IDs 
                and attention masks.

        Returns:
            tf.Tensor: The hidden state corresponding to the [CLS] token from 
                the transformer output.
        """
        input_ids, attention_mask = inputs
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return transformer_output.last_hidden_state[:, 0, :]
    
    def get_config(self):
        """
        Returns the configuration of the custom layer.

        This method is used to serialize the layer, allowing it to be 
        saved and restored correctly. The configuration includes all the 
        necessary parameters that define the layer's behavior.

        Returns:
            dict: A dictionary containing the configuration of the layer.
        """
        config = super(CustomDistilBertLayer, self).get_config()
        config.update({
            "transformer": self.transformer,  
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Recreates an instance of the class from its configuration.

        Args:
            config (dict): A dictionary containing the configuration 
                           parameters for the custom layer.

        Returns:
            An instance of the class initialized with the provided config.
        """
        return cls(**config)



def create_and_train_bert_model(train_dataset: tf.data.Dataset, 
                           val_dataset: tf.data.Dataset, 
                           max_len: int, 
                           transformer: Model, 
                           batch_size: int, 
                           epochs: int) -> Tuple[History, Model]:
    """
    Creates and trains a model using a DistilBERT transformer.

    This function defines a model architecture consisting of a CustomDistilBertLayer, 
    followed by dense layers, and compiles the model. It then trains the model 
    on the provided training dataset and validates it on the validation dataset.

    Args:
        train_dataset (tf.data.Dataset): The dataset for training the model.
        val_dataset (tf.data.Dataset): The dataset for validating the model.
        max_len (int): The maximum sequence length for the input tokens.
        transformer (tf.keras.Model): The pre-trained transformer model to be used.
        batch_size (int): The number of samples per gradient update.
        epochs (int): The number of epochs to train the model.

    Returns:
        Tuple[History, Model]: A tuple containing the 
            training history and the trained model.
    """
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    
    bert_output = CustomDistilBertLayer(transformer)([input_ids, attention_mask])
    
    x = Dense(512, activation="relu")(bert_output)
    x = Dropout(0.1)(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    optimizer = Adam(learning_rate=2e-5)
    
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    checkpoint_callback = ModelCheckpoint(
        filepath="bert_model.keras",  
        monitor="val_loss",
        save_best_only=True,  
        save_weights_only=False,
        mode="min",
        verbose=1
    )
    
     
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  
        verbose=1
    )

    lr_callback = ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.3, 
        patience=3,
        min_lr=2e-7,
        verbose=1
    ) 
    
    result = model.fit(train_dataset, 
                       validation_data=val_dataset,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[checkpoint_callback, early_stopping_callback, lr_callback])
    
    return result, model


# Evaluate

def evaluate_bert(test_dataset: tf.data.Dataset, 
                  y_test: np.ndarray, 
                  model: Model) -> None:
    """
    Evaluates the BERT model on the given test dataset and displays the results.

    This function calculates the test accuracy, prints the evaluation metrics, 
    and generates a confusion matrix heatmap to visualize the model's performance.

    Args:
        test_dataset (tf.data.Dataset): The dataset to evaluate the model on.
        y_test (np.ndarray): The true labels for the test dataset.
        model (Model): The Keras model to be evaluated.

    Returns:
        None
    """
    
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")
    
    y_pred = model.predict(test_dataset)
    y_pred = (y_pred > 0.5).astype(int)

    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()


# Prediction

def bert_predict(text: str, tokenizer, model: Model, max_len: int) -> str:
    """
    Predicts the label for a given input text using the trained BERT model.

    This function preprocesses the input text, generates predictions, and returns
    the predicted label based on a specified threshold.

    Args:
        text (str): The input text to classify.
        tokenizer: The tokenizer used to preprocess the input text.
        model (Model): The trained Keras model for predictions.
        max_len (int): The maximum length for input sequences.
        threshold (float): The threshold for binary classification.

    Returns:
        str: The predicted label ("Positive" or "Negative").
    """
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', max_length=max_len, truncation=True)
    
    predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
    probability = predictions[0][0]

    predicted_label = "positive" if probability > 0.5 else "negative"
    
    print(f"{text}\n\n{predicted_label}")
    