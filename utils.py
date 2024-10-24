import re
from typing import Union

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from textblob import Word
from wordcloud import WordCloud


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
    print(f'Review:  \n{review[0]} \n\n Prediction: {label}')


