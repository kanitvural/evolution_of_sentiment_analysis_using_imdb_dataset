# Evolution of Sentiment Analysis Using IMDB Dataset

In the rapidly evolving field of Natural Language Processing (NLP), the ability of machines to understand and interpret human language has progressed from simple rule-based approaches to the complexities of deep learning and AI-driven models. In this project, we explore the evolution of sentiment analysis using IMDB movie reviews, highlighting the advancements in NLP techniques over time while also showcasing the differences in performance between machine learning, deep learning, and transformer-based models.

## Topics Covered in This Project:
- Data validation, text analysis, and visualizations
- Cleaning special characters in text
- Removing rare words
- Lemmatization of words
- Various vectorization techniques
- Sentiment analysis using Logistic Regression and Naive Bayes with Bag of Words and TF-IDF vectorization techniques
- Sentiment analysis with deep learning models using Bidirectional LSTM, Self Multi-Head Attention, and word2vec embeddings
- Sentiment analysis using a transformer-based BERT model
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Conclusion and final thoughts

All functions used in this project are well-documented and can be found in the `utils.py` file.

GitHub repository for this project:

[https://github.com/kanitvural/evolution_of_sentiment_analysis_using_imdb_dataset](https://github.com/kanitvural/evolution_of_sentiment_analysis_using_imdb_dataset)

---

## Table of Contents:

### **1. Data Validation**
### **2. Text Preprocessing**
- **2.1. Text Cleaning**
- **2.2. Remove Duplicates**
- **2.3. Remove Stopwords**
- **2.4. Remove Rarewords**
- **2.5. Lemmatization**
- **2.6. Text Preprocessing Function**

### **3. Exploratory Data Analysis**
- **3.1. Countplot of Target**
- **3.2. WordCloud**
- **3.3. Distribution of Number of Words Per Reviews**
- **3.4. Most Frequent Words in Negative and Positive Reviews**

### **4. Vectorization**

### **5. Sentiment Modelling**
- **5.1 Sentiment Analysis with Machine Learning Models (Logistic Regression, Naive Bayes)**
  - **5.1.1. Bag of Words (BoW)**
  - **5.1.2. Term Frequency-Inverse Document Frequency (TF-IDF)**
  
- **5.2 Sentiment Analysis with Deep Learning Models (CNN, LSTM, Multi-Head Self-Attention)**
- **5.3 Sentiment Analysis with BERT Model**

### **6. Topic Modeling**
- **6.1. Data Preprocessing**
- **6.2. Tokenization**
- **6.3. Creating a Dictionary**
- **6.4. Creating a Corpus**
- **6.5. Creating the Latent Dirichlet Allocation (LDA) Model**
- **6.6. Calculating Coherence Score**
- **6.7. Other Topic Modeling Methods**

### **7. Conclusion and Final Thoughts**

---

## Technologies Used:

- **Working Environment:** Ubuntu 22.04 LTS
- **Python:** The programming language used for the project.
- **TensorFlow & Keras:** Used to build deep learning models, including LSTM and Transformer architectures.
- **Transformers Library:** Utilized for fine-tuning and implementing the BERT model.
- **NLTK & TextBlob:** For text preprocessing, tokenization, and sentiment analysis.
- **Scikit-learn:** Used for machine learning models such as Logistic Regression and Naive Bayes, along with vectorization techniques like TF-IDF and BoW.
- **Gensim:** Applied for topic modeling using the Latent Dirichlet Allocation (LDA) method.
- **Seaborn & Matplotlib:** For visualizations and exploratory data analysis (EDA).
- **Pandas & Numpy:** For handling data and performing numerical computations.




## Installation

To run this project locally, follow these steps:

   ```bash
   git clone https://github.com/kanitvural/evolution_of_sentiment_analysis_using_imdb_dataset.git
   cd evolution_of_sentiment_analysis_using_imdb_dataset
   python3 -m venv venv
   - For Linux/macOS
   source venv/bin/activate
   - For Windows:
   .\venv\Scripts\activate
   pip install -r requirements.txt