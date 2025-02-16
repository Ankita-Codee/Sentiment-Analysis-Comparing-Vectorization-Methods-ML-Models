# Sentiment Analysis: Word2Vec, TF-IDF, BOW Vector Representation, and Random Forest/Naive Bayes Models

## Dataset
The dataset used for this project is the **Kindle Reviews** dataset, which can be downloaded from [Kaggle: Amazon Kindle Book Review for Sentiment Analysis](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis). A detailed description of the dataset can be found at the provided Kaggle link.
Basically, it contains product review text and associated ratings. The data is preprocessed by converting text to lowercase, removing stop words, and tokenizing the sentences. The dataset is split into training and test sets for model evaluation.

## Methods
This project applies three different text representation techniques to convert text into numerical features:
1. **Bag of Words (BOW)**: A simple representation where each word in the document is represented by its frequency of occurrence.
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure used to evaluate how important a word is to a document in a collection or corpus.
3. **Word2Vec**:
We used the **Word2Vec** model for generating vector representations of words. Word2Vec is a deep learning model that captures semantic relationships between words by converting them into dense vector representations. For this project, we utilized the **pretrained Word2Vec model** provided by the **Gensim library** and specifically loaded the **'word2vec-google-news-300'** model. 

### Classifiers Used:
1. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, used for classification tasks in text.
2. **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees and aggregating their results.

## Results Summary

| Model            | Feature Representation | Accuracy |
|------------------|-------------------------|----------|
| Naive Bayes      | BOW                     | 59%       |
| Naive Bayes      | TF-IDF                  | 62%       |
| Naive Bayes      | Word2Vec                | 74%       |
| Random Forest    | BOW                     | 80%       |
| Random Forest    | TF-IDF                  | 80%       |
| Random Forest    | Word2Vec                | 78%       |

### Key Insights:
- **Naive Bayes** performs best with **Word2Vec** achieving 74% accuracy, an improvement over BOW (59%) and TF-IDF (62%).
- **Random Forest** achieves the highest accuracy (80%) with **BOW** and **TF-IDF**, while **Word2Vec** performs slightly lower at 78%.
- **Word2Vec** improves the accuracy for **Naive Bayes** but doesn’t show the same improvement for **Random Forest**.

