# Sentiment-Analysis-on-Tripadvisor-Reviews-Using-Machine-Learning
This project focuses on performing sentiment analysis on Tripadvisor reviews to classify them as positive or negative based on the content of the reviews. The process begins with text preprocessing, followed by feature extraction, and model training to predict sentiment.

## Table of Contents
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Techniques and Tools Used](#techniques-and-tools-used)
- [Outcome](#outcome)
- [Technologies Used](#technologies-used)

## Dataset

The dataset used in this project is sourced from Kaggle: [TripAdvisor Reviews Dataset](https://www.kaggle.com/datasets/ilhamfp31/dataset-tripadvisor). It contains hotel reviews from TripAdvisor, which are labeled with sentiments (Positive or Negative). The dataset is used to train and evaluate the text classification models.

### Dataset Features:
- **Content**: The main body of text that represents user reviews.
- **Polarity**: The label for the review, which is either Positive or Negative.

## Exploratory Data Analysis (EDA)

To understand the dataset better, an exploratory data analysis was performed, including visualizing the distribution of sentiments. The dataset contains reviews labeled as **Positive** or **Negative**, and their distribution provides insight into the balance of the dataset.

### Sentiment Distribution:

The following bar chart illustrates the count of reviews for each sentiment:

- **Positive**: Represents reviews with a favorable sentiment.
- **Negative**: Represents reviews with an unfavorable sentiment.

This visualization helps in identifying whether the dataset is balanced or imbalanced, which could affect model performance.

![Distribusi Sentimen](https://github.com/user-attachments/assets/50611b5c-5b65-45bb-99b5-88455bff6fa3)

## Techniques and Tools Used

### Text Preprocessing:
- **Case Folding**: Converting all text to lowercase to maintain uniformity and avoid case-sensitive discrepancies.
- **Word Normalization**: Standardizing non-standard words (e.g., slang or misspellings) into their standard form.
- **Stopword Removal**: Eliminating common words (e.g., "the", "and", "is") that do not contribute to the meaning of the text.
- **Stemming**: Reducing words to their root form (e.g., "running" becomes "run").

### Feature Extraction:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert the text data into numerical features based on the importance of each word in the corpus.
- **Feature Selection (SelectKBest)**: Applied Chi-square tests to select the most relevant features from the TF-IDF results, reducing the dimensionality and improving model performance.

### Machine Learning Models:
- **Naive Bayes**: A probabilistic classifier that assumes the features are independent. It provides a simple yet effective model for text classification.
- **Logistic Regression**: A statistical model used for binary classification that achieved the highest accuracy of 80%.
- **Random Forest**: An ensemble learning method that combines multiple decision trees for better prediction accuracy.

### Model Evaluation:
- **Accuracy**: The percentage of correct predictions out of the total predictions.
- **Confusion Matrix**: A detailed breakdown of predictions, showing true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Cross-Validation**: Used to assess the performance of the models across different data splits to ensure robustness.

### Deployment:
- The final **Logistic Regression** model was deployed for automated sentiment prediction on new Tripadvisor reviews, with real-time preprocessing to handle user inputs efficiently.

## Outcome:
This project successfully built a model that can classify Tripadvisor reviews into positive or negative sentiments with an accuracy of 80%. The deployment allows for automatic sentiment analysis on new data, providing an easy-to-use solution for analyzing user feedback at scale.

![Screenshot (1489)](https://github.com/user-attachments/assets/0642007c-350c-4f78-b4b5-88cbbc389192)

## Technologies Used:
- **Python** (libraries such as scikit-learn, pandas, nltk)
- **Google Colab**
- **Machine Learning Algorithms** (Naive Bayes, Logistic Regression, Random Forest)
- **TF-IDF** for feature extraction
- **Model evaluation metrics** (accuracy, confusion matrix, classification report)
