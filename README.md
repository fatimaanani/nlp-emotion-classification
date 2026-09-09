# Real-Time Emotion Detection from Text Using NLP

A Natural Language Processing project that classifies text into six emotion categories using TF-IDF and supervised machine learning.

The project compares Logistic Regression and Multinomial Naive Bayes for text classification and includes a simple prediction function for testing new sentences.

## Emotion Classes

The model classifies text into six emotions:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Dataset

The project uses the **Emotions Dataset** from Kaggle.

Dataset:
https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset

The dataset contains text samples paired with numerical emotion labels.

| Label | Emotion |
|------:|---------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

## Project Workflow

### 1. Exploratory Data Analysis

The dataset is inspected to understand its:

- Structure and dimensions
- Columns and data types
- Missing values
- Duplicate records
- Emotion class distribution

The class distribution is also examined to account for dataset imbalance when evaluating the models.

### 2. Data Cleaning

The dataset is cleaned by:

- Removing missing values
- Removing duplicate records
- Converting emotion labels to integers
- Converting text to lowercase
- Removing punctuation and numbers

### 3. Text Preprocessing

Stopwords are removed using scikit-learn's English stopword collection.

Negation words such as `not`, `no`, and `never` are intentionally preserved because they can significantly change the emotional meaning of a sentence.

The words `want` and `really` are added to the custom stopword collection.

### 4. Train-Test Split

The processed dataset is divided into:

- 80% training data
- 20% testing data

A stratified split is used to preserve the distribution of emotion classes across the training and testing sets.

### 5. TF-IDF Vectorization

`TfidfVectorizer` is used to transform the cleaned text into numerical features that can be processed by the machine learning models.

The vectorizer is fitted only on the training data, then used to transform both the training and testing sets.

### 6. Model Training

Two machine learning classifiers are trained and compared:

#### Logistic Regression

Logistic Regression is trained on the TF-IDF features with a maximum of 1000 iterations.

#### Multinomial Naive Bayes

A Multinomial Naive Bayes classifier is trained on the same TF-IDF representation for comparison.

## Model Evaluation

Both models are evaluated using:

- Accuracy
- F1-score
- Classification report
- Confusion matrix

Because the dataset contains an imbalanced distribution of emotion classes, F1-score is considered alongside accuracy when evaluating model performance.

Confusion matrices are visualized using Matplotlib and Seaborn to show how predictions are distributed across the six emotion classes.

## Model Comparison

The notebook compares the performance of Logistic Regression and Naive Bayes.

Logistic Regression achieved stronger performance in the experiment and is therefore used for the final emotion prediction function.

## Testing New Sentences

The final section allows a user to enter a new sentence:

```text
Enter a sentence: I finally finished everything and I feel amazing
```

The sentence goes through the same preprocessing and TF-IDF transformation pipeline before being classified by the trained Logistic Regression model.

Example output:

```text
Predicted Emotion: joy
```

## Technologies

- Python
- Pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Matplotlib
- Seaborn
- Google Colab

## Project Structure

```text
emotion-classification/
├── Emotion_Detection_Project1.ipynb
└── README.md
```

## Running the Project

### Google Colab

The notebook can be opened and executed in Google Colab.

The dataset must be available in Google Drive or the dataset loading path should be updated before running the notebook.

### Locally

The `.ipynb` file can also be downloaded and opened using Jupyter Notebook or JupyterLab.

Required Python libraries include:

```text
pandas
scikit-learn
matplotlib
seaborn
```

## Author

**Fatima Anani**

Computer Science Graduate
