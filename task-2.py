import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('twitter-sentiment-analysis.csv', encoding='latin-1', names=columns)

df['target'] = df['target'].replace(4, 1)

df_sample = df.sample(500000, random_state=42)

def clean_tweet(text):

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|[^a-z\s]", '', text)
    
    tokens = [word for word in text.split() if word not in stop_words]
    
    lemmed_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmed_tokens)

print("Preprocessing text data with Lemmatization...")
df_sample['clean_text'] = df_sample['text'].apply(clean_tweet)

print("Converting to TF-IDF (50,000 features, (1,3) n-grams)...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 3))
X = tfidf.fit_transform(df_sample['clean_text'])
y = df_sample['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training LinearSVC model with C=0.5...")
model = LinearSVC(C=0.5, max_iter=2000, random_state=42)
model.fit(X_train, y_train)

print("\n--- FINAL EVALUATION REPORT ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))