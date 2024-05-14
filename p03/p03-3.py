import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from sklearn.metrics import accuracy_score


news = load_files('data', encoding='utf-8', decode_error='replace')
X = news.data
y = news.target

my_stop_words = stopwords.words('english')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=93)

vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = 'ignore')

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model_base = [('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)), ('nb', MultinomialNB())]

stacking_model = StackingClassifier(estimators = model_base, final_estimator = LogisticRegression(multi_class = 'multinomial', random_state = 30, max_iter = 1000))
print("\n Voting model \n")
print(stacking_model)

accuracy = stacking_model.fit(X_train_vectors, y_train).score(X_test_vectors, y_test)


result = []
result.append(accuracy)

print("\n Stacking model accuracy", accuracy, '\n')

