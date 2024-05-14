import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import threading

news = load_files('data', encoding='utf-8', decode_error='replace')
X = news.data
y = news.target

d = {}
d['soft'] = []
d['hard'] = []

def threading_f(x):
  print('Thread:', x)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=x)
  my_stop_words = stopwords.words('english')

  vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = 'ignore')

  X_train_vectors = vectorizer.fit_transform(X_train)
  X_test_vectors = vectorizer.transform(X_test)

  model1 = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', random_state = 30, max_iter = 1000)
  model2 = RandomForestClassifier(n_estimators = 1000, max_depth = 100, random_state = 1)
  model3 = MultinomialNB()

  for i in ['soft', 'hard']:
    voting_model = VotingClassifier(estimators = [('lg',model1), ('rf', model2), ('nb', model3)], voting = i)
    voting_model = voting_model.fit(X_train_vectors, y_train)
    predict = voting_model.predict(X_test_vectors)
    print("-Random State:", x, '-Voting:', i, '-Accuracy:', accuracy_score(y_test, predict))
    d[i].append((x, accuracy_score(y_test, predict)))

# for x in range(1,100):
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=x)
  # my_stop_words = stopwords.words('english')

  # vectorizer = TfidfVectorizer(norm = None, stop_words = my_stop_words, max_features = 1000, decode_error = 'ignore')

  # X_train_vectors = vectorizer.fit_transform(X_train)
  # X_test_vectors = vectorizer.transform(X_test)

  # model1 = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', random_state = 30, max_iter = 1000)
  # model2 = RandomForestClassifier(n_estimators = 1000, max_depth = 100, random_state = 1)
  # model3 = MultinomialNB()

  # for i in ['soft', 'hard']:
  #   voting_model = VotingClassifier(estimators = [('lg',model1), ('rf', model2), ('nb', model3)], voting = i)
  #   voting_model = voting_model.fit(X_train_vectors, y_train)
  #   predict = voting_model.predict(X_test_vectors)
  #   print("-Random State:", x, '-Voting:', i, '-Accuracy:', accuracy_score(y_test, predict))
  #   d[i].append((x, accuracy_score(y_test, predict)))

if __name__ == '__main__':
  threads = list()
  for x in range(1, 100):
    t = threading.Thread(target=threading_f, args=(x,))
    threads.append(t)
    t.start()
  

  for t in threads:
    t.join()
  print('\n Best Results:')

  h = max(d['hard'], key = lambda x: x[1])
  s = max(d['soft'], key = lambda x: x[1])

  print('-Random State:', h[0], '-Voting: hard', '-Accuracy:', h[1])
  print('-Random State:', s[0], '-Voting: soft', '-Accuracy:', s[1])
  print('\n')


