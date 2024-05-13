import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

## Loading the data
print("## Loading the data")
subjects = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']


def load_data():
    api_reddit = praw.Reddit(
        client_id="L8Vwdc_r2RwRfFVCwmYq6Q",
        client_secret="QhGegFNMfw85Tprb546qqVY7SsG8Ng",
        password="",
        user_agent="Cerf DSA App",
        username="_cerf",
    )

    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))
    mask = lambda post: char_count(post) >= 100
    
    data = []
    labels= []
    
    for i, subject in enumerate(subjects):
        subreddit_data = api_reddit.subreddit(subject).new(limit=1000)
        posts = [post.selftext for post in filter(mask, subreddit_data)]
        data.extend(posts)
        labels.extend([i] * len(posts))

        print(f"Number of posts in r/{subject}: {len(posts)}",
              f"\n One example post: {posts[0][:600]}...\n",
              "_" *80 + '\n')
    return data, labels


## Dividing the data into training and testing sets
print("## Dividing the data into training and testing sets")
TEST_SIZE = 0.2
RANDOM_STATE = 0

def split_data(data, labels):
    print(f"Split {100 * TEST_SIZE}% of the data into the test set")

    X_train, X_test, Y_train, Y_test = train_test_split(data, 
                                                       labels, 
                                                       test_size=TEST_SIZE, 
                                                       random_state=RANDOM_STATE)

    print(f"Number of test samples: {len(Y_test)}")

    return X_train, X_test, Y_train, Y_test



## Preprocessing the data
print("## Preprocessing the data")
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

def prerocessing_pipeline():
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    vectorizer = TfidfVectorizer(preprocessor=preprocessor, stop_words='english',
                                 min_df=MIN_DOC_FREQ)

    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline


## selecting the model
print("## Selecting the model")
N_NEIGHBORS = 4
CV = 4

def build_model():
    model1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model2 = RandomForestClassifier(random_state=RANDOM_STATE)
    model3 = LogisticRegressionCV(cv=CV,random_state=RANDOM_STATE)
    models = [("KNN", model1), ("RandomForest", model2), ("LogReg", model3)]
    return models


## training and evaluating the model
print("## Training and evaluating the model")
def train_and_evaluate(models, pipeline, X_train, X_test, Y_train, Y_test):

    results=[]

    for name, model in models:
        pipe = Pipeline(pipeline + [(name, model)])

        print(f"Training the model {name} with the training data")
        pipe.fit(X_train, Y_train)

        y_pred = pipe.predict(X_test)

        report = classification_report(Y_test, y_pred)
        print(f"Cassification report for {name}: \n {report}")

        results.append([model, {'model': name, 'preview': y_pred, 'report': report}])

        return results


if __name__ == "__main__":
    data, labels = load_data()

    x_train, x_test, y_train, y_test = split_data(data, labels)

    pipeline = prerocessing_pipeline()

    models = build_model()

    results = train_and_evaluate(models, pipeline, x_train, x_test, y_train, y_test)

print("End of the program")


def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15,6), dpi=120)
    plt.title("Distribution of the number of posts per subreddit")
    sns.barplot(x=subjects, y=counts)
    plt.legend([' '.join([f.title(), f"- {c} posts"]) for f,c in zip(subjects, counts)])
    plt.show()

def plot_confusion(result):
    print("Classification report\n", result[-1]['report'])
    y_pred=result[-1]['predictions']
    conf_matrix=confusion_matrix(y_test,y_pred)
    _,test_counts = np.unique(y_test, return_conts=True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize=(9,8), dpi=120)
    plt.title(result[-1]['model'].upper() + 'Results')
    plt.xlabel('Real value')
    plt.ylabel('Model prediction')
    ticklabels=[f"r/{sub}" for sub in subjects]
    sns.heatmap(data=conf_matrix_percent,xticklabels=ticklabels,yticklabels=ticklabels,annot=True,fmt='.2f')
    plt.show()


plot_distribution()
plot_confusion(results[0])
plot_confusion(results[1])
plot_confusion(results[2])







