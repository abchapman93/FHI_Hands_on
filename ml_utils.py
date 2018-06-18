from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import collections


import matplotlib.pyplot as plt

def vectorize_documents(documents, vectorizer=None, ngrams=(1,1), min_df=1):
    if not vectorizer:
        vectorizer = CountVectorizer(tokenizer=word_tokenize, ngram_range=ngrams,
                                
                                     min_df=min_df, lowercase=False)
        X = vectorizer.fit_transform(documents)
    else:
        X = vectorizer.transform(documents)
    return X, vectorizer

def display_word_matrix(X, vectorizer, n=5):
    df = pd.DataFrame(X.todense())
    df.columns = vectorizer.get_feature_names()
    return df.head(n)



def read_in_data():
    from collections import Counter
    pos_doc_type='FAM_BREAST_CA_DOC'
    docs_train = read_doc_annotations(archive_file='data/bc_train.zip', pos_type=pos_doc_type)

    texts_train, labels_train = zip(*((doc.text, "Positive Document")
                          if doc.annotations[0].type == pos_doc_type
                          else (doc.text, "Negative Document") 
                          for doc in docs_train.values()))


    docs_test = read_doc_annotations(archive_file='img/bc_test.zip', pos_type=pos_doc_type)
    texts_test, labels_test = zip(*((doc.text, "Positive Document")
                          if doc.annotations[0].type == pos_doc_type
                          else (doc.text, "Negative Document") 
                          for doc in docs_test.values()))

    
def preprocess(text):
    text = text.lower()
    # Remove punctuation, special symbols
    text = re.sub("[:./,%#()'\"&+-;<>@?]*", "", text)
    # Change any combination digits to be a special NUM symbol
    text = re.sub("[\d]+", "NUM", text)
    # Remove excess whitespace for human readability
    text = re.sub("[\n\s]+", " ", text)
    # Add additional code here
    ## 
    
    return text


    # texts += test_texts
    # labels += test_labels
    c_train = Counter(labels_train)
    c_test = Counter(labels_test)
    print()
    print("Number of positive training docs: {}".format(c_train["Positive Document"]))
    print("Number of negative training docs: {}".format(c_train["Negative Document"]))
    print()
    print("Number of positive testing docs: {}".format(c_test["Positive Document"]))
    print("Number of negative testing docs: {}".format(c_test["Negative Document"]))
    
    return texts_train, labels_train, texts_test, labels_test


def evaluate_cross_val_clfs(X, y, clfs):
    clf_names = []
    clf_scores = []
    for clf in clfs:
        model_name = clf.__repr__().split("(")[0]
        pred = cross_val_predict(clf, X, y, cv=5)
        clf_names.append(model_name)
        clf_scores.append(f1_score(y, pred, pos_label='Positive Document'))
    plot_results(clf_names, clf_scores)
    return clf_names, clf_scores

def plot_results(clf_names, clf_scores):
    x = range(len(clf_names))
    plt.plot(x, clf_scores, marker='.')
    plt.xticks(x, clf_names, rotation=45)
    plt.xlabel("Classifier Name")
    plt.ylabel("F1")
    return


def visualize_tree(dtree, feature_names):
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                    filled=True, rounded=True,
                    feature_names=feature_names,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    return Image(graph.create_png())