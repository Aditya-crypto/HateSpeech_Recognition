import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer



df=pd.read_csv("/content/drive/My Drive/assign5_dataset/train.csv")
# print(df)
test=pd.read_csv("/content/drive/My Drive/assign5_dataset/test.csv")
# print(test)

labels=df['labels']

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def clean_text(text):
  text=text.lower()     
  text=" ".join(filter(lambda x:x[0]!='@', text.split()))
  text=remove_punctuation(text)
  text=re.sub(r"http\S+", "",text)
  text=remove_numbers(text)
  text=stem_words(text)
  return text

round1= lambda x: clean_text(x)

tempdata=df.text.apply(round1)
clean_data=pd.DataFrame(tempdata)


tfidf = TfidfVectorizer(stop_words='english')  

data_cv1=tfidf.fit_transform(clean_data['text'])
dataset=pd.DataFrame(data_cv1.toarray(),columns=tfidf.get_feature_names())


Finaltest=pd.read_csv("/content/drive/My Drive/assign5_dataset/f_test.csv")
print(Finaltest.columns)


Finaltest=pd.read_csv("/content/drive/My Drive/assign5_dataset/f_test.csv")
clean_test_data=pd.DataFrame(Finaltest.text.apply(round1))
data_cv1=tfidf.transform(clean_test_data.text)
Finaltestdata=pd.DataFrame(data_cv1.toarray(),columns=tfidf.get_feature_names())

print(Finaltestdata)


labels=np.array(labels)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset,labels, test_size=0.1)

from sklearn.svm import SVC

from sklearn import svm


SVMmodel=svm.SVC(kernel='rbf')
SVMmodel.fit(X_train,y_train)
SVMlabels = SVMmodel.predict(X_test)

finalpredlabels = SVMmodel.predict(Finaltestdata)


def create_file(predictLabels,testdata):
  smiles=testdata['Unnamed: 0'] 
  list_of_tuples = list(zip(smiles, predictLabels))  
  sub = pd.DataFrame(list_of_tuples, columns = [' ', 'labels'])   
  print(sub)
  sub.to_csv('submission.csv')

create_file(finalpredlabels,Finaltest)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,SVMlabels)

from sklearn.metrics import f1_score
f1_score(y_test, SVMlabels)


