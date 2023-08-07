#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


dataset = pd.read_csv("E:/malicious_phish1.csv")
dataset.head(20)


# In[3]:


pd.value_counts(dataset['type']).plot.bar()


# In[4]:


X = dataset['url']
y = dataset['type']


# In[5]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[9]:


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)


# In[10]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)
dt_predictions = dt_model.predict(X_test_tfidf)


# In[11]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train_tfidf, y_train)
xgb_predictions = xgb_model.predict(X_test_tfidf)


# In[12]:


def evaluate_model(predictions, model_name):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')  # Update this line
    recall = recall_score(y_test, predictions, average='weighted')  # Update this line
    f1 = f1_score(y_test, predictions, average='weighted')  # Update this line
    print(f'{model_name} Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}\n')


# In[13]:


evaluate_model(lr_predictions, 'Logistic Regression')
evaluate_model(dt_predictions, 'Decision Tree')
evaluate_model(xgb_predictions, 'XGBClassifier')


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


lr_accuracy = accuracy_score(y_test, lr_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)


# In[18]:


models = ['Logistic Regression', 'Decision Tree', 'XGBClassifier']
accuracies = [lr_accuracy, dt_accuracy, xgb_accuracy]


# In[17]:


plt.bar(models, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1.0)
plt.show()


# In[20]:


lr_precision = precision_score(y_test, lr_predictions, average='weighted')
dt_precision = precision_score(y_test, dt_predictions, average='weighted')
xgb_precision = precision_score(y_test, xgb_predictions, average='weighted')


# In[21]:


models = ['Logistic Regression', 'Decision Tree', 'XGBClassifier']
precisions = [lr_precision, dt_precision, xgb_precision]


# In[22]:


plt.bar(models, precisions)
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Precision Comparison')
plt.ylim(0, 1.0)
plt.show()


# In[23]:


lr_recall = recall_score(y_test, lr_predictions, average='weighted')
dt_recall = recall_score(y_test, dt_predictions, average='weighted')
xgb_recall = recall_score(y_test, xgb_predictions, average='weighted')


# In[24]:


models = ['Logistic Regression', 'Decision Tree', 'XGBClassifier']
recalls = [lr_recall, dt_recall, xgb_recall]


# In[25]:


plt.bar(models, recalls)
plt.xlabel('Model')
plt.ylabel('Recall')
plt.title('Recall Comparison')
plt.ylim(0, 1.0)
plt.show()


# In[27]:


lr_F1score = f1_score(y_test, lr_predictions, average='weighted')
dt_F1score = f1_score(y_test, dt_predictions, average='weighted')
xgb_F1score = f1_score(y_test, xgb_predictions, average='weighted')


# In[28]:


models = ['Logistic Regression', 'Decision Tree', 'XGBClassifier']
f1_scores = [lr_F1score, dt_F1score, xgb_F1score]


# In[29]:


plt.bar(models, f1_scores)
plt.xlabel('Model')
plt.ylabel('F1-score')
plt.title('F-measure Comparison')
plt.ylim(0, 1.0)
plt.show()


# In[ ]:




