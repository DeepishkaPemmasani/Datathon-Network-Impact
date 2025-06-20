#!/usr/bin/env python
# coding: utf-8

#  ### 1. using a pre-trained model available in "gender_guesser" library in Python. 

# In[2]:


get_ipython().system(' pip install tensorflow')


# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import sklearn
from sklearn.model_selection import train_test_split


# In[5]:


get_ipython().system(' pip install gender-guesser')


# In[4]:


import pandas as pd
import gender_guesser.detector as gender

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMDATupdated.csv")

# Create a gender detector object
detector = gender.Detector()

# Function to predict gender
def predict_gender(name):
    first_name = name.split()[0]
    return detector.get_gender(first_name)

# Apply the predict_gender function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_SCCMDATupdated.csv", index=False)


# In[5]:


import pandas as pd
import gender_guesser.detector as gender

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMPRESupdated.csv")

# Create a gender detector object
detector = gender.Detector()

# Function to predict gender
def predict_gender(name):
    first_name = name.split()[0]
    return detector.get_gender(first_name)

# Apply the predict_gender function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_SCCMPRESupdated.csv", index=False)


# ### 2. Using an API or web service that provides gender predictions based on names, Gender API (https://gender-api.com/) or Genderize.io (https://genderize.io/). These services often have more extensive databases and can provide more accurate predictions.

# In[9]:


get_ipython().system(' pip install requests')


# In[12]:


import pandas as pd
import requests

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMDATupdated.csv")

# Function to predict gender using Genderize.io API
def predict_gender(name):
    first_name = name.split()[0]
    url = f"https://api.genderize.io/?name={first_name}"
    response = requests.get(url)
    data = response.json()
    
    if "gender" in data:
        return data["gender"] if data["gender"] is not None else "unknown"
    else:
        return "unknown"

# Apply the predict_gender function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_api_Authors_SCCMDATupdated.csv", index=False)


# In[13]:


import pandas as pd
import requests

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMPRESupdated.csv")

# Function to predict gender using Genderize.io API
def predict_gender(name):
    first_name = name.split()[0]
    url = f"https://api.genderize.io/?name={first_name}"
    response = requests.get(url)
    data = response.json()
    
    if "gender" in data:
        return data["gender"] if data["gender"] is not None else "unknown"
    else:
        return "unknown"

# Apply the predict_gender function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_api_Authors_SCCMPRESupdated.csv", index=False)


# ### Combining the results from multiple methods, such as the "gender_guesser" library and the Genderize.io API.

# In[14]:


import pandas as pd
import gender_guesser.detector as gender
import requests

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMDATupdated.csv")

# Create a gender detector object
detector = gender.Detector()

# Function to predict gender using gender_guesser library
def predict_gender_guesser(name):
    first_name = name.split()[0]
    return detector.get_gender(first_name)

# Function to predict gender using Genderize.io API
def predict_gender_api(name):
    first_name = name.split()[0]
    url = f"https://api.genderize.io/?name={first_name}"
    response = requests.get(url)
    data = response.json()
    
    if "gender" in data:
        return data["gender"] if data["gender"] is not None else "unknown"
    else:
        return "unknown"

# Function to combine gender predictions using the Ensemble Approach
def predict_gender_ensemble(name):
    gender_guesser_pred = predict_gender_guesser(name)
    gender_api_pred = predict_gender_api(name)
    
    if gender_guesser_pred == gender_api_pred:
        return gender_guesser_pred
    else:
        return "unknown"

# Apply the predict_gender_ensemble function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender_ensemble)

# Save the results to a new CSV file
authors_df.to_csv("Authors_SCCMDAT_with_gender_ensemble.csv", index=False)


# ### Gender API (https://genderize.io/), NamSor (https://www.namsor.com/)

# In[ ]:


import pandas as pd
import requests

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMDATupdated.csv")

# Function to predict gender using Genderize.io API
def predict_gender_genderize(name):
    first_name = name.split()[0]
    url = f"https://api.genderize.io/?name={first_name}"
    response = requests.get(url)
    data = response.json()
    
    if "gender" in data:
        return data["gender"] if data["gender"] is not None else "unknown"
    else:
        return "unknown"

# Function to predict gender using Gender API
def predict_gender_gender_api(name, api_key):
    first_name = name.split()[0]
    url = f"https://gender-api.com/get?name={first_name}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "gender" in data:
        return data["gender"] if data["gender"] is not None else "unknown"
    else:
        return "unknown"

# Function to predict gender using NamSor API
def predict_gender_namsor(name, api_key):
    first_name = name.split()[0]
    url = f"https://v2.namsor.com/NamSorAPIv2/api2/json/gender/{first_name}"
    headers = {"X-API-KEY": api_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if "likelyGender" in data:
        return data["likelyGender"] if data["likelyGender"] is not None else "unknown"
    else:
        return "unknown"

# API keys
gender_api_key = "07bc0fd02ba68800a05f8fb1962f68340b7bd72d1e3cf37c3d690bafeaf3ba95"
namsor_api_key = "9ad6aeff4990a09d8f15626801b0300a"

# Apply the gender prediction functions to the 'Authors' column
authors_df['Gender_Genderize'] = authors_df['Authors'].apply(predict_gender_genderize)
authors_df['Gender_GenderAPI'] = authors_df['Authors'].apply(lambda x: predict_gender_gender_api(x, gender_api_key))
authors_df['Gender_NamSor'] = authors_df['Authors'].apply(lambda x: predict_gender_namsor(x, namsor_api_key))

# Save the results to a new CSV file
authors_df.to_csv("Authors_SCCMDAT_with_gender_apis.csv", index=False)


# ### Gender Identification by Author name using NLTK

# In[2]:


get_ipython().system(' pip install nltk')


# In[3]:


import nltk
nltk.download('names')


# In[6]:


import pandas as pd
import random
from nltk.corpus import names
import nltk

def gender_features(word):
    return {'last_letter': word[-1]}

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMDATupdated.csv")

# Prepare a list of labeled names from the NLTK corpus
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# Extract features from the labeled names
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the feature sets into training and test sets
train_set, test_set = featuresets[500:], featuresets[:500]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Function to predict gender using the trained classifier
def predict_gender(name):
    first_name = name.split()[0]
    return classifier.classify(gender_features(first_name))

# Apply the gender prediction function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_nltk_SCCMDATupdated.csv", index=False)

# Print the accuracy of the classifier on the training set
print("Accuracy on training set:", nltk.classify.accuracy(classifier, train_set))

# Print the accuracy of the classifier on the test set
print("Accuracy on test set:", nltk.classify.accuracy(classifier, test_set))

# Show the most informative features of the classifier
classifier.show_most_informative_features(10)


# In[7]:


import pandas as pd
import random
from nltk.corpus import names
import nltk

def gender_features(word):
    return {'last_letter': word[-1]}

# Load your authors data
authors_df = pd.read_csv("Authors_SCCMPRESupdated.csv")

# Prepare a list of labeled names from the NLTK corpus
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# Extract features from the labeled names
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the feature sets into training and test sets
train_set, test_set = featuresets[500:], featuresets[:500]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Function to predict gender using the trained classifier
def predict_gender(name):
    first_name = name.split()[0]
    return classifier.classify(gender_features(first_name))

# Apply the gender prediction function to the 'Authors' column
authors_df['Gender'] = authors_df['Authors'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("authors_with_gender_nltk_SCCMPRESupdated.csv", index=False)

# Print the accuracy of the classifier on the training set
print("Accuracy on training set:", nltk.classify.accuracy(classifier, train_set))

# Print the accuracy of the classifier on the test set
print("Accuracy on test set:", nltk.classify.accuracy(classifier, test_set))

# Show the most informative features of the classifier
classifier.show_most_informative_features(10)


# ### Gender Identification by Fork name using NLTK

# In[8]:


import pandas as pd
import random
from nltk.corpus import names
import nltk

def gender_features(word):
    return {'last_letter': word[-1]}

# Load your authors data
authors_df = pd.read_csv("forkname SCCMDAT.csv")

# Prepare a list of labeled names from the NLTK corpus
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# Extract features from the labeled names
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the feature sets into training and test sets
train_set, test_set = featuresets[500:], featuresets[:500]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Function to predict gender using the trained classifier
def predict_gender(name):
    first_name = name.split()[0]
    return classifier.classify(gender_features(first_name))

# Apply the gender prediction function to the 'Fork Name' column
authors_df['Gender'] = authors_df['Fork Name'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("forkname_gender_nltk_SCCMDATupdated.csv", index=False)

# Print the accuracy of the classifier on the training set
print("Accuracy on training set:", nltk.classify.accuracy(classifier, train_set))

# Print the accuracy of the classifier on the test set
print("Accuracy on test set:", nltk.classify.accuracy(classifier, test_set))

# Show the most informative features of the classifier
classifier.show_most_informative_features(10)


# In[9]:


import pandas as pd
import random
from nltk.corpus import names
import nltk

def gender_features(word):
    return {'last_letter': word[-1]}

# Load your authors data
authors_df = pd.read_csv("forkname SCCMPRES.csv")

# Prepare a list of labeled names from the NLTK corpus
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# Extract features from the labeled names
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the feature sets into training and test sets
train_set, test_set = featuresets[500:], featuresets[:500]

# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Function to predict gender using the trained classifier
def predict_gender(name):
    first_name = name.split()[0]
    return classifier.classify(gender_features(first_name))

# Apply the gender prediction function to the 'Fork Name' column
authors_df['Gender'] = authors_df['Fork Name'].apply(predict_gender)

# Save the results to a new CSV file
authors_df.to_csv("forkname_gender_nltk_SCCMPRESupdated.csv", index=False)

# Print the accuracy of the classifier on the training set
print("Accuracy on training set:", nltk.classify.accuracy(classifier, train_set))

# Print the accuracy of the classifier on the test set
print("Accuracy on test set:", nltk.classify.accuracy(classifier, test_set))

# Show the most informative features of the classifier
classifier.show_most_informative_features(10)


# In[10]:


SCCMDAT = pd.read_csv("SCCMDATupdated_FINALANALYSISCSV.csv")
SCCMPRES = pd.read_csv("SCCMPRESupdated_FINALANALYSIS_csv.csv")


# ### groupby publication title and calculate the gender ratio of Author and Fork for both datasets

# In[13]:


def calculate_gender_ratio(df, group_col, gender_col):
    gender_counts = df.groupby([group_col, gender_col]).size().unstack(fill_value=0)
    gender_ratio = gender_counts.div(gender_counts.sum(axis=1), axis=0)
    return gender_ratio


# In[14]:


# SCCMDAT dataset
sccmdat_author_ratio = calculate_gender_ratio(SCCMDAT, 'Publication Title', 'Gender - Author')
sccmdat_fork_ratio = calculate_gender_ratio(SCCMDAT, 'Publication Title', 'Gender - Fork')

sccmdat_result = pd.concat([sccmdat_author_ratio, sccmdat_fork_ratio], axis=1, keys=['Author', 'Fork'])
sccmdat_result.to_csv('sccmdat_gender_ratio.csv')


# In[15]:


sccmdat_gender_ratio=pd.read_csv("sccmdat_gender_ratio.csv")


# In[16]:


sccmdat_gender_ratio.head()


# In[17]:


# SCCMPRES dataset
sccmpres_author_ratio = calculate_gender_ratio(SCCMPRES, 'Publication Title', 'Gender - Author')
sccmpres_fork_ratio = calculate_gender_ratio(SCCMPRES, 'Publication Title', 'Gender - Fork')

sccmpres_result = pd.concat([sccmpres_author_ratio, sccmpres_fork_ratio], axis=1, keys=['Author', 'Fork'])
sccmpres_result.to_csv('sccmpres_gender_ratio.csv')


# In[18]:


sccmpres_gender_ratio=pd.read_csv("sccmpres_gender_ratio.csv")


# ### Calculating gender ratios per publication for both datasets (SCCMDAT and SCCMPRES) for the years 2017 and 2023

# In[31]:


import pandas as pd

def calculate_gender_ratio(df, group_col, gender_col):
    gender_counts = df.groupby([group_col, gender_col]).size().unstack(fill_value=0)
    gender_ratio = gender_counts.div(gender_counts.sum(axis=1), axis=0)
    return gender_ratio

def calculate_descriptive_stats(df):
    return df.describe()

# Filter data for years 2017 and 2023
sccmdat_2017 = SCCMDAT[SCCMDAT['Year'] == 2017]
sccmdat_2023 = SCCMDAT[SCCMDAT['Year'] == 2023]
sccmpres_2017 = SCCMPRES[SCCMPRES['Year'] == 2017]
sccmpres_2023 = SCCMPRES[SCCMPRES['Year'] == 2023]

# SCCMDAT dataset - 2017
sccmdat_author_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Publication Title', 'Gender - Author')
sccmdat_fork_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Publication Title', 'Gender - Fork')

sccmdat_result_2017 = pd.concat([
    sccmdat_author_ratio_2017.add_prefix('Author_'),
    sccmdat_fork_ratio_2017.add_prefix('Fork_')
], axis=1)

sccmdat_result_2017.to_csv('sccmdat_gender_ratio_per_publication_2017.csv')

# SCCMDAT dataset - 2023
if not sccmdat_2023.empty:
    sccmdat_author_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Publication Title', 'Gender - Author')
    sccmdat_fork_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Publication Title', 'Gender - Fork')

    sccmdat_result_2023 = pd.concat([
        sccmdat_author_ratio_2023.add_prefix('Author_'),
        sccmdat_fork_ratio_2023.add_prefix('Fork_')
    ], axis=1)

    sccmdat_result_2023.to_csv('sccmdat_gender_ratio_per_publication_2023.csv')

# SCCMPRES dataset - 2017
sccmpres_author_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Publication Title', 'Gender - Author')
sccmpres_fork_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Publication Title', 'Gender - Fork')

sccmpres_result_2017 = pd.concat([
    sccmpres_author_ratio_2017.add_prefix('Author_'),
    sccmpres_fork_ratio_2017.add_prefix('Fork_')
], axis=1)

sccmpres_result_2017.to_csv('sccmpres_gender_ratio_per_publication_2017.csv')

# SCCMPRES dataset - 2023
if not sccmpres_2023.empty:
    sccmpres_author_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Publication Title', 'Gender - Author')
    sccmpres_fork_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Publication Title', 'Gender - Fork')

    sccmpres_result_2023 = pd.concat([
        sccmpres_author_ratio_2023.add_prefix('Author_'),
        sccmpres_fork_ratio_2023.add_prefix('Fork_')
    ], axis=1)

    sccmpres_result_2023.to_csv('sccmpres_gender_ratio_per_publication_2023.csv')

# Descriptive statistics for SCCMDAT dataset
print("Descriptive Statistics for SCCMDAT dataset - 2017:")
print(calculate_descriptive_stats(sccmdat_result_2017))

if not sccmdat_2023.empty:
    print("\nDescriptive Statistics for SCCMDAT dataset - 2023:")
    print(calculate_descriptive_stats(sccmdat_result_2023))

# Descriptive statistics for SCCMPRES dataset
print("\nDescriptive Statistics for SCCMPRES dataset - 2017:")
print(calculate_descriptive_stats(sccmpres_result_2017))

if not sccmpres_2023.empty:
    print("\nDescriptive Statistics for SCCMPRES dataset - 2023:")
    print(calculate_descriptive_stats(sccmpres_result_2023))


# ### Comparing the gender ratios of the years 2017 and 2023 for both datasets (SCCMDAT and SCCMPRES)

# In[20]:


import pandas as pd

def calculate_gender_ratio(df, gender_col):
    gender_counts = df.groupby(gender_col).size()
    total_count = gender_counts.sum()
    gender_ratio = gender_counts / total_count
    return gender_ratio

# Filter data for years 2017 and 2023
sccmdat_2017 = SCCMDAT[SCCMDAT['Year'] == 2017]
sccmdat_2023 = SCCMDAT[SCCMDAT['Year'] == 2023]
sccmpres_2017 = SCCMPRES[SCCMPRES['Year'] == 2017]
sccmpres_2023 = SCCMPRES[SCCMPRES['Year'] == 2023]

# SCCMDAT dataset
sccmdat_author_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Gender - Author')
sccmdat_author_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Gender - Author')
sccmdat_fork_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Gender - Fork')
sccmdat_fork_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Gender - Fork')

sccmdat_result = pd.DataFrame({
    'Author_2017': sccmdat_author_ratio_2017,
    'Author_2023': sccmdat_author_ratio_2023,
    'Fork_2017': sccmdat_fork_ratio_2017,
    'Fork_2023': sccmdat_fork_ratio_2023
})
sccmdat_result.to_csv('sccmdat_gender_ratio_comparison.csv')

# SCCMPRES dataset
sccmpres_author_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Gender - Author')
sccmpres_author_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Gender - Author')
sccmpres_fork_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Gender - Fork')
sccmpres_fork_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Gender - Fork')

sccmpres_result = pd.DataFrame({
    'Author_2017': sccmpres_author_ratio_2017,
    'Author_2023': sccmpres_author_ratio_2023,
    'Fork_2017': sccmpres_fork_ratio_2017,
    'Fork_2023': sccmpres_fork_ratio_2023
})
sccmpres_result.to_csv('sccmpres_gender_ratio_comparison.csv')


# In[22]:


sccmdat_result


# In[23]:


sccmpres_result


# ### To compare the gender ratios of authors and forks for the years 2017 and 2023 per publication

# In[25]:


import pandas as pd

def calculate_gender_ratio(df, group_col, gender_col):
    gender_counts = df.groupby([group_col, gender_col]).size().unstack(fill_value=0)
    gender_ratio = gender_counts.div(gender_counts.sum(axis=1), axis=0)
    return gender_ratio

# Filter data for years 2017 and 2023
sccmdat_2017 = SCCMDAT[SCCMDAT['Year'] == 2017]
sccmdat_2023 = SCCMDAT[SCCMDAT['Year'] == 2023]
sccmpres_2017 = SCCMPRES[SCCMPRES['Year'] == 2017]
sccmpres_2023 = SCCMPRES[SCCMPRES['Year'] == 2023]

# SCCMDAT dataset
sccmdat_author_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Publication Title', 'Gender - Author')
sccmdat_author_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Publication Title', 'Gender - Author')
sccmdat_fork_ratio_2017 = calculate_gender_ratio(sccmdat_2017, 'Publication Title', 'Gender - Fork')
sccmdat_fork_ratio_2023 = calculate_gender_ratio(sccmdat_2023, 'Publication Title', 'Gender - Fork')

sccmdat_gender_ratio_comparison_per_publication = pd.concat([
    sccmdat_author_ratio_2017.add_prefix('Author_2017_'),
    sccmdat_author_ratio_2023.add_prefix('Author_2023_'),
    sccmdat_fork_ratio_2017.add_prefix('Fork_2017_'),
    sccmdat_fork_ratio_2023.add_prefix('Fork_2023_')
], axis=1)
sccmdat_gender_ratio_comparison_per_publication.to_csv('sccmdat_gender_ratio_comparison_per_publication.csv')

# SCCMPRES dataset
sccmpres_author_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Publication Title', 'Gender - Author')
sccmpres_author_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Publication Title', 'Gender - Author')
sccmpres_fork_ratio_2017 = calculate_gender_ratio(sccmpres_2017, 'Publication Title', 'Gender - Fork')
sccmpres_fork_ratio_2023 = calculate_gender_ratio(sccmpres_2023, 'Publication Title', 'Gender - Fork')

sccmpres_gender_ratio_comparison_per_publication = pd.concat([
    sccmpres_author_ratio_2017.add_prefix('Author_2017_'),
    sccmpres_author_ratio_2023.add_prefix('Author_2023_'),
    sccmpres_fork_ratio_2017.add_prefix('Fork_2017_'),
    sccmpres_fork_ratio_2023.add_prefix('Fork_2023_')
], axis=1)
sccmpres_gender_ratio_comparison_per_publication.to_csv('sccmpres_gender_ratio_comparison_per_publication.csv')


# In[30]:


sccmdat_gender_ratio_comparison_per_publication.head()


# In[28]:


sccmpres_gender_ratio_comparison_per_publication.head()


# In[ ]:




