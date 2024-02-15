#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importing required libraries.
import os
import csv
import re
import nltk
import string
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en import English
from rouge import Rouge
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import punkt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from textblob import TextBlob


# In[4]:


current_dir = os.getcwd()
dataset_folder = os.path.join(current_dir, '')
dataset_folder


# In[5]:


# Setting up file paths
business = os.path.join(dataset_folder, 'business')
entertainment = os.path.join(dataset_folder, 'entertainment')
politics = os.path.join(dataset_folder, 'politics')
sports = os.path.join(dataset_folder, 'sports')
tech = os.path.join(dataset_folder, 'tech')


# In[6]:


# Function to read and extract text from a .txt file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# In[7]:


# Function to preprocess the text
def preprocess_text(text):
    # Removing special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Converting to lowercase
    text = " ".join(x.lower() for x in text.split())
    
    # Tokenization
    text = word_tokenize(text)
    
    # Removing stop-words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    
    # Creating string
    text = ' '.join(text)
    
    return text


# In[9]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[11]:


# Lists to store data rows for each category
business_data_rows = []
entertainment_data_rows = []
politics_data_rows = []
sports_data_rows = []
tech_data_rows = []


# In[12]:


# Iterate through each file in the directory
for filename in os.listdir(business):
    if filename.endswith(".txt"):
        file_path = os.path.join(business, filename)
        # Read and extract text from the file
        text = read_text_from_file(file_path)
        # Perform basic preprocessing
        processed_text = preprocess_text(text)
        # Add the processed text and filename to the list
        business_data_rows.append({'article_name': filename, 'article_content': text, 'processed_article_content': processed_text})

        
for filename in os.listdir(entertainment):
    if filename.endswith(".txt"):
        file_path = os.path.join(entertainment, filename)
        # Read and extract text from the file
        text = read_text_from_file(file_path)
        # Perform basic preprocessing
        processed_text = preprocess_text(text)
        # Add the processed text and filename to the list
        entertainment_data_rows.append({'article_name': filename, 'article_content': text, 'processed_article_content': processed_text})

        
for filename in os.listdir(politics):
    if filename.endswith(".txt"):
        file_path = os.path.join(politics, filename)
        # Read and extract text from the file
        text = read_text_from_file(file_path)
        # Perform basic preprocessing
        processed_text = preprocess_text(text)
        # Add the processed text and filename to the list
        politics_data_rows.append({'article_name': filename, 'article_content': text, 'processed_article_content': processed_text})

        
for filename in os.listdir(sports):
    if filename.endswith(".txt"):
        file_path = os.path.join(sports, filename)
        # Read and extract text from the file
        text = read_text_from_file(file_path)
        # Perform basic preprocessing
        processed_text = preprocess_text(text)
        # Add the processed text and filename to the list
        sports_data_rows.append({'article_name': filename, 'article_content': text, 'processed_article_content': processed_text})

        
for filename in os.listdir(tech):
    if filename.endswith(".txt"):
        file_path = os.path.join(tech, filename)
        # Read and extract text from the file
        text = read_text_from_file(file_path)
        # Perform basic preprocessing
        processed_text = preprocess_text(text)
        # Add the processed text and filename to the list
        tech_data_rows.append({'article_name': filename, 'article_content': text, 'processed_article_content': processed_text})


# In[22]:


# Write the data to a CSV file
with open('business_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Specify the CSV columns
    fieldnames = ['article_name', 'article_content','processed_article_content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data rows
    writer.writerows(business_data_rows)

print(f"CSV file business_dataset.csv created successfully.")

# Write the data to a CSV file
with open('entertainment_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Specify the CSV columns
    fieldnames = ['article_name', 'article_content','processed_article_content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data rows
    writer.writerows(entertainment_data_rows)

print(f"CSV file entertainment_dataset.csv created successfully.")

# Write the data to a CSV file
with open('politics_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Specify the CSV columns
    fieldnames = ['article_name', 'article_content','processed_article_content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data rows
    writer.writerows(politics_data_rows)

print(f"CSV file politics_dataset.csv created successfully.")

# Write the data to a CSV file
with open('sports_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Specify the CSV columns
    fieldnames = ['article_name', 'article_content','processed_article_content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data rows
    writer.writerows(sports_data_rows)

print(f"CSV file sports_dataset.csv created successfully.")

# Write the data to a CSV file
with open('tech_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Specify the CSV columns
    fieldnames = ['article_name', 'article_content','processed_article_content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data rows
    writer.writerows(tech_data_rows)

print(f"CSV file tech_dataset.csv created successfully.")


# In[23]:


# Reading CSV files into dataframes for each category
business_df = pd.read_csv('business_dataset.csv')
entertainment_df = pd.read_csv('entertainment_dataset.csv')
politics_df = pd.read_csv('politics_dataset.csv')
sports_df = pd.read_csv('sports_dataset.csv')
tech_df = pd.read_csv('tech_dataset.csv')


# In[24]:


#function to create Semantic Networks of each news
def build_semantic_network(x, i, ax):
    vectorizer = CountVectorizer(max_features=10)
    document_vectors = vectorizer.fit_transform([x])
    feature_names = list(vectorizer.vocabulary_.keys())

    co_occurrence_matrix = np.dot(document_vectors.T, document_vectors)
    co_occurrence_matrix[co_occurrence_matrix > 1] = 1

    G = nx.from_numpy_array(co_occurrence_matrix)
    node_size = [v*1000 for v in nx.degree_centrality(G).values()]
    node_color = [v for v in nx.degree_centrality(G).values()]
    node_labels = {i: feature for i, feature in enumerate(feature_names)}

    
    pos = nx.spring_layout(G, seed=42)

    nx.drawing.nx_pylab.draw_networkx(G,node_size=node_size, node_color=node_color ,pos=pos,with_labels=True,
                                      labels=node_labels,font_size=8,font_color='white',font_weight='bold', ax=ax)

    ax.set_title('semantic network'+str(i+1))
    


# In[25]:


temp_lst = business_df["processed_article_content"].iloc[:10]    
n_rows = int(np.ceil(len(temp_lst)/3))   
fig, axs = plt.subplots(n_rows, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    if i < len(temp_lst):
        build_semantic_network(temp_lst.iloc[i], i, ax)
    else:
        ax.axis('off')
plt.show()


# In[28]:


temp_lst = entertainment_df["processed_article_content"].iloc[:10]    
n_rows = int(np.ceil(len(temp_lst)/3))   
fig, axs = plt.subplots(n_rows, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    if i < len(temp_lst):
        build_semantic_network(temp_lst.iloc[i], i, ax)
    else:
        ax.axis('off')
plt.show()


# In[27]:


temp_lst = politics_df["processed_article_content"].iloc[:10]    
n_rows = int(np.ceil(len(temp_lst)/3))   
fig, axs = plt.subplots(n_rows, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    if i < len(temp_lst):
        build_semantic_network(temp_lst.iloc[i], i, ax)
    else:
        ax.axis('off')
plt.show()


# In[29]:


temp_lst = sports_df["processed_article_content"].iloc[:10]    
n_rows = int(np.ceil(len(temp_lst)/3))   
fig, axs = plt.subplots(n_rows, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    if i < len(temp_lst):
        build_semantic_network(temp_lst.iloc[i], i, ax)
    else:
        ax.axis('off')
plt.show()


# In[30]:


temp_lst = tech_df["processed_article_content"].iloc[:10]    
n_rows = int(np.ceil(len(temp_lst)/3))   
fig, axs = plt.subplots(n_rows, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axs.flatten()):
    if i < len(temp_lst):
        build_semantic_network(temp_lst.iloc[i], i, ax)
    else:
        ax.axis('off')
plt.show()


# In[31]:


#visua;ization function to get count of the text per news article
def text_visualization(strings):
    n_rows = int(np.ceil(len(strings)/3))
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, 4*n_rows), squeeze=False)

    for i, x in enumerate(strings):
        row_idx = i // 3
        col_idx = i % 3
        ax = axs[row_idx, col_idx]

        sentences = x
        #splitiing the words in the sentence passed
        word_lists = sentences.split()
        # Counting the frequency of each word in the word_lists
        freq = {}

        for word in word_lists:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
        #selecting the words that occurs most frequently to display in the plot
        j = 2
        vocab = {word: freq for word, freq in freq.items() if freq > j}

        while ( len(vocab) > 10):
            j += 1
            vocab = {word: freq for word, freq in freq.items() if freq > j}

        df1 = pd.DataFrame(list(vocab.items()), columns=['Word', 'Count'])

        # Create a bar chart of the word frequencies
        df1.plot.barh(x='Word', y='Count', rot=0, ax=ax)

        # Set the chart title and axis labels
        ax.set_title(f'Word Frequency for String {i+1}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Word')

    plt.tight_layout()
    plt.show()


# In[32]:


text_visualization(business_df["processed_article_content"].iloc[:10])


# In[33]:


text_visualization(entertainment_df["processed_article_content"].iloc[:10])


# In[34]:


text_visualization(politics_df["processed_article_content"].iloc[:10])


# In[35]:


text_visualization(sports_df["processed_article_content"].iloc[:10])


# In[36]:


text_visualization(tech_df["processed_article_content"].iloc[:10])


# In[37]:


# Functions for TF-IDF calculation
stop_words = stopwords.words('english')
def get_doc_tokens(doc):
    tokens=[token.strip() \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count

def tfidf(docs):
    # step 2. process all documents to get list of token list
    docs_tokens={idx:get_doc_tokens(doc) \
             for idx,doc in enumerate(docs)}

    # step 3. get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
    dtm = dtm.sort_index(axis = 0)
      
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1, keepdims=True)
    tf=np.divide(tf, doc_len)
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    #idf=np.log(np.divide(len(docs), \
    #    np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf


# In[38]:


Arr_TF_IDF = tfidf(business_df['processed_article_content'])
Arr_TF_IDF.shape


# In[39]:


Arr_TF_IDF = tfidf(entertainment_df['processed_article_content'])
Arr_TF_IDF.shape


# In[40]:


Arr_TF_IDF = tfidf(politics_df['processed_article_content'])
Arr_TF_IDF.shape


# In[41]:


Arr_TF_IDF = tfidf(sports_df['processed_article_content'])
Arr_TF_IDF.shape


# In[43]:


Arr_TF_IDF = tfidf(tech_df['processed_article_content'])
Arr_TF_IDF.shape


# In[53]:


# Function to create TF-IDF vectors using Latent Semantic Analysis (LSA)
def tf_idf_vectors(main_text):
    summary_arr = []
    for i in main_text:
        #first need to split article in sentences using . split
        temp_sen_arr = i.split('.')

        #convertion of sentences in TFIDF cevtors
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
        X = vectorizer.fit_transform(temp_sen_arr) 

        #Decomposition using SVD(support vector decomposition)
        U, S, Vt = svds(X, k=2)

        #creation of sentences to vectors
        sentence_vectors = np.matmul(X.toarray(), Vt.T)

        #select the number of important sentences for summary here we are summarizing with top most 3 important sentences
        num_sentences = 3
        summary_indices = np.argsort(sentence_vectors.sum(axis=1))[::-1][:num_sentences]
        summary_indices.sort()
        summary = '. '.join([temp_sen_arr[i] for i in summary_indices])

        #append to summary array
        summary_arr.append(summary)
    return summary_arr

# Applying TF-IDF vectors function to each category
business_main_text = business_df['article_content']
business_summary_arr1 = tf_idf_vectors(business_main_text)

entertainment_main_text = entertainment_df['article_content']
entertainment_summary_arr1 = tf_idf_vectors(entertainment_main_text)

politics_main_text = politics_df['article_content']
politics_summary_arr1 = tf_idf_vectors(politics_main_text)

sports_main_text = sports_df['article_content']
sports_summary_arr1 = tf_idf_vectors(sports_main_text)

tech_main_text = tech_df['article_content']
tech_summary_arr1 = tf_idf_vectors(tech_main_text)


# In[54]:


business_df['text_summary_using_LSA'] = business_summary_arr1
business_df.head()


# In[55]:


entertainment_df['text_summary_using_LSA'] = entertainment_summary_arr1
entertainment_df.head()


# In[56]:


politics_df['text_summary_using_LSA'] = politics_summary_arr1
politics_df.head()


# In[57]:


sports_df['text_summary_using_LSA'] = sports_summary_arr1
sports_df.head()


# In[58]:


tech_df['text_summary_using_LSA'] = tech_summary_arr1
tech_df.head()


# In[59]:


# Function to generate TextRank summary
nlp = English()

def textrank_summary(text, num_sentences=3):
    
    #first need to split article in sentences using . split
    sentences = text.split('. ')
    
    #convertion of sentences in TFIDF cevtors
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    X = vectorizer.fit_transform(sentences)
    
    #creating cosine similarity matrix
    sim_matrix = cosine_similarity(X)
    
    #creating graph from the sim_matrix
    nx_graph = nx.from_numpy_array(sim_matrix)
    
    # Get the scores using the pagerank algorithm
    scores = nx.pagerank(nx_graph)
    
    # Sort the sentences by score and select the top sentences
    num_sentences = min(num_sentences, len(sentences))
    if num_sentences <= 0:
        return ''
    elif num_sentences == 1:
        return sentences[np.argmax(scores)]
    else:
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        summary = '. '.join([ranked_sentences[i][1] for i in range(num_sentences)])
        return summary


# In[62]:


# Function to apply TextRank summarization
def textrank(main_text):
    summary_arr = []
    for article in main_text:
        summary = textrank_summary(article)
        summary_arr.append(summary)
    return summary_arr
    
    
business_main_text = business_df['article_content']
business_summary_arr2 = textrank(business_main_text)

entertainment_main_text = entertainment_df['article_content']
entertainment_summary_arr2 = textrank(entertainment_main_text)

politics_main_text = politics_df['article_content']
politics_summary_arr2 = textrank(politics_main_text)

sports_main_text = sports_df['article_content']
sports_summary_arr2 = textrank(sports_main_text)

tech_main_text = tech_df['article_content']
tech_summary_arr2 = textrank(tech_main_text)


# In[63]:


business_df['text_summary_using_textrank']=business_summary_arr2
business_df.head()


# In[64]:


entertainment_df['text_summary_using_textrank']=entertainment_summary_arr2
entertainment_df.head()


# In[65]:


politics_df['text_summary_using_textrank']=politics_summary_arr2
politics_df.head()


# In[66]:


sports_df['text_summary_using_textrank']=sports_summary_arr2
sports_df.head()


# In[67]:


tech_df['text_summary_using_textrank']=tech_summary_arr2
tech_df.head()


# In[68]:


#function for evaluation
def check_summary_quality(summary, article):
    #Tokenization
    summary_tokens = [word.lower() for word in word_tokenize(summary) if word.isalpha() and word.lower() not in stopwords.words('english')]
    
    #checking of speling errors
    spelling_errors = [word for word in summary_tokens if TextBlob(word).correct() != word]
    num_spelling_errors = len(spelling_errors)
    
    #calculate the polarity and subjectivity
    polarity, subjectivity = TextBlob(summary).sentiment    

    #calculate similarity
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    X = vectorizer.fit_transform([summary, article])
    sim_matrix = cosine_similarity(X)
    similarity = sim_matrix[0, 1]
    return {
        'num_spelling_errors': num_spelling_errors,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'similarity': similarity
    }


# In[ ]:


#replicate it accordingly 
res=check_summary_quality(summary_arr1[0],main_text[0])
res


# In[72]:


# Function for evaluating summary quality
def checking_quality(summary_arr1, summary_arr2, main_text):
    
    spelling_Error_LSA=[]
    spelling_Error_TextRank=[]
    polarity_LSA=[]
    polarity_TextRank=[]
    subjectivity_LSA=[]
    subjectivity_TextRank=[]
    similarity_LSA=[]
    similarity_TextRank=[]
    
    for i in range(200):
        res=check_summary_quality(summary_arr1[i],main_text[i])
        spelling_Error_LSA.append(res['num_spelling_errors'])
        polarity_LSA.append(res['polarity'])
        subjectivity_LSA.append(res['subjectivity'])
        similarity_LSA.append(res['similarity'])

    for i in range(200):
        res=check_summary_quality(summary_arr2[i],main_text[i])
        spelling_Error_TextRank.append(res['num_spelling_errors'])
        polarity_TextRank.append(res['polarity'])
        subjectivity_TextRank.append(res['subjectivity'])
        similarity_TextRank.append(res['similarity'])
        
    lsa_spelling=sum(spelling_Error_LSA) / len(spelling_Error_LSA)
    textrank_spelling=sum(spelling_Error_TextRank) / len(spelling_Error_TextRank)
    lsa_polarity=sum(polarity_LSA) / len(polarity_LSA)
    textrank_polarity=sum(polarity_TextRank) / len(polarity_TextRank)
    lsa_subjectivity=sum(subjectivity_LSA) / len(subjectivity_LSA)
    textrank_subjectivity=sum(subjectivity_TextRank) / len(subjectivity_TextRank)
    lsa_similarity=sum(similarity_LSA) / len(similarity_LSA)
    textrank_similarity=sum(similarity_TextRank) / len(similarity_TextRank)

    print('**********spelling Errors**********')
    print('Using LSA:- ',lsa_spelling)
    print('Using TextRank:- ',textrank_spelling)
    print('**********polarity**********')
    print('Using LSA:- ',lsa_polarity)
    print('Using TextRank:- ',textrank_polarity)
    print('**********subjectivity**********')
    print('Using LSA:- ',lsa_subjectivity)
    print('Using TextRank:- ',textrank_subjectivity)
    print('**********similarity**********')
    print('Using LSA:- ',lsa_similarity)
    print('Using TextRank:- ',textrank_similarity)

    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    spelling_errors = [lsa_spelling, textrank_spelling]
    labels = ['LSA', 'TextRank']
    axs[0, 0].bar(labels, spelling_errors)
    axs[0, 0].set_title('Spelling Errors')

    polarity = [lsa_polarity, textrank_polarity]
    axs[0, 1].bar(labels, polarity)
    axs[0, 1].set_title('Polarity')

    subjectivity = [lsa_subjectivity, textrank_subjectivity]
    axs[1, 0].bar(labels, subjectivity)
    axs[1, 0].set_title('Subjectivity')

    similarity = [lsa_similarity, textrank_similarity]
    axs[1, 1].bar(labels, similarity)
    axs[1, 1].set_title('Similarity')

    fig.suptitle('Summary Quality Evaluation')

    plt.show()


# In[74]:


print("Running for Business")
checking_quality(business_summary_arr1, business_summary_arr2, business_main_text)


# In[75]:


print("Running for entertainment")
checking_quality(entertainment_summary_arr1, entertainment_summary_arr2, entertainment_main_text)


# In[76]:


print("Running for politics")
checking_quality(politics_summary_arr1, politics_summary_arr2, politics_main_text)


# In[77]:


print("Running for sports")
checking_quality(sports_summary_arr1, sports_summary_arr2, sports_main_text)


# In[78]:


print("Running for tech")
checking_quality(tech_summary_arr1, tech_summary_arr2, tech_main_text)


# In[ ]:




