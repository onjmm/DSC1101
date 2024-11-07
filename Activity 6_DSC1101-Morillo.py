#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import contractions
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data = pd.read_csv("C:\\Users\\Marco\\youtoxic_english_1000.csv")
sentences = data['Text']
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def expand_contractions(text):
    return contractions.fix(text)

nltk_results = []
for sentence in sentences:
    expanded_sentence = expand_contractions(sentence)
    tokens = word_tokenize(expanded_sentence) 
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    nltk_results.append((tokens, lemmas))

textblob_results = []
for sentence in sentences:
    expanded_sentence = expand_contractions(sentence)
    blob = TextBlob(expanded_sentence)
    tokens = blob.words
    lemmas = [word.lemmatize() for word in blob.words]
    textblob_results.append((tokens, lemmas))

nltk_token_count = Counter([token for tokens, _ in nltk_results for token in tokens])
nltk_lemma_count = Counter([lemma for _, lemmas in nltk_results for lemma in lemmas])
textblob_token_count = Counter([token for tokens, _ in textblob_results for token in tokens])
textblob_lemma_count = Counter([lemma for _, lemmas in textblob_results for lemma in lemmas])
avg_nltk_token_length = sum(len(token) for tokens, _ in nltk_results for token in tokens) / sum(len(tokens) for tokens, _ in nltk_results)
avg_nltk_lemma_length = sum(len(lemma) for _, lemmas in nltk_results for lemma in lemmas) / sum(len(lemmas) for _, lemmas in nltk_results)
avg_textblob_token_length = sum(len(token) for tokens, _ in textblob_results for token in tokens) / sum(len(tokens) for tokens, _ in textblob_results)
avg_textblob_lemma_length = sum(len(lemma) for _, lemmas in textblob_results for lemma in lemmas) / sum(len(lemmas) for _, lemmas in textblob_results)

textblob_sentiments = [TextBlob(sentence).sentiment for sentence in sentences]
nltk_sentiments = []
for sentence in sentences:
    expanded_sentence = expand_contractions(sentence)
    sentiment = sia.polarity_scores(expanded_sentence)
    nltk_sentiments.append(sentiment)
    
avg_textblob_polarity = sum(sentiment.polarity for sentiment in textblob_sentiments) / len(textblob_sentiments)
avg_textblob_subjectivity = sum(sentiment.subjectivity for sentiment in textblob_sentiments) / len(textblob_sentiments)

avg_nltk_compound = sum(sentiment['compound'] for sentiment in nltk_sentiments) / len(nltk_sentiments)


# In[39]:


for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print()


# In[40]:


for i, sentence in enumerate(sentences):
    print(f"NLTK Tokens: {nltk_results[i][0]}")
    print(f"TextBlob Tokens: {textblob_results[i][0]}")
    print()


# In[41]:


for i, sentence in enumerate(sentences):
    print(f"NLTK Lemmas: {nltk_results[i][1]}")
    print(f"TextBlob Lemmas: {textblob_results[i][1]}")
    print()


# In[42]:


for i, sentence in enumerate(sentences):
    print(f"TextBlob Sentiment (Polarity, Subjectivity): {textblob_sentiments[i]}")
    print(f"NLTK Sentiment (Positive, Neutral, Negative, Compound): {nltk_sentiments[i]}")
    print()


# In[43]:


print(f"NLTK Token frequency: {nltk_token_count}")
print(f"NLTK Lemma frequency: {nltk_lemma_count}")


# In[44]:


print(f"TextBlob Token frequency: {textblob_token_count}")
print(f"TextBlob Lemma frequency: {textblob_lemma_count}")


# In[45]:


print(f"Average NLTK token length: {avg_nltk_token_length}")
print(f"Average NLTK lemma length: {avg_nltk_lemma_length}")
print(f"Average TextBlob token length: {avg_textblob_token_length}")
print(f"Average TextBlob lemma length: {avg_textblob_lemma_length}")


# In[48]:


print(f"Average TextBlob Polarity: {avg_textblob_polarity}")
print(f"Average TextBlob Subjectivity: {avg_textblob_subjectivity}")
print(f"Average NLTK Compound Score: {avg_nltk_compound}")


# In[ ]:




