import numpy as np
import pandas

import requests
from bs4 import BeautifulSoup
import datetime

import nltk
from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews

def get_continuous_chunks(text):
    chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk

def word_feats(words):
    return dict([(word, True) for word in words])

countries_dataframe = pandas.read_csv("countries.csv", encoding = "ISO-8859-1")
countries_as_list_of_lists = countries_dataframe.values.tolist()
countries = []
for country_sub_array in countries_as_list_of_lists:
    countries.append(country_sub_array[0])

print "Start training classifier"
category_features = []
polarity_features = []
corpus = pandas.read_csv("categorized_article_corpus.csv", encoding = "ISO-8859-1")

article_category_corpus = corpus[['Article Body','Category']]
article_category_corpus_as_list = article_category_corpus.values.tolist()
for (article_body, category) in article_category_corpus_as_list: 
    category_features.append((word_feats(nltk.word_tokenize(article_body)), category))  
category_classifier = nltk.classify.NaiveBayesClassifier.train(category_features)
    
article_polarity_corpus = corpus[['Article Body','Polarity']]
article_polarity_corpus_as_list = article_polarity_corpus.values.tolist()
for (article_body, polarity) in article_polarity_corpus_as_list: 
    polarity_features.append((word_feats(nltk.word_tokenize(article_body)), polarity))  
polarity_classifier = nltk.classify.NaiveBayesClassifier.train(polarity_features)

# Movie reviews
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

trainfeats = negfeats[:] + posfeats[:]

movie_reviews_classifier = nltk.classify.NaiveBayesClassifier.train(trainfeats)

# Extraction of articles
print ("Start extraction of articles")
articles_page_request = requests.get("http://www.news.com.au/finance/business/breaking-news")
articles_page_content=articles_page_request.content
articles_page_html=BeautifulSoup(articles_page_content,"html.parser")

article_urls = []
for article in articles_page_html.find_all("div",{"class":"breaking-news-content"}):
    for article_url in article.find_all('a'):
        article_urls.append(article_url.get("href"))

results_summary = []
article_bodies = []
for article_url in article_urls:
    article_request=requests.get(article_url)
    article_content=article_request.content
    article_html=BeautifulSoup(article_content,"html.parser")
    
    # Extraction of article date
    article_date_array=article_html.find_all("span",{"class":"datestamp"})
    article_date = str(article_html.find("span",{"class":"datestamp"}).text).replace(str(article_html.find("span",{"class":"time"}).text), "")
    print("Article date: " + article_date)

    # Extraction of article title
    article_title = article_html.title.text
    print("Article title: " + article_title)
    identified_entities = get_continuous_chunks(article_title)

    # Extraction of article body
    article_body_array = []
    for article_body in article_html.find_all("div",{"class":"story-content"}):
        article_paragraphs = article_body.find_all('p')
        for paragraph in article_paragraphs:
            article_body_array.append(paragraph.text)
    
    article_body_sentences = map(str, article_body_array)
    article_body = " ".join(article_body_array)
    
    # Breakdown article body sentences into words
    article_body_words = []
    for article_body_sentence in article_body_sentences:
        article_body_words.extend(nltk.word_tokenize(article_body_sentence))
    
    # Text processing of article body words
    word_net_lemmatizer = WordNetLemmatizer()
    article_body_words = [t for t in article_body_words if len(t) > 2] # remove short words, they're probably not useful
    article_body_words = [word_net_lemmatizer.lemmatize(t) for t in article_body_words] # put words into base form
    article_body_words = [t for t in article_body_words if t not in stopwords.words("english")] # remove stopwords
    
    # Sentiment analysis of article body (sentence)
    sid = SentimentIntensityAnalyzer()
    summarized_polarity_score = 0
    for sentence in article_body_sentences:
        print(sentence)
        polarity_scores = sid.polarity_scores(sentence)
        for sentence in sorted(polarity_scores):
            print('{0}: {1}, '.format(sentence, polarity_scores[sentence]))
            summarized_polarity_score += polarity_scores["compound"]
    if summarized_polarity_score > 0:
        polarity = "Positive"
    elif summarized_polarity_score < 0:
        polarity = "Negative"
    else:
        polarity = "Neutral"
    print()
      
    # Sentiment analysis of article body (Naive Bayes)
    classified_polarity = polarity_classifier.classify(word_feats(nltk.word_tokenize(article_body)))
       
    # Sentiment analysis of article body (Movie Reviews)
    movie_reviews_classified_polarity = movie_reviews_classifier.classify(word_feats(article_body_words))
    if movie_reviews_classified_polarity == "pos":
        movie_reviews_classified_polarity = "Positive"
    else:
        movie_reviews_classified_polarity = "Negative"
    
    # Classifying category of article
    classified_category = category_classifier.classify(word_feats(nltk.word_tokenize(article_body)))
    print (article_title, category)
    
    # Classifying geo-impact of article
    country_count = 0
    relevant_countries_array = []
    for country in countries:
        if country in article_body:
            country_count += 1
            relevant_countries_array.append(country)
    
    if country_count == 0:
        relevant_country = "Australia"
    elif country_count > 0:
        for relevant_countries_element in relevant_countries_array:
            if relevant_countries_element == "US" or relevant_countries_element == "USA" or relevant_countries_element == "America":
                relevant_countries_element = "United States"
        relevant_countries = ','.join(map(str, relevant_countries_array)) 
    
    # Summarizing article features
    for identified_entity in identified_entities:
        result_summary = [article_html.title.text, article_date, identified_entity, summarized_polarity_score, polarity, classified_polarity, movie_reviews_classified_polarity, classified_category, relevant_countries if country_count > 0 else relevant_country]
        results_summary.append(result_summary)
        print("Based on this sentence, " + identified_entity + " has been assigned a sentiment polarity score of:", summarized_polarity_score)
            
    """
    capitalized_nouns = []
    for tagged_word_pair in tagged_words:
        for word, label in tagged_word_pair:
            print(word, label)
            if word[0].isupper() and (label == "NN" or label == "NNP"or label == "NNS"):
                #print(word, label)
                capitalized_nouns.append(word)
                
    count = Counter(capitalized_nouns)
    related_entity = count.most_common(1)[0][0]
    """

print(nltk.classify.util.accuracy(polarity_classifier, polarity_features))
polarity_classifier.show_most_informative_features()

print(nltk.classify.util.accuracy(movie_reviews_classifier, trainfeats))
movie_reviews_classifier.show_most_informative_features()
    
results_summary = np.array(results_summary)
news_dot_com_data_table = pandas.DataFrame(results_summary, columns=['Article Title', "Article Date", 'Entity', "Sentiment Polarity Score (VADER)", "Sentiment Polarity (VADER)", "Sentiment Polarity (Naive Bayes with Article Corpus)", "Sentiment Polarity (Naive Bayes with Movie Reviews)", "Category", "Countries Affected"])
news_dot_com_data_table.to_csv("news_dot_com_data " + str(datetime.datetime.now())[:-16] + ".csv", index=False)

