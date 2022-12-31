print("Ran by Shashank Khanna\n")
# importing the required libraries
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# opening the stopwords file
stopwords_file = open('stopwords_en.txt','r')
# requesting the data from the URL
nbc_business = "https://www.nbcnews.com/business"
res = requests.get(nbc_business)
soup = BeautifulSoup(res.content, 'html.parser')
# using the following h2 class to get all headlines with the h2 class
headlines = soup.find_all('h2',{'class':'wide-tease-item__headline'})

# putting the stopwords into a list
stopwords = []
for word in stopwords_file:
    stopwords.append(word.strip())

# using for loop to store each headline string into a list
print("The headlines are shown below :")
headline_list = []
for i in range(len(headlines)):
    print(headlines[i].string)
    headline_list.append((headlines[i].string))

# adding each sentences from the headlines to a array in the list
output_list = []
for sentence in headline_list:
    temp_list = []
    for word in sentence.split():
        if word.lower() not in stopwords:
            temp_list.append(word)
    output_list.append(' '.join(temp_list))
print('\nList of Headlines are shown below :')
print(output_list)

clean_words = []
word_list = []

# removing the stopwords to clean the words
for line in headline_list:
    item = line.strip()
    items = item.split()
    for word in items:
        if word.lower() not in stopwords and len(word.lower()) > 2:
            word_list.append(word.lower())

# removing all non alpha words from the clean words list
clean_words = [word.lower() for word in word_list if word.isalpha()]
print('\nList of clean words are shown below : \n',clean_words)


##### BIGRAM #####

# obtaining bigrams from clean text
headlines_bigrams = list(nltk.bigrams(clean_words))
print('\nThe retrieved bigrams from the text are listed below :')
print(headlines_bigrams)
# joining of bigrams in user defined format to create a list
print('\nThe formatted bigrams are listed below :')
bigram_list = ['_'.join(tups) for tups in headlines_bigrams]
print(bigram_list)

# merge the clean text words list and bi-gram list
final_bigram_list = clean_words + bigram_list
print('\nThe combined list of clean text and bigrams is shown below :')
print(final_bigram_list)
print("\n")


 ##### SENTIMENT #####

# Calculating sentiment of each headline and getting the highest to the lowest sentiment
print("\nSentiment analysis for the list of headlines :\n")
analyzer = SentimentIntensityAnalyzer()
#creating a temp dict to store the sentimental ratings
temp_dict = {}
#populating the temporary dictionary that we have created to contain the compound element
for clean_text in output_list:
    text_sentiment = analyzer.polarity_scores(clean_text)
    sentiment_rating = text_sentiment ['compound']
    temp_dict[clean_text] = sentiment_rating
#creating a sorted list with all the elements
sorted_sentiments = sorted(temp_dict.items(), key=lambda kv: (kv[1], kv[0]))
for i in output_list:
    sentiment = analyzer.polarity_scores(i)
# using sorted keyword for getting highest to lowest sentiment
    for key in sorted(sentiment):
        print('{0}: {1} '.format(key, sentiment[key]), end='')
# for the compound of a headline to be greater than 0.05 we find a positive sentiment
    if sentiment["compound"] >= 0.05:
        print("\n The headline : ",i," is POSITIVE\n")
# for the compound of a headline to be less than -0.05 we find a negative sentiment
    elif sentiment["compound"] <= -0.05:
        print("\n The headline : ",i," is NEGATIVE\n")
# for the compound of a headline to be between 0.05 and -0.05 we find a neutral sentiment
    else:
        print("\n The headline : ",i," is NEUTRAL\n")
#using the for loop to get only those headlines with compound element as negative in ascending order
print("\nTop 3 Headlines with the most negative sentiment and their respective compound element: ")
for headline in sorted_sentiments[:3]:
    print(headline[0:])
# reversing the sorted list
sorted_sentiments.reverse()
print("\nTop 3 Headlines with the most positive sentiment and their respective compound element: ")
for headline in sorted_sentiments[:3]:
    print(headline[0:])


##### WORDCLOUD #####

headline_str = ' '.join(final_bigram_list)
# defining the wordcloud parameters
wc = WordCloud(background_color = 'white', max_words=2000)
# generating word cloud
wc.generate(headline_str)
# storing to file
wc.to_file('text.png')
# showing the cloud
plt.imshow(wc)
plt.axis('off')
plt.title("WORD CLOUD")
plt.show()