print("Ran by Shashank Khanna\n")

# Importing the required libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# reading input files
pros_file = open('ex07_Pros.txt','r')
cons_file = open('ex07_Cons.txt','r')
stopwords_file = open('stopwords_en.txt','r')

# initializing lists
stopwords = []
pros_words = []
cons_words = []

# create a list of stopwords which contain the obviously frequently used words in both the pros and cons.
additional_pros_stopwords = ['dress', 'codes', 'code']
additional_cons_stopwords = ['dress', 'codes']

# populating the list of stopwords
for word in stopwords_file:
    stopwords.append(word.strip())

# cleaning the text input files
for pros_file, word_list in [(pros_file, pros_words)]:
    for line in pros_file:
        parts = line.strip().split()
        for word in parts:
            if word.lower() not in stopwords and len(word.lower()) > 2:
                word_list.append(word.lower())
            for text in list(pros_words):
                if text in additional_pros_stopwords:
                    pros_words.remove(text)
for cons_file, word_list in [(cons_file, cons_words)]:
    for line in cons_file:
        parts = line.strip().split()
        for word in parts:
            if word.lower() not in stopwords and len(word.lower()) > 2:
                word_list.append(word.lower())
                for text in list(cons_words):
                    if text in additional_cons_stopwords:
                        cons_words.remove(text)

# Eliminate non alpha elements
text_list1 = [word.lower() for word in pros_words if word.isalpha()]
text_list2 = [word.lower() for word in cons_words if word.isalpha()]

# final clean text for pros and cons
print("Pros final clean text list:\n",text_list1,"\n")
print("Cons final clean text list:\n",text_list2,"\n")


##### BIGRAM #####
# using the final lists of pros and cons to generate the bigrams

bigram_pros = list(nltk.bigrams(text_list1))
print('\nList of Bigrams extracted from the PROS file:')
print(bigram_pros)

bigram_cons = list(nltk.bigrams(text_list2))
print('\nList of Bigrams extracted from the CONS file:')
print(bigram_cons,"\n")


##### SENTIMENT #####

analyzer = SentimentIntensityAnalyzer()

# converting the final list to a string for display
text_str1 = ' '.join(text_list1)
text_str2 = ' '.join(text_list2)

# finding out the polarity scores
sentiment_pros = analyzer.polarity_scores(text_str1)
sentiment_cons = analyzer.polarity_scores(text_str2)

# polarity score for the pros string
pos_pros = sentiment_pros['pos']
neg_pros = sentiment_pros['neg']
neu_pros = sentiment_pros['neu']

#polarity score for the cons string
pos_cons = sentiment_cons['pos']
neg_cons = sentiment_cons['neg']
neu_cons = sentiment_cons['neu']

print('\nThe following is the Sentiment Analysis for the PROS file:')
print('\nPositive:', '{:.1%}'.format(pos_pros))
print('\nNegative:', '{:.1%}'.format(neg_pros))
print('\nNeutral:', '{:.1%}'.format(neu_pros))

print('\nThe following is the Sentiment Analysis for the CONS file:')
print('\nPositive:', '{:.1%}'.format(pos_cons))
print('\nNegative:', '{:.1%}'.format(neg_cons))
print('\nNeutral:', '{:.1%}'.format(neu_cons), '\n')


##### WORDCLOUD #####

# defining the wordcloud parameters
wc = WordCloud(background_color = 'white', max_words=5000)
# generating word cloud for Pros
wc.generate(text_str1)
# storing to file
wc.to_file('txt1_pros.png')
# showing the cloud
plt.imshow(wc)
plt.axis('off')
plt.title("PROS WORD CLOUD")
plt.show()
#generating the world cloud for Cons
wc.generate(text_str2)
# storing to file
wc.to_file('txt2_cons.png')
# showing the cloud
plt.imshow(wc)
plt.axis('off')
plt.title("CONS WORD CLOUD")
plt.show()