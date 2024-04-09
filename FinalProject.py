from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
import nltk
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import base64
from io import BytesIO
import networkx as nx


app = Flask(__name__)

# Function to load stopwords from file
def load_stopwords(file_path):
    with open(file_path, "r") as file:
        stopwords = set(file.read().splitlines())
    return stopwords

stopwords = load_stopwords("stopwords_en.txt")

def remove_stopwords(text, stopwords):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

# Function to scrape news article and remove stopwords
def scrape_news_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            cont = ' '.join(para.get_text() for para in paragraphs)
            # Remove stopwords from the content
            content = remove_stopwords(cont, stopwords)
            return content

        else:
            return None
    except Exception as e:
        print(f"An error occurred while scraping the article: {e}")
        return None

# Function to summarize text
def summarize_text(text, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    max_chunk = 1024
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summary = []
    for chunk in text_chunks:
        max_len = min(150, max(50, len(chunk.split()) // 2))
        summary_part = summarizer(chunk, max_length=max_len, min_length=50, do_sample=False)[0]['summary_text']
        summary.append(summary_part)
    return summary


def sentiment_analysis(text, model_name="siebert/sentiment-roberta-large-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    max_length = tokenizer.model_max_length - 2  # leave space for special tokens like [CLS] and [SEP]

    # Split the text into chunks that are smaller than the max_length
    def chunk_text(text, max_length):
        tokens = tokenizer.tokenize(text)
        for i in range(0, len(tokens), max_length):
            yield tokenizer.convert_tokens_to_string(tokens[i:i + max_length])

    # Process each text chunk
    sentiments = []
    for chunk in chunk_text(text, max_length):
        analysis = sentiment_pipeline(chunk)
        sentiments.extend(analysis)

    # Combine or average the sentiments as needed
    # ...

    return sentiments

# Function to extract keywords and perform sentiment analysis on summaries
def process_summaries(summaries):
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5)  # Adjust number of keywords as needed
    X = vectorizer.fit_transform(summaries)
    keywords_array = vectorizer.get_feature_names_out()

    processed_summaries = []
    for idx, summary in enumerate(summaries):
        summary_keywords = ', '.join(keywords_array[X[idx].toarray()[0].argsort()[-3:][::-1]])  # Top 3 keywords
        sentiment_result = sentiment_analysis(summary)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']
        processed_summaries.append({
            'keywords': summary_keywords,
            'summary': summary,
            'sentiment': f"{sentiment_label} ({sentiment_score:.3f})"
        })
    return processed_summaries


# Function to generate a word cloud and return as base64 encoded image
def generate_wordcloud_base64(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        collocations=False,  # This is crucial for ngram word clouds
        min_font_size=10,  # Can be adjusted
        max_font_size=150,  # Can be adjusted
        scale=2  # Can be adjusted
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save image to a BytesIO object
    img_io = BytesIO()
    plt.savefig(img_io, format='PNG', bbox_inches='tight')
    plt.close()

    # Encode as base64
    img_io.seek(0)
    base64_img = base64.b64encode(img_io.getvalue()).decode('utf8')
    return base64_img

# Function to get top N words
def get_top_n_words(text, n=10):
    vectorizer = CountVectorizer(stop_words=stopwords)
    word_count = vectorizer.fit_transform([text])
    sum_words = word_count.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Function to generate ngrams from text
def generate_ngrams(tokens, n):
    # tokens = nltk.word_tokenize(text)
    return ['_'.join(gram) for gram in ngrams(tokens, n)]

# Function to generate a word cloud from ngrams
def generate_ngram_wordcloud(ngrams):
    ngram_text = ' '.join(ngrams)
    return generate_wordcloud_base64(ngram_text)  # Reuse the existing word cloud function

def replace_ngrams(tokens, bigrams, trigrams):
    # Create a single string from the tokens for easier replacement
    text = ' '.join(tokens)

    # Replace each bigram with its concatenated version (with underscore)
    for bigram in bigrams:
        bigram_combined = '_'.join(bigram)
        bigram_single = ' '.join(bigram)
        text = text.replace(bigram_single, bigram_combined)

    # Replace each trigram with its concatenated version (with underscore)
    for trigram in trigrams:
        trigram_combined = '_'.join(trigram)
        trigram_single = ' '.join(trigram)
        text = text.replace(trigram_single, trigram_combined)

    # Split the text back into tokens
    new_tokens = text.split()

    return new_tokens

# Function to process summaries with cluster IDs and extract keywords for each summary
def process_summaries_with_clusters(summaries, clustered_summaries):
    # Create a mapping from summary to its cluster ID
    summary_to_cluster = {}
    for cluster_id, cluster_summaries in clustered_summaries.items():
        for summary in cluster_summaries:
            summary_to_cluster[summary] = cluster_id

    processed_data = []
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5)  # Adjust the number of features as needed

    # Fit the vectorizer to the summaries and transform
    X = vectorizer.fit_transform(summaries)
    keywords_array = vectorizer.get_feature_names_out()

    for idx, summary in enumerate(summaries):
        cluster_id = summary_to_cluster.get(summary, "N/A")
        sentiment_result = sentiment_analysis(summary)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        # Extract the top keywords for this summary
        top_keywords = ', '.join(keywords_array[X[idx].toarray()[0].argsort()[-3:][::-1]])  # Top 3 keywords

        processed_data.append({
            'cluster_id': f"Cluster {cluster_id}",
            'keywords': top_keywords,
            'summary': summary,
            'sentiment': f"{sentiment_label} ({sentiment_score:.3f})"
        })
    return processed_data


def cluster_summaries(summaries, n_clusters=5):
    # Make sure there are enough summaries to form the desired number of clusters
    n_clusters = min(n_clusters, len(summaries))

    if n_clusters < 1:
        raise ValueError("Number of clusters must be at least 1.")

    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(summaries)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    clustered_summaries = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clustered_summaries:
            clustered_summaries[label] = []
        clustered_summaries[label].append(summaries[i])
    return clustered_summaries

# Adjusted Function to Format Clustered Summaries
def format_clustered_summaries(clustered_summaries):
    formatted_clusters = []
    for cluster_id, summaries in clustered_summaries.items():
        cluster_summary = ' '.join(summaries)
        top_words_tuples = get_top_n_words(cluster_summary, 3)  # This returns a list of tuples
        cluster_keywords = ', '.join([word for word, freq in top_words_tuples])  # Extract just the words
        formatted_clusters.append({
            'cluster_id': cluster_id,
            'summaries': summaries,
            'keywords': cluster_keywords
        })
    return formatted_clusters

def generate_cluster_graph_base64(clustered_summaries):
    G = nx.Graph()

    # Define a base size for all nodes
    base_size = 300

    def get_cluster_keywords(summaries, n=3):
        # Combine all summaries in the cluster into a single text
        combined_text = ' '.join(summaries)
        # Extract top N keywords (assuming get_top_n_words returns [(word, frequency), ...])
        top_keywords = get_top_n_words(combined_text, n)
        # Extract only the words, not the frequencies
        top_words = [word for word, freq in top_keywords]
        return ', '.join(top_words)

    # Add cluster nodes and keyword nodes to the graph
    for cluster_id, summaries in clustered_summaries.items():
        cluster_node = f"Cluster {cluster_id}"
        keywords = get_cluster_keywords(summaries)
        keyword_node = f"Keywords: {keywords}"

        G.add_node(cluster_node, size=base_size)
        G.add_node(keyword_node, size=base_size / 2)  # Adjust size for keyword nodes
        G.add_edge(cluster_node, keyword_node)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black',
            linewidths=1, font_size=10,
            nodelist=G.nodes(),
            node_size=[G.nodes[node]['size'] for node in G.nodes()])

    # Save image to a BytesIO object
    img_io = BytesIO()
    plt.savefig(img_io, format='PNG', bbox_inches='tight')
    plt.close()

    # Encode as base64
    img_io.seek(0)
    base64_img = base64.b64encode(img_io.getvalue()).decode('utf8')
    return base64_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        article_content = scrape_news_article(url)
        tokens = article_content.split()
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords]

        if article_content:
            summaries = summarize_text(article_content)
            #processed_summaries = process_summaries(summaries)
            combined_summary = ' '.join(summaries)
            average_sentiment = sentiment_analysis(combined_summary)[0]['label']

            top_words = get_top_n_words(article_content)
            top_words_str = ', '.join([word for word, freq in top_words])  # Convert to string for HTML display

            # Generate ngram word clouds
            bigrams = generate_ngrams(tokens, 2)
            trigrams = generate_ngrams(tokens, 3)
            # Generate base64 encoded word clouds
            wordcloud_image = generate_wordcloud_base64(article_content) # Ensure this returns base64 encoded image

            tokens = replace_ngrams(tokens, bigrams, trigrams)
            bigram_string = ' '.join(bigrams)

            bigram_wordcloud = generate_wordcloud_base64(bigram_string)
            trigram_string = ' '.join(trigrams)

            trigram_wordcloud = generate_wordcloud_base64(trigram_string)

            clustered_summaries = cluster_summaries(summaries)  # Ensure clusters are processed for display
            raw_clustered_summaries = cluster_summaries(summaries)
            formatted_clustered_summaries = format_clustered_summaries(raw_clustered_summaries)
            cluster_graph_base64 = generate_cluster_graph_base64(clustered_summaries)

            # Prepare display_data as per HTML template structure
            display_data = process_summaries_with_clusters(summaries, clustered_summaries)

            return render_template('results.html', display_data=display_data,
                                   combined_summary=combined_summary, average_sentiment=average_sentiment,
                                   wordcloud_base64=wordcloud_image,
                                   bigram_wordcloud_base64=bigram_wordcloud,
                                   trigram_wordcloud_base64=trigram_wordcloud, top_words=top_words_str,
                                   cluster_graph_base64=cluster_graph_base64)

        else:
            return render_template('index.html', error_message="Failed to retrieve or process the article.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
