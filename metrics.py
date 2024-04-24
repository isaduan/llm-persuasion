# import all packages
import nltk
from collections import Counter
from convokit import Corpus, download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import itertools
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from convokit.model import utterance
from convokit.model import speaker
from convokit import coordination
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from langchain.embeddings import GPT4AllEmbeddings

# define function word categories
tag_to_category = {
    'DT': 'Determiners',  
    'MD': 'Auxiliary verbs',
    'CC': 'Conjunctions',
    'IN': 'Prepositions or Subordinating Conjunction',
    'RB': "Adverbs",
    'PRP': "Personal pronouns",
    'TO': 'To',
    'RP': 'Particles',
    'CD': 'Cardinal number',
    'PDT': 'Predeterminer',
    'UH':'Interjection',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WRB': 'Wh-adverb'
}

# download bert and gpt4all models to be used in method 8 and 9
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt4all_embd = GPT4AllEmbeddings()

# method 1
def calculate_function_word_cat_frequency(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1)
    tokens2 = nltk.word_tokenize(utterance2)

    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)

    # identify and categorize functional words
    categories1 = [tag_to_category.get(tag, 'Other') for word, tag in pos_tags1 if tag in tag_to_category]
    categories2 = [tag_to_category.get(tag, 'Other') for word, tag in pos_tags2 if tag in tag_to_category]

    counts1 = Counter(categories1)
    counts2 = Counter(categories2)

    total_words1 = len(tokens1)
    total_words2 = len(tokens2)

    freq1 = {cat: count / total_words1 for cat, count in counts1.items()}
    freq2 = {cat: count / total_words2 for cat, count in counts2.items()}

    common_categories = set(freq1.keys()).intersection(set(freq2.keys()))

    category_differences = {}

    if not common_categories:
        return 0  # no common categories

    coordination_scores = []
    for cat in common_categories:
        diff = abs(freq1[cat] - freq2[cat])
        coordination_scores.append(diff)
        category_differences[cat] = diff

    overall_coordination_score = sum(coordination_scores) / len(common_categories) if common_categories else 0

    return 1 - overall_coordination_score


def identify_function_words(tokens):
    """
    Helper Function to identify function words based on POS tags
    """
    pos_tags = nltk.pos_tag(tokens)
    # function words typically fall under these POS tags
    function_word_tags = ['CC', 'DT', 'IN', 'RB', 'MD', 'PDT', 'PRP', 'RP', 'CD',
                          'UH', 'TO', 'WDT', 'WP', 'WRB']
    return [word for word, tag in pos_tags if tag in function_word_tags]

# method 2
def calculate_function_word_frequency(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1.lower())
    tokens2 = nltk.word_tokenize(utterance2.lower())

    func_words1 = identify_function_words(tokens1)
    func_words2 = identify_function_words(tokens2)

    counts1 = Counter(func_words1)
    counts2 = Counter(func_words2)

    total_words1 = len(tokens1)
    total_words2 = len(tokens2)

    freq1 = {word: count / total_words1 for word, count in counts1.items()}
    freq2 = {word: count / total_words2 for word, count in counts2.items()}

    common_categories = set(freq1.keys()).intersection(set(freq2.keys()))
  
    category_differences = {}

    if not common_categories:
        return 0  # no common categories

    coordination_scores = []
    for cat in common_categories:
        diff = abs(freq1[cat] - freq2[cat])
        coordination_scores.append(diff)
        category_differences[cat] = diff

    overall_coordination_score = sum(coordination_scores) / len(common_categories) if common_categories else 0

    return 1 - overall_coordination_score

# method 3
def calculate_function_word_count(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1.lower())
    tokens2 = nltk.word_tokenize(utterance2.lower())

    func_words1 = identify_function_words(tokens1)
    func_words2 = identify_function_words(tokens2)

    counts1 = Counter(func_words1)
    counts2 = Counter(func_words2)

    common_categories = set(counts1.keys()).intersection(set(counts2.keys()))

    category_differences = {}

    if not common_categories:
        return 0  # no common categories

    coordination_scores = []
    for cat in common_categories:
        diff = abs(counts1[cat] - counts2[cat])
        coordination_scores.append(diff)
        category_differences[cat] = diff

    overall_coordination_score = sum(coordination_scores) / len(common_categories) if common_categories else 0

    if overall_coordination_score == 0:
      return 1

    return 1 / overall_coordination_score

# method 4
def calculate_function_word_cat_count(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1)
    tokens2 = nltk.word_tokenize(utterance2)

    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)

    categories1 = [tag_to_category.get(tag, 'Other') for word, tag in pos_tags1 if tag in tag_to_category]
    categories2 = [tag_to_category.get(tag, 'Other') for word, tag in pos_tags2 if tag in tag_to_category]

    counts1 = Counter(categories1)
    counts2 = Counter(categories2)

    common_categories = set(counts1.keys()).intersection(set(counts2.keys()))

    category_differences = {}

    if not common_categories:
        return 0  # no common categories

    coordination_scores = []
    for cat in common_categories:
        diff = abs(counts1[cat] - counts2[cat])
        coordination_scores.append(diff)
        category_differences[cat] = diff

    overall_coordination_score = sum(coordination_scores) / len(common_categories) if common_categories else 0

    if overall_coordination_score == 0:
      return 1

    return 1 / overall_coordination_score

# method 5
def calculate_function_word_cat_presence(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1)
    tokens2 = nltk.word_tokenize(utterance2)

    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)

    categories1 = set(tag_to_category.get(tag, 'Other') for word, tag in pos_tags1 if tag in tag_to_category)
    categories2 = set(tag_to_category.get(tag, 'Other') for word, tag in pos_tags2 if tag in tag_to_category)

    common_categories = categories1.intersection(categories2)
    total_categories = categories1.union(categories2)

    if not total_categories:
        return 0 # avoid division by zero

    coordination_score = len(common_categories) / len(total_categories)

    # identify which categories are common
    category_presence = {cat: cat in common_categories for cat in total_categories}

    return coordination_score

# method 6
def calculate_tfidf_coordination_score(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1.lower())
    tokens2 = nltk.word_tokenize(utterance2.lower())

    text1 = ' '.join(tokens1)
    text2 = ' '.join(tokens2)

    vectorizer = TfidfVectorizer()
    try:
      tfidf_matrix = vectorizer.fit_transform([text1, text2])

    except:
      return None

    coordination_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return coordination_score

# method 7
def calculate_count_vectorizer_coordination_score(utterance1: str, utterance2: str):
    tokens1 = nltk.word_tokenize(utterance1.lower())
    tokens2 = nltk.word_tokenize(utterance2.lower())

    text1 = ' '.join(tokens1)
    text2 = ' '.join(tokens2)

    vectorizer = CountVectorizer()
    try:
      count_matrix = vectorizer.fit_transform([text1, text2])

    except:
      return None

    coordination_score = cosine_similarity(count_matrix[0:1], count_matrix[1:2])[0][0]

    return coordination_score

# method 8
def calculate_sbert_coordination_score(utterance1: str, utterance2: str, bert_model=bert_model):
    embedding1 = bert_model.encode(utterance1)
    embedding2 = bert_model.encode(utterance2)

    coordination_score = cosine_similarity([embedding1], [embedding2])[0][0]

    return coordination_score

# method 9
def calculate_gpt4all_coordination_score(utterance1: str, utterance2: str, gpt4all_embd=gpt4all_embd):
    try:
      embedding1 = gpt4all_embd.embed_query(utterance1)
      embedding2 = gpt4all_embd.embed_query(utterance2)
    except:
      print('something wrong')
      return None

    coordination_score = cosine_similarity([embedding1], [embedding2])[0][0]

    return coordination_score

def are_subreddits_similar(subreddit1, subreddit2, threshold=0.5):
    # Convert subreddit names to lowercase to match the format in subreddit_similarity_df
    subreddit1 = subreddit1.lower()
    subreddit2 = subreddit2.lower()

    # Check if the similarity value exists in the DataFrame
    if subreddit1 in subreddit_similarity_df.index and subreddit2 in subreddit_similarity_df.columns:
        similarity = subreddit_similarity_df.loc[subreddit1, subreddit2]
        return similarity >= threshold
    else:
        # Return False or some default value if the subreddits are not in the similarity DataFrame
        return False

def prepare_corpus():
    # download reddit data and read it as a dataframe
    reddit = Corpus(filename=download("reddit-corpus-small"))
    df = reddit.get_utterances_dataframe()
  
    # read similarity matrix datafrmame
    subreddit_similarity_df = pd.read_csv('/similarity_matrix.csv', index_col=0)
  
    # define number of samples
    num_positive_samples = 2000
    num_negative_samples = 2000
  
    # sample positive pairs
    positive_pairs = []
    grouped = df.groupby('speaker')
    num_groups = len(grouped)
    samples_per_group = max(num_positive_samples // num_groups, 1)  # ensure at least one sample per group
  
    # sample negative pairs
    negative_pairs = []
    for _ in range(num_negative_samples // 2):  # half from similar subreddits
        pair = df.sample(2)
        while pair.iloc[0]['speaker'] == pair.iloc[1]['speaker'] or not are_subreddits_similar(pair.iloc[0]['meta.subreddit'], pair.iloc[1]['meta.subreddit']):
            pair = df.sample(2)
        negative_pairs.append((pair.iloc[0]['text'], pair.iloc[1]['text'], 0, 1))
    
    for _ in range(num_negative_samples // 2):  # half from dissimilar subreddits
        pair = df.sample(2)
        while pair.iloc[0]['speaker'] == pair.iloc[1]['speaker'] or are_subreddits_similar(pair.iloc[0]['meta.subreddit'], pair.iloc[1]['meta.subreddit']):
            pair = df.sample(2)
        negative_pairs.append((pair.iloc[0]['text'], pair.iloc[1]['text'], 0, 0))

    # combine, shuffle, and split the dataset
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    dataset = pd.DataFrame(all_pairs, columns=['Utterance1', 'Utterance2', 'SameSpeakerLabel', 'SimilarSubredditLabel'])

    return dataset

def train_and_evaluate_all(datasets, feature_functions, label_column='SameSpeakerLabel'):
    results = {}

    for feature_function in feature_functions:
        print(f"Training and evaluating with {feature_function.__name__}...")

        # Calculate features using the current function
        features = [feature_function(row['Utterance1'], row['Utterance2']) for _, row in datasets.iterrows()]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(features, datasets[label_column], test_size=0.3, random_state=42)

        # Impute missing values
        imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
        X_train_imputed = imputer.fit_transform(np.array(X_train).reshape(-1,1))
        X_test_imputed = imputer.transform(np.array(X_test).reshape(-1,1))

        # Create and train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train_imputed, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_imputed)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # Store results
        results[feature_function.__name__] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    return results

def run_train_and_evaluate_all():
  dataset = prepare_corpus()
  similar_dataset = dataset[dataset['SimilarSubredditLabel'] == 1]
  dissimilar_dataset = dataset[dataset['SimilarSubredditLabel'] == 0]
  
  feature_functions = [calculate_function_word_cat_frequency,
                     calculate_function_word_frequency,
                     calculate_function_word_count,
                     calculate_function_word_cat_count,
                     calculate_function_word_cat_presence,
                     calculate_tfidf_coordination_score,
                     calculate_count_vectorizer_coordination_score,
                     calculate_sbert_coordination_score,
                     calculate_gpt4all_coordination_score]
  
  results_similar = train_and_evaluate_all(similar_dataset, feature_functions)
  results_dissimilar = train_and_evaluate_all(dissimilar_dataset, feature_functions)
  # store results in JSON files
  with open('results_similar.json', 'w') as similar_file:
      json.dump(results_similar, similar_file, indent=4)
  with open('results_dissimilar.json', 'w') as dissimilar_file:
      json.dump(results_dissimilar, dissimilar_file, indent=4)

if __name__ == "__main__":
    run_train_and_evaluate_all()
