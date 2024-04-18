import metrics
from convokit import Corpus, download
from sklearn.preprocessing import MinMaxScaler
# download bert and gpt4all models to be used in method 8 and 9
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt4all_embd = GPT4AllEmbeddings()

import openai
from openai import OpenAI
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="ENTER YOUR KEY",
)


def get_any_speaker(corpus, num_utterance, character_limit, df_requirement):
  """
  Input: Corpus object
  Output:
    Two lists of utterances, with texts, speaker, and reply-to info.
    One stores "select utterance," the other "utterance they reply to."
  """

  ids = corpus.get_conversation_ids()
  select_utts = []
  reply_utts = []
  record = 0

  while record < num_utterance:
    select_utt = corpus.random_utterance()
    reply_id = select_utt.reply_to
    sample_speaker = select_utt.get_speaker()
    texts_df = sample_speaker.get_utterances_dataframe()
    try:
      reply_utt = corpus.get_utterance(reply_id)
      reply_speaker = reply_utt.get_speaker()
      reply_df = reply_speaker.get_utterances_dataframe()
      if len(select_utt.text) >= character_limit and select_utt.reply_to and len(reply_utt.text) >= character_limit \
      and len(texts_df) >= df_requirement and len(reply_df) >= df_requirement:
        record += 1
        select_utts.append(select_utt)
        reply_utts.append(reply_utt)
    except:
      continue
    if record == num_utterance:
      break
  return select_utts, reply_utts

def llm_style_match(utts_lst, model_name, prompt):
  """
  Input: list of uttereance objects.
  Output:
    A list of utterance objects, each one is created by LLM to match the original
    utterance.
  """
  LLM = speaker.Speaker(id=model_name)
  LLM_utt_id = 0
  LLM_utt_lst = []
  for i, utt in enumerate(utts_lst):
    target_text = utt.text
    total_prompt = prompt + "Texts: " + target_text
    llm_time = utt.timestamp+1

    # API call
    response = client.chat.completions.create(
    model= model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled at generating persuasive messages,"},
        {"role": "user", "content": "{}".format(total_prompt)}
    ] # can also set max length
  )
    response_text = response.choices[0].message.content
    print(response_text)
    LLM_utt_id += 1

    # construct a utterance object
    utt_LLM = utterance.Utterance(speaker=LLM,
                               id = '{}'.format(LLM_utt_id),
                               reply_to = '{}'.format(utt.id),
                               text='{}'.format(response_text),
                               timestamp=llm_time
                               )
    LLM_utt_lst.append(utt_LLM)

  return LLM_utt_lst

def get_coordination_scores(utterances1, utterances2):
    if len(utterances1) != len(utterances2):
        raise ValueError("The lists of utterances must be of the same length")

    scores = []
    for u1, u2 in zip(utterances1, utterances2):
        score = {
            'cat_freq': metrics.calculate_function_word_cat_frequency(u1.text, u2.text),
            'exact_word_freq': metrics.calculate_function_word_frequency(u1.text, u2.text),
            'exact_word_count': metrics.calculate_function_word_count(u1.text, u2.text),
            'cat_count': metrics.calculate_function_word_cat_count(u1.text, u2.text),
            'cat_boolean': metrics.calculate_function_word_cat_presence(u1.text, u2.text),
            'tfidf': metrics.calculate_tfidf_coordination_score(u1.text, u2.text),
            'count_vectorizer': metrics.calculate_count_vectorizer_coordination_score(u1.text, u2.text),
            'sent_bert': metrics.calculate_sbert_coordination_score(u1.text, u2.text, bert_model),
            'GP4all': metrics.calculate_gpt4all_coordination_score(u1.text, u2.text, gpt4all_embd)
        }
        scores.append(score)

    return scores

def transform_values(scores_lst):
  values = [[d[key] for key in d] for d in scores_lst]
  scaler = MinMaxScaler()
  scaler.fit(values)
  transformed_values = scaler.transform(values)
  for i, d in enumerate(scores_lst):
    for j, key in enumerate(d):
        d[key] = transformed_values[i][j]

  return scores_lst

def calculate_average_scores(score_list):
    if not score_list:
        return {}
    total_scores = {}
    for score_dict in score_list:
        for method, score in score_dict.items():
            if method not in total_scores:
                total_scores[method] = 0
            total_scores[method] += score
    average_scores = {method: total / len(score_list) for method, total in total_scores.items()}

    return average_scores

def llm_rewrite_replies(select_utts, reply_utts, model_name, prompt):
  """
  Input: list of uttereance objects.
  Output:
    A list of utterance objects, each one is created by LLM to rewrite select_utts to match the original
    utterance, reply_utts.
  """
  LLM_rewrite = speaker.Speaker(id=model_name+"_rewrite")
  LLM_utt_id = 1000000
  LLM_reply_lst = []
  for i, utt in enumerate(reply_utts):
    target_text = utt.text # original texts
    rewrite_text = select_utts[i].text # texts to be rewritten
    total_prompt = prompt + "Texts to be rewrite: "+ rewrite_text + "/n" + "Original Texts: " + target_text
    llm_time = utt.timestamp+1

    # API call
    response = client.chat.completions.create(
    model= model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled at generating persuasive messages,"},
        {"role": "user", "content": "{}".format(total_prompt)}
    ] # can also set max length
  )
    response_text = response.choices[0].message.content
    print(response_text)
    LLM_utt_id += 1

    # construct a utterance object
    utt_LLM = utterance.Utterance(speaker=LLM_rewrite,
                               id = '{}'.format(LLM_utt_id),
                               reply_to = '{}'.format(utt.id),
                               text='{}'.format(response_text),
                               timestamp=llm_time
                               )
    LLM_reply_lst.append(utt_LLM)

  return LLM_reply_lst

specific_mc_prompt = """In no more than 100 words, craft me an argument that the first fully automated McDonalds will open in the United States
before 2030. Use function words (i.e., article, auxiliary verb, conjunction, indefinite pronouns, adverb,
prepositional pronouns, prepositions, and quantifiers) in the similar frequency and patterns
of the given texts."""

specific_rewrit_prompt = """The following are two texts. Can you rewrite the first text in the same linguistic styles of the
second text, i.e., use article, auxiliary verb, conjunction, indefinite pronouns, adverb, prepositional pronouns, prepositions, 
and quantifiers in the similar frequency and patterns of the second texts. Be sure to keep your answer at similar 
length of the first texts."""

general_mc_prompt = """In no more than 100 words, craft me an argument that the first fully automated McDonalds will open in the United States
before 2030. Be responsive to the linguistic style of the following text. """

general_rewrit_prompt = """The following are two texts. Can you rewrite the first text such that
it's responsive to the linguistic style of the second text. Be sure to keep your answer at similar length of the first texts."""

def prepare_corpus():
  reddit = Corpus(filename=download("reddit-corpus-small"))
  select_utts, reply_utts = get_any_speaker(reddit, 50, 100, 1)
  return select_utts, reply_utts

def get_llm_mcdonald_scores(prompt, reply_utts):
  llm_lst = llm_style_match(reply_utts, "gpt-4", prompt)
  llm_scores = get_coordination_scores(llm_lst, reply_utts)
  llm_scores = transform_values(llm_scores)
  average_scores = calculate_average_scores(llm_scores)
  return average_scores

def get_original_reply_scores(select_utts, reply_utts):
  reddit_reply_scores = get_coordination_scores(select_utts, reply_utts)
  transformed_reply_scores = transform_values(reddit_reply_scores)
  average_scores_reddit_reply = calculate_average_scores(transformed_reply_scores)
  return average_scores_reddit_reply

def get_llm_rewritten_reply_scores(prompt2, select_utts, reply_utts):
  LLM_reply_lst = llm_rewrite_replies(select_utts, reply_utts, 'gpt-4', prompt2)
  llm_scores = get_coordination_scores(LLM_reply_lst, reply_utts)
  average_scores = calculate_average_scores(llm_scores)
  return average_scores

def get_all_scores(specific_mc_prompt, specific_rewrit_prompt, general_mc_prompt, general_rewrit_prompt):
  select_utts, reply_utts = prepare_corpus()
  original_reply_scores = get_original_reply_scores(select_utts, reply_utts)
  # specific prompts
  sp_mc = get_llm_mcdonald_scores(specific_mc_prompt, reply_utts)
  sp_rw = get_llm_rewritten_reply_scores(specific_rewrit_prompt, select_utts, reply_utts)
  
  # general prompts
  ge_mc = get_llm_mcdonald_scores(general_mc_prompt, reply_utts)
  ge_rw = get_llm_rewritten_reply_scores(general_rewrit_prompt, select_utts, reply_utts)

  return original_reply_scores, sp_mc, sp_rw, ge_mc, ge_rw
