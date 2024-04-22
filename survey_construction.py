import pandas as pd
import numpy as np
from openai import OpenAI
client = OpenAI(
  api_key="ENTER_YOUR_API_KEY") # enter your key here

pre_prompt = """In no more than 100 words, craft me an argument that"""

post_prompt = """Use function words (i.e., article, auxiliary verb, conjunction, indefinite pronouns, adverb,
prepositional pronouns, prepositions, and quantifiers) in the similar frequency and patterns
of the following texts."""

# a list of arguments we will ask GPT-4 to generate
true_arguments = ['The first fully automated McDonalds will open in the United States before 2030',
                  'Iran will possess a nuclear weapon before 2030',
                  'Solar power on Earth will dominate renewable energy consumption before 2030',
                  'Someone born before 2001 will live to be 150'
                  ]
false_arguments = ['The first fully automated McDonalds will not open in the United States before 2030',
                  'Iran will not possess a nuclear weapon before 2030',
                  'Solar power on Earth will not dominate renewable energy consumption before 2030',
                  'Someone born before 2001 will not live to be 150'
                  ]

def gpt4_personalize(pre_prompt, argument, post_prompt, user_text, model_name='gpt-4'):
    # API call
    total_prompt = pre_prompt + argument + post_prompt + '\n' + user_text

    response = client.chat.completions.create(
    model= model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled at generating persuasive messages,"},
        {"role": "user", "content": "{}".format(total_prompt)}
    ] # can also set max length
  )
    response_text = response.choices[0].message.content
    print(response_text)
    return response_text

def gpt4_general(pre_prompt, argument, model_name='gpt-4'):
    # API call
    total_prompt = pre_prompt + argument

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled at generating persuasive messages,"},
            {"role": "user", "content": "{}".format(total_prompt)}
        ]  # can also set max length
    )

    response_text = response.choices[0].message.content
    return response_text

def generate_general_arguments(pre_prompt, arguments):
    generated_lst = []
    for argument in arguments:
        generated = gpt4_general(pre_prompt, argument, model_name='gpt-4')
        generated_lst.append(generated)
    return generated_lst

def create_rules(data):
  """
  Given "data," a pandas dataframe where each rows repreesnts a participants, create 4 rules columns
  that determines what types of arguments each participant receive for each of the 4 prediction question. 
  """
  # define the possible values for each column; for each tuple
  # the first True/False value determine whether the direction of persuasion is positive (i.e. the future event will happen)
  # the second True/False value determines whether the argument is personalized 
  possible_values = [(True, True), (False, True), (True, False), (False, False)]

  # iterate over each row and randomly shuffle the possible values
  for index, row in data.iterrows():
      np.random.shuffle(possible_values)
      # assign shuffled values to each column
      for i, (val1, val2) in enumerate(possible_values):
          data.at[index, f'rule{i+1}'] = (val1, val2)
  return data

def assign_arguments(row, rule, rule_num, general_true, general_false, true_arguments, false_arguments):
  """
  A function to handle argument assignment based on rules
  """  
  # determine whether the argument is specific
    if row[rule][1] == False: # if the argument is general
          if row[rule][0] == True:
            return general_true[int(rule_num) - 1]
          else:
            return general_false[int(rule_num) - 1]

    else: # needs GPT4 to personalize
        if row[rule][0] == True:
          arg = true_arguments[int(rule[-1]) - 1]
        else:
          arg = false_arguments[int(rule[-1]) - 1]
        print('argument to personalize: ' + arg)
        response = gpt4_personalize(pre_prompt, arg, post_prompt, row['Texts'], model_name='gpt-4')
        return response

if __name__ == "__main__":
  data = pd.read_csv('text_data.csv') 
  # each row represents a participant, a 'Text' column must stores the participant's text data as lists of strings
  data = create_rules(data)
  general_false = generate_general_arguments(pre_prompt, false_arguments)
  general_true = generate_general_arguments(pre_prompt, true_arguments)
  for index, row in data.iterrows():
    for rule_num in range(1, 5):
        rule_name = f'rule{rule_num}'
        arg_column_name = f'arg{rule_num}'
        data.loc[index, arg_column_name] = assign_arguments(row, rule_name,
                                                             rule_num,
                                                             general_true=general_true,
                                                             general_false=general_false,
                                                             true_arguments=true_arguments, false_arguments=false_arguments)
  data.to_csv('survey_ready.csv')
