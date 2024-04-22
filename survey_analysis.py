from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_prediction_change(dataframe):
    # convert columns to numeric data types
    numeric_cols = ['Q2', 'Q4', 'Q6', 'Q8', 'Q10', 'Q13', 'Q16', 'Q19', 'Q3', 'Q5', 'Q7', 'Q11', 'Q14', 'Q17', 'Q20', 'Q9']
    dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # measure prediction change
    q1_lst = dataframe['Q10'] - dataframe['Q2']
    q2_lst = dataframe['Q13'] - dataframe['Q4']
    q3_lst = dataframe['Q16'] - dataframe['Q6']
    q4_lst = dataframe['Q19'] - dataframe['Q8']

    # measure certainty counted prediction change
    # transform certainty by +2 to avoid negative number
    # prediction wth high certainty matters more
    q1_pc = (dataframe['Q11'] + 2) * dataframe['Q10'] - (dataframe['Q3'] + 2) * dataframe['Q2']
    q2_pc = (dataframe['Q14'] + 2) * dataframe['Q13'] - (dataframe['Q5'] + 2) * dataframe['Q4']
    q3_pc = (dataframe['Q17'] + 2) * dataframe['Q16'] - (dataframe['Q7'] + 2) * dataframe['Q6']
    q4_pc = (dataframe['Q20'] + 2) * dataframe['Q19'] - (dataframe['Q9'] + 2) * dataframe['Q8']

    # create new DataFrame
    df = pd.DataFrame({
        'q1': q1_lst,
        'q2': q2_lst,
        'q3': q3_lst,
        'q4': q4_lst,
        'q1_pc': q1_pc,
        'q2_pc': q2_pc,
        'q3_pc': q3_pc,
        'q4_pc': q4_pc,
        'p1_before': dataframe['Q2'],
        'p2_before': dataframe['Q4'],
        'p3_before': dataframe['Q6'],
        'p4_before': dataframe['Q8'],
        'rule1': dataframe['rule1'],
        'rule2': dataframe['rule2'],
        'rule3': dataframe['rule3'],
        'rule4': dataframe['rule4'],
        'a1': dataframe['Q12'],
        'a2': dataframe['Q15'],
        'a3': dataframe['Q18'],
        'a4': dataframe['Q21'],
    })

    return df

 def count_value(dataframe, tuple_value):

   # create boolean masks for each condition
    rule1_mask = (dataframe['rule1'] == tuple_value)
    rule2_mask = (dataframe['rule2'] == tuple_value)
    rule3_mask = (dataframe['rule3'] == tuple_value)
    rule4_mask = (dataframe['rule4'] == tuple_value)

    agg_q1 = dataframe.loc[rule1_mask, 'q1'].tolist()
    agg_q2 = dataframe.loc[rule2_mask, 'q2'].tolist()
    agg_q3 = dataframe.loc[rule3_mask, 'q3'].tolist()
    agg_q4 = dataframe.loc[rule4_mask, 'q4'].tolist()

    # combine the four lists into a single long list
    total_aggregate = agg_q1 + agg_q2 + agg_q3 + agg_q4

    # return the total aggregate
    return total_aggregate

def plot_and_t_test_prediction_change(dataframe):
    # postive, general
    pg = count_value(dataframe, '(True, False)')
    
    # positive, personalized
    pp = count_value(dataframe, '(True, True)')
    
    # negative, general
    ng = count_value(dataframe, '(False, False)')
    
    # negative, personalized
    np = count_value(dataframe, '(False, True)')

    t_stat, p_value = stats.ttest_ind(ng, np)
    print("negative persuasion t-statistic:", t_stat)
    print("negative persuasion p-value:", p_value)
  
    t_stat, p_value = stats.ttest_ind(pg, pp)
    print("positive persuasion t-statistic:", t_stat)
    print("positive persuasion p-value:", p_value)  
    
    graph = {
        'Positive General': pg,
        'Positive Personalized': pp,
        'Negative General': ng,
        'Negative Personalized': np,
    }

    sns.set(style="whitegrid")  
    plt.figure(figsize=(10, 6))  
    ax = sns.violinplot(data=graph, palette="Set2", inner="box")  
    ax.set_title('Change in Predictions across Different Persuasions')  
    ax.set_ylabel('Prediction Change')  
    
    plt.savefig('simple_change.png', dpi=900)

 def count_value_weighted(dataframe, tuple_value):
    """
    Similar to the function `count_value` but weighting the change in prediction
    with the level of certainty
    """
    rule1_mask = (dataframe['rule1'] == tuple_value)
    rule2_mask = (dataframe['rule2'] == tuple_value)
    rule3_mask = (dataframe['rule3'] == tuple_value)
    rule4_mask = (dataframe['rule4'] == tuple_value)

    agg_q1 = dataframe.loc[rule1_mask, 'q1_pc'].tolist()
    agg_q2 = dataframe.loc[rule2_mask, 'q2_pc'].tolist()
    agg_q3 = dataframe.loc[rule3_mask, 'q3_pc'].tolist()
    agg_q4 = dataframe.loc[rule4_mask, 'q4_pc'].tolist()

    total_aggregate = agg_q1 + agg_q2 + agg_q3 + agg_q4

    return total_aggregate

def plot_and_t_test_weighted_prediction_change(dataframe):
  # postive, general
  pg_cp = count_value_weighted(simple_data, '(True, False)')
  
  # positive, personalized
  pp_cp = count_value_weighted(simple_data, '(True, True)')
  
  # negative, general
  ng_cp = count_value_weighted(simple_data, '(False, False)')
  
  # negative, personalized
  np_cp = count_value_weighted(simple_data, '(False, True)')

  
  t_stat, p_value = stats.ttest_ind(pg_cp, pp_cp)
  print("positive weighted t-statistic:", t_stat)
  print("positive weighted p-value:", p_value)

  t_stat, p_value = stats.ttest_ind(ng_cp, np_cp)
  print("negative weighted t-statistic:", t_stat)
  print("negative weighted p-value:", p_value)

  graph = {
      'Positive General': pg_cp,
      'Positive Personalized': pp_cp,
      'Negative General': ng_cp,
      'Negative Personalized': np_cp,
  }
  
  sns.set(style="whitegrid")  
  plt.figure(figsize=(10, 6))  
  ax = sns.violinplot(data=graph, palette="Set2", inner="box", cut=0)  
  ax.set_title('Certainty-weighted Change in Predictions across Different Persuasions')  
  ax.set_ylabel('Certainty-weighted prediction Change')  
  
  plt.savefig('weighted_change.png', dpi=900)

 def count_appealingness(dataframe, tuple_value):
    rule1_mask = (dataframe['rule1'] == tuple_value)
    rule2_mask = (dataframe['rule2'] == tuple_value)
    rule3_mask = (dataframe['rule3'] == tuple_value)
    rule4_mask = (dataframe['rule4'] == tuple_value)

    agg_q1 = list(dataframe.loc[rule1_mask, 'a1'])
    agg_q2 = list(dataframe.loc[rule2_mask, 'a2'])
    agg_q3 = list(dataframe.loc[rule3_mask, 'a3'])
    agg_q4 = list(dataframe.loc[rule4_mask, 'a4'])

    total_aggregate = agg_q1 + agg_q2 + agg_q3 + agg_q4

    return [int(x) for x in total_aggregate]

  def plot_applingnness(dataframe):
    positive_general = count_appealingness(dataframe, '(True, False)')
    positive_personalized = count_appealingness(dataframe, '(True, True)')
    negative_personalized = count_appealingness(dataframe, '(False, True)')
    negative_general  = count_appealingness(dataframe, '(False, False)')
    graph = {
        'Positive General': positive_general,
        'Positive Personalized': positive_personalized,
        'Negative General': negative_general,
        'Negative Personalized': negative_personalized,
    }
    
    sns.set(style="whitegrid") 
    plt.figure(figsize=(10, 6))  
    ax = sns.violinplot(data=graph, palette="Set2", inner='box')  
    ax.set_title('Appealingness across Different Persuasions')  
    ax.set_ylabel('Appealingness') 

    plt.savefig('appeal.png', dpi=900)

def check_sign(value, condition):
    """
    Helper function to check the sign of a value based on a condition.
    """
    if condition == '(True, False)' or condition == '(True, True)':
        return value > 0
    else:
        return value < 0

def conditional_on_prior(dataframe):
    """
    Check the affirmative status based on the conditions specified for each rule.
    """
    affirming_dict = {1: [], 2: [], 3: [], 4: []}

    for _, row in dataframe.iterrows():

        # Check each rule and corresponding prior value
        for rule_num in range(1, 5):
            affirming = False
            condition = row[f'rule{rule_num}']
            prior_value = row[f'p{rule_num}_before']
            if check_sign(prior_value, condition):
                affirming = True
            affirming_dict[rule_num].append(affirming)

    dataframe['affirm1'] = affirming_dict[1]
    dataframe['affirm2'] = affirming_dict[2]
    dataframe['affirm3'] = affirming_dict[3]
    dataframe['affirm4'] = affirming_dict[4]

    return dataframe

 def analyze_prior(dataframe, whether_affirm):
    rule1_mask = (dataframe['affirm1'] == whether_affirm)
    rule2_mask = (dataframe['affirm2'] == whether_affirm)
    rule3_mask = (dataframe['affirm3'] == whether_affirm)
    rule4_mask = (dataframe['affirm4'] == whether_affirm)

    agg_q1 = [abs(value) for value in dataframe.loc[rule1_mask, 'q1']]
    agg_q2 = [abs(value) for value in dataframe.loc[rule2_mask, 'q2']]
    agg_q3 = [abs(value) for value in dataframe.loc[rule3_mask, 'q3']]
    agg_q4 = [abs(value) for value in dataframe.loc[rule4_mask, 'q4']]

    # Combine the four lists into a single long list
    total_aggregate = agg_q1 + agg_q2 + agg_q3 + agg_q4

    return total_aggregate

 def plot_prior(dataframe):
    dataframe = conditional_on_prior(dataframe)
    affirm_agg = analyze_prior(dataframe, True)
    not_affirm_agg = analyze_prior(dataframe, False)
    graph = {
    'Confirming': affirm_agg,
    'Opposing': not_affirm_agg,
    }
    sns.set(style="whitegrid")  
    plt.figure(figsize=(10, 6))  
    ax = sns.violinplot(data=graph, palette="Set2", inner='box')  
    ax.set_title('Absolute Prediction Change When Arguments Confirming vs. Opposing Prior Belief')  
    ax.set_ylabel('Prediction Change')  
    plt.savefig('confirm.png', dpi=900)


if __name__ == "__main__":
  dataframe = pd.read_csv('survey_data.csv')
  dataframe = calculate_prediction_change(dataframe)
  plot_and_t_test_prediction_change(dataframe)
  plot_and_t_test_weighted_prediction_change(dataframe)
  plot_applingnness(dataframe)
  plot_prior(dataframe)
