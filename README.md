# Large-Scale Linguistic Mimicry: Evaluating Persuasion Efficiency of Large Language Models

The code in this repository is the supplemental code for Isabella Duan's MA thesis at the University of Chicago.

The code is written in Python 3.9.7 and all of its dependencies can be installed by running the following in the terminal:

```
pip install -r requirements.txt
```

You can run `metrics.py` to reproduce our analysis of how well our 9 linguistic mimicry metrics perform on 4000 samples constrcuted from a small reddit corpus made available by [Covokit](https://convokit.cornell.edu/documentation/reddit-small.html#usage). This script will generate two json files,  `results_similar.json` and  `results_dissimilar.json`, which reproduce data in Table 1. 

```
python metrics.py
```
Next, you can use `benchmark.py` to reproduce our analysis on benchmarking GPT-4's liguistic mimicry capability with the mimicry naturally occuring between humans. You will need to provide your own OpenAI key on line 11 of the script. This script will render and save two figures, `three.png`, which contrasts GPT-4's mimicry capability when construct new arguments and when rewriting original reply with that of original human reply, as well as `prompt.png`, which shows GPT-4's linguistic mimicry capability under general vs. specific prompts.

```
python benchmark.py
```

![Linguistic mimicry in GPT-4 Generated Texts with Specific vs. General Prompts](https://github.com/isaduan/llm-persuasion/blob/main/prompt.png)

![Linguistic mimicry in GPT-4 Generated Texts vs. Reddit User Reply, with Specific Prompts](https://github.com/isaduan/llm-persuasion/blob/main/three.png)

You can use `survey_construction.py` to reproduce how we constructed personalized surveys for each participant, testing linguistically personalized arguments' persuasive effectiveness against general arguments. For privacy and data security concerns, we do not provide the raw data or the code to access participants' reddit text history. However, you can implement using your own CSV-formatted data, `text_data.csv`, with each row representing a participant and a 'Text' column storing text data as lists of strings. The script will produce a new csv file `survey_ready.csv` with new columns e.g. "arg1" that stores arguments ready to run the survey experiment.
