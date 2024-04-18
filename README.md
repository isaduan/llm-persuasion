# Large-Scale Linguistic Mimicry: Evaluating Persuasion Efficiency of Large Language Models

The code in this repository is the supplemental code for Isabella Duan's MA thesis at the University of Chicago.

The code is written in Python 3.9.7 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

You can run `metrics.py` to reproduce our analysis of how well our 9 linguistic mimicry metrics perform on 4000 samples constrcuted from a small reddit corpus made available by [Covokit](https://convokit.cornell.edu/documentation/reddit-small.html#usage). This script will generate two json files,  `results_similar.json` and  `results_dissimilar.json`, which reproduce data in Table 1. 

```
python metrics.py
```
