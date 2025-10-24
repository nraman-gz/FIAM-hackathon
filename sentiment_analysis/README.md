
# Sentiment Analysis Using 10-K/10-Q Reports

### sentiment_analysis.py
This script performs sentiment analysis on long financial texts using a fine-tuned BERT model, it handles long texts by splitting them into manageable chunks and aggregating the results.

### filter_sentiment.py
This script merges sentiment data with stock price data based on nearest previous date, it also filters stocks that are not in USA or have intrinsic_value <= 500mm.

### get_split.py
This script identifies stock splits based on price changes in merged data

### process_sentiment.py
This script processes sentiment data to calculate net and delta scores

### eval.py
This script evaluates a simple trading strategy based on sentiment delta scores, it simulates investing $10,000 in stocks with delta_score > 0.1 each month and calculates the overall return.

# Instructions

Using conda with gpu is recommended

You must place ```ret_sample.csv```,  ```cik_gvkey_linktable_USA_only.csv``` in the root dir. 

Place all text files in ```text_files```




```
pip install -r requirements.txt
```


```
python sentiment_analysis.py
python filter_sentiment.py
python get_split.py
python process_sentiment.py
python eval.py
```

A few more linux command were used to combine and manipulate csv files. 

I have also attached some algorithms that I have attempted in the ```correlation_with_delta.ipynb```. They are NOT well-organized and are NOT used as part of our results. 

# On FinBERT

FinBERT model [link](https://huggingface.co/yiyanghkust/finbert-tone)

## Intro
BERT (Bidirectional Encoder Representations from Transformers) model is not a LLM like ChatGPT: it doesn't generate text; it understands it. There are many type of BERT models trained for different scenarios, like Reddit posts and Tweets. On the other hand, FinBERT is sepecifically trained on 10-K/10-Q reports. Therefore, it can interpret financial text much better than many of its counterparts. P.S. I tested general purpose BERT and it struggled with the vague languages that are commonly used in the financial world.

## WHY USE FinBERT NOT CHATGPT???
Instead of returning a long responces like ChatGPT, FinBERT returns three float numbers: positive, neutral, and negative score. This makes FinBERT much faster and efficient than other LLMs. For example, on average, it takes less than one second for FinBERT to go through a full report using one RTX 4060 ti. On the other hand, in order to run a local LLM with 32B, we may need much more computing power (at lease 4x).

## Limitations 
### Text length
Since FinBERT can only read 512 tokens at a time, it is impossible to input the full text at once. We need to split them into smaller chunks. For the first stage, I divided the reports into chunks of up to 400 tokens. However, some reseachers splited the reports by sentences instead. [Link to paper](https://www.sciencedirect.com/science/article/pii/S219985312200350X). I will try the new method once I have time. 

### Inability to read graphs and figures
Graphs and Figures are essential parts of corporate reports. However, BERT can only read text. I believe that we are also missing many numerical data such as earnings and projections. We may need extra finicial data. P.S. we can try using earnings trend to predict the sentiment. 

### Other concerns
FinBERT is published in 2022, it is not up-to-date. 


Many other teams are using FinBERT, maybe we should add extra layers. 

## EXTRA
we can try FinGPT