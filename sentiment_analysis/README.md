
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