import pickle 
import pandas as pd 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import numpy as np 
from torch.nn.functional import softmax 
import re

class BERTSentimentAnalyzer: 
    def __init__(self, model_name="yiyanghkust/finbert-tone"): 
        """ 
                Initialize BERT sentiment analyzer 
                         
                                 Args: 
                                             model_name (str): Pre-trained model name from Hugging Face 
                                                     """ 
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=3) 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(self.device) 
        self.model.to(self.device) 
        self.model.eval() 
     
    def preprocess_text(self, text): 
        """ 
        Preprocess text for BERT input 
         
        Args: 
            text (str): Input text 
             
        Returns: 
            dict: Tokenized inputs 
        """ 
        return self.tokenizer( 
            text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt" 
        ) 
    
    def split_text_by_sentences(self, text, max_length=400):
        """
        Split text into chunks by sentences, respecting max token length
        
        Args:
            text (str): Input text
            max_length (int): Maximum tokens per chunk (leaving room for special tokens)
            
        Returns:
            list: List of text chunks
        """
        # Split by sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed token limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            tokens = self.tokenizer.tokenize(test_chunk)
            
            if len(tokens) <= max_length:
                current_chunk = test_chunk
            else:
                # If current chunk has content, save it and start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
                # If single sentence is too long, split by words
                if len(self.tokenizer.tokenize(sentence)) > max_length:
                    word_chunks = self.split_long_sentence(sentence, max_length)
                    chunks.extend(word_chunks[:-1])  # Add all but last
                    current_chunk = word_chunks[-1]  # Keep last as current
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def split_long_sentence(self, sentence, max_length):
        """
        Split a very long sentence by words
        
        Args:
            sentence (str): Long sentence
            max_length (int): Maximum tokens per chunk
            
        Returns:
            list: List of sentence chunks
        """
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            tokens = self.tokenizer.tokenize(test_chunk)
            
            if len(tokens) <= max_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def predict_sentiment(self, text): 
        """ 
        Predict sentiment for a single text 
         
        Args: 
            text (str): Input text 
             
        Returns: 
            dict: Prediction results with sentiment and confidence 
        """ 
        # Tokenize input 
        inputs = self.preprocess_text(text) 
        inputs = {key: value.to(self.device) for key, value in inputs.items()} 
         
        # Make prediction 
        with torch.no_grad(): 
            outputs = self.model(**inputs) 
            predictions = softmax(outputs.logits, dim=-1) 
         
        # Get predicted class and confidence 
        predicted_class = torch.argmax(predictions, dim=-1).item() 
        confidence = torch.max(predictions).item() 
         
        # Map to sentiment labels (adjust based on your model) 
        sentiment_labels = {0: "neutral", 1: "positive", 2: "negative"} 
         
        sentiment = sentiment_labels.get(predicted_class, "unknown") 
         
        return { 
            "text": text, 
            "sentiment": sentiment, 
            "confidence": confidence, 
            "class_id": predicted_class, 
            "all_scores": predictions.cpu().numpy().flatten(),
            "num_chunks": 1
        }
    
    def predict_long_text_sentiment(self, text, aggregation_method="weighted_average"):
        """
        Predict sentiment for long text by chunking and aggregating results
        
        Args:
            text (str): Input text (can be very long)
            aggregation_method (str): How to combine chunk results
                - "weighted_average": Weight by confidence scores
                - "majority_vote": Most common sentiment
                - "average": Simple average of scores
                - "most_confident": Use result from most confident chunk
                
        Returns:
            dict: Aggregated prediction results
        """
        # Check if text is short enough for direct processing
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= 400:  # Direct processing for short texts
            return self.predict_sentiment(text)
        
        # Split long text into chunks
        chunks = self.split_text_by_sentences(text)
        
        if not chunks:
            return {"error": "No valid chunks created"}
        
        # Analyze each chunk
        chunk_results = []
        for chunk in chunks:
            result = self.predict_sentiment(chunk)
            chunk_results.append(result)
        
        # Aggregate results
        return self.aggregate_chunk_results(chunk_results, text, aggregation_method)
    
    def aggregate_chunk_results(self, chunk_results, original_text, method="weighted_average"):
        """
        Aggregate sentiment results from multiple chunks
        
        Args:
            chunk_results (list): List of individual chunk results
            original_text (str): Original full text
            method (str): Aggregation method
            
        Returns:
            dict: Aggregated results
        """
        if not chunk_results:
            return {"error": "No chunk results to aggregate"}
        
        sentiment_labels = {0: "neutral", 1: "positive", 2: "negative"}
        
        if method == "weighted_average":
            # Weight by confidence scores
            total_weight = sum(r['confidence'] for r in chunk_results)
            if total_weight == 0:
                total_weight = len(chunk_results)
            
            # Calculate weighted average for each class
            num_classes = len(chunk_results[0]['all_scores'])
            weighted_scores = np.zeros(num_classes)
            
            for result in chunk_results:
                weight = result['confidence'] / total_weight
                weighted_scores += result['all_scores'] * weight
            
            predicted_class = np.argmax(weighted_scores)
            confidence = np.max(weighted_scores)
            
        elif method == "majority_vote":
            # Count sentiment votes
            sentiment_counts = {}
            for result in chunk_results:
                sentiment = result['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Find majority sentiment
            majority_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
            predicted_class = next(k for k, v in sentiment_labels.items() if v == majority_sentiment)
            
            # Average confidence for majority sentiment
            majority_results = [r for r in chunk_results if r['sentiment'] == majority_sentiment]
            confidence = np.mean([r['confidence'] for r in majority_results])
            weighted_scores = np.mean([r['all_scores'] for r in chunk_results], axis=0)
            
        elif method == "average":
            # Simple average of all scores
            weighted_scores = np.mean([r['all_scores'] for r in chunk_results], axis=0)
            predicted_class = np.argmax(weighted_scores)
            confidence = np.max(weighted_scores)
            
        elif method == "most_confident":
            # Use result from most confident chunk
            most_confident_result = max(chunk_results, key=lambda x: x['confidence'])
            predicted_class = most_confident_result['class_id']
            confidence = most_confident_result['confidence']
            weighted_scores = most_confident_result['all_scores']
        
        sentiment = sentiment_labels.get(predicted_class, "unknown")
        
        return {
            "text": original_text,
            "sentiment": sentiment,
            "confidence": float(confidence),
            "class_id": int(predicted_class),
            "all_scores": weighted_scores.tolist() if hasattr(weighted_scores, 'tolist') else weighted_scores,
            "num_chunks": len(chunk_results),
            "chunk_results": chunk_results,  # Include individual chunk results
            "aggregation_method": method
        }
     
    def predict_batch(self, texts, handle_long_texts=True): 
        """ 
        Predict sentiment for multiple texts 
         
        Args: 
            texts (list): List of input texts 
            handle_long_texts (bool): Whether to use chunking for long texts
             
        Returns: 
            list: List of prediction results 
        """ 
        results = [] 
        for text in texts: 
            if handle_long_texts:
                result = self.predict_long_text_sentiment(text)
            else:
                result = self.predict_sentiment(text)
            results.append(result) 
        return results 
     
    def analyze_dataframe(self, df, text_column, handle_long_texts=True): 
        """ 
        Analyze sentiment for texts in a pandas DataFrame 
         
        Args: 
            df (pd.DataFrame): Input dataframe 
            text_column (str): Name of the text column 
            handle_long_texts (bool): Whether to use chunking for long texts
             
        Returns: 
            pd.DataFrame: DataFrame with sentiment analysis results 
        """ 
        results = self.predict_batch(df[text_column].tolist(), handle_long_texts) 
         
        # Add results to dataframe 
        df_results = df.copy() 
        df_results['sentiment'] = [r['sentiment'] for r in results] 
        df_results['confidence'] = [r['confidence'] for r in results] 
        df_results['class_id'] = [r['class_id'] for r in results] 
        # Add additional info for long texts
        if handle_long_texts:
            df_results['num_chunks'] = [r.get('num_chunks', 1) for r in results]
            df_results['aggregation_method'] = [r.get('aggregation_method', 'direct') for r in results]
         
        return df_results

stock_ticker_df = pd.read_csv('cik_gvkey_linktable_USA_only.csv')
def main():
    # Initialize the analyzer
    print("Loading BERT model...")
    analyzer = BERTSentimentAnalyzer()
    
    content = ""
    dataframe_sentiment = pd.DataFrame(columns=["sentiment","neutral_score", "positive_score", "negative_score", "ticker", "gvkey", "date", "chunks"])
    for year in range(2006, 2026):
        with open(f"./text_files/text_us_{year}.pkl", "rb") as f:
            content = pickle.load(f)
    
        for i in range(len(content)):
            try:
                text = content['mgmt'].iloc[i]
                result = analyzer.predict_long_text_sentiment(text, aggregation_method="weighted_average")

                gvkey = content['gvkey'].iloc[i]
                stock_index = stock_ticker_df[stock_ticker_df["gvkey"] == gvkey].index.tolist()[0]

                date = content['date'].iloc[i]

                neutral_score = result["all_scores"][0]
                positive_score = result["all_scores"][1]
                negative_score = result["all_scores"][2]

                dataframe_sentiment.loc[i] = [result['sentiment'], round(neutral_score, 5), round(positive_score, 5), round(negative_score, 5), stock_ticker_df['tic'].iloc[stock_index], gvkey, date, result['num_chunks']]
                #print(result)
                print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f}) {stock_ticker_df['tic'].iloc[stock_index], stock_ticker_df['conm'].iloc[stock_index]} {date} Number of chunks: {result['num_chunks']}")
            except Exception as e:
                print(f"Error processing index {i}: {e}")
                continue

        dataframe_sentiment.to_csv(f"./sentiment_result/sentiment_{year}_finebert.csv", index=False)
if __name__ == "__main__":
    main()
    