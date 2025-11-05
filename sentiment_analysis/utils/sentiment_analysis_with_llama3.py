import pickle 
import pandas as pd 
from llama_cpp import Llama
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
#import nltk    
#nltk.download('punkt_tab')
from to_paragraph import split_into_paragraphs
import warnings
warnings.filterwarnings("ignore")

# Configuration
#MODEL_PATH = "llama-3-lora/FinGPT-MT-Llama-3-8B-LoRA-Q8_0.gguf" 
MODEL_PATH = "llama-3-lora/Llama-3-8B-Instruct-Finance-RAG.Q4_K_M.gguf" 
#https://huggingface.co/QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF
N_CTX = 8192  # Context window size
N_GPU_LAYERS = -1  # Set to -1 to use GPU, 0 for CPU only
TEMPERATURE = 0.9
MAX_TOKENS = 5
MIN_TOKENS = 1
TOP_P = 0.95
TOP_K = 40

def load_model(model_path, n_ctx=2048, n_gpu_layers=0):
    """Load the GGUF model."""
    print(f"Loading model from: {model_path}")
    print(f"Context size: {n_ctx}")
    print(f"GPU layers: {n_gpu_layers}")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    
    print("Model loaded successfully!\n")
    return llm

def evaluate_prompt(llm, prompt, temperature=0.7, max_tokens=512, top_p=0.95, top_k=40):
    """Evaluate a single prompt and return the result."""
    #print(f"Prompt: {prompt}")
    #print("-" * 80)
    
    start_time = time.time()
    
    response = llm(
        prompt,
        max_tokens=max_tokens,
        #min_tokens=MIN_TOKENS,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        echo=False,
        stop=["</s>"],
        
    )
    
    elapsed_time = time.time() - start_time
    
    output_text = response['choices'][0]['text']
    tokens_generated = response['usage']['completion_tokens']
    tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
    
    #print(f"Response: '{repr(output_text)}'")
    #print(f"\nTokens: {tokens_generated} | Time: {elapsed_time:.2f}s | Speed: {tokens_per_sec:.2f} tok/s")
    #print("=" * 80)
    #print()
    
    return output_text.upper()

def is_integer_string(s):
    try:
        int(s)  # Attempt to convert the string to an integer
        return True
    except ValueError:
        return False

def run_evaluation(llm, prompts, output_file=None):
    """Run evaluation on all prompts."""

    # Evaluate all prompts
    results = []
    result_sets = [0,0,0]
    
    for i, prompt in enumerate(prompts, 1):
        #print(f"\n{'='*80}")
        #print(f"Evaluating prompt {i}/{len(prompts)}")
        #print(f"{'='*80}\n")
        
        result = evaluate_prompt(
            llm, 
            prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K
        )
        #print(f"Result: {result}")
        result = result.strip()
        if result == "POSITIVE":
            result_sets[0] += 1
        elif result == "NEGATIVE":
            result_sets[1] += 1
        elif result == "NEUTRAL":
            result_sets[2] += 1
        elif is_integer_string(result):
            # in case the model outputs a number corresponding to the choices
            num = int(result)
            if num > 0:
                result_sets[0] += 1
            elif num < 0:
                result_sets[1] += 1
            elif num == 0:
                result_sets[2] += 1
        else:
            print("error")
    #total = sum(result_sets)
    #print()
    return result_sets

def combine_pairs_loop(lst):
    paragraphs = []
    for i in lst:
        # filter out very short sentences
        if len(i.split()) >= 15:
            paragraphs.append(i)
    """Combine every two consecutive elements using a simple loop"""
    result = []
    for i in range(0, len(paragraphs), 2):
        if i + 1 < len(paragraphs):
            result.append(paragraphs[i] + paragraphs[i + 1])
        else:
            # Handle odd number of elements
            result.append(paragraphs[i])
        #group = paragraphs[i:i+4]
        #result.append(''.join(group))
    return result
    
stock_ticker_df = pd.read_csv('cik_gvkey_linktable_USA_only.csv')
def main():    
    content = ""
    dataframe_sentiment = pd.DataFrame(columns=["neutral_score", "positive_score", "negative_score", "ticker", "gvkey", "date", "chunks"])
    llm = load_model(MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS)

    for year in range(2015, 2026):
        with open(f"./text_files/text_us_{year}.pkl", "rb") as f:
            content = pickle.load(f)
    
        for i in tqdm(range(len(content))):
            try:
                prompts=[]
                text = content['mgmt'].iloc[i]
                #sentences = nltk.sent_tokenize(text)
                paragraphs = split_into_paragraphs(text, keep_tables_together=True)
                #print(paragraphs)
  
  
                #paragraphs = combine_pairs_loop(paragraphs)

                for chunks in paragraphs:
                    if len(chunks.split()) > 15:
                        prompts.append(f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
                        You are a sentiment classifier that must respond with exactly one of Positive, Negative, or Neutral. 
                        <|eot_id|><|start_header_id|>user<|end_header_id|>\n

                        Text: "''' + chunks.strip() +  '"\nAnswer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>' )
                #sentences = nltk.sent_tokenize(text)

                result_sets = run_evaluation(llm, prompts)
                total = result_sets[0] + result_sets[1] + result_sets[2]
                if total == 0:
                    continue
                positive_score, negative_score, neutral_score = result_sets[0]/total, result_sets[1]/total, result_sets[2]/total

                #print(positive_score, negative_score, neutral_score)
                gvkey = content['gvkey'].iloc[i]
                stock_index = stock_ticker_df[stock_ticker_df["gvkey"] == gvkey].index.tolist()[0]

                date = content['date'].iloc[i]

                dataframe_sentiment.loc[i] = [round(neutral_score, 5), round(positive_score, 5), round(negative_score, 5), stock_ticker_df['tic'].iloc[stock_index], gvkey, date, len(prompts)]
                #print(result)
                #print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f}) {stock_ticker_df['tic'].iloc[stock_index], stock_ticker_df['conm'].iloc[stock_index]} {date} Number of chunks: {result['num_chunks']}")
                
            except Exception as e:
                print(f"Error processing index {i}: {e}")
                continue
            
        dataframe_sentiment.to_csv(f"./sentiment_result/sentiment_{year}_llama.csv", index=False)
if __name__ == "__main__":
    main()
    