import pandas as pd

'''
this script processes sentiment data to calculate net and delta scores
'''

# Read the CSV file
df = pd.read_csv('merged.csv')

df['negative_score'] = pd.to_numeric(df['negative_score'], errors='coerce') 
df['positive_score'] = pd.to_numeric(df['positive_score'], errors='coerce') 

# Sort by ticker and then by date
df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
df = df.round({"positive_score": 4, "negative_score": 4, "neutral_score": 4})


# Add net_score column (positive_score - negative_score)
df['net_score'] = (df['positive_score'] - df['negative_score']).round(4)



# Calculate delta_score (change in net_score within each ticker)
df['delta_score'] = df.groupby('ticker')['net_score'].diff().round(4)



# Display the first few rows
print(df.head(20))

# Save to a new CSV file
df.to_csv('processed_sentiment.csv', index=False)
print("\nProcessed file saved as 'processed_sentiment.csv'")

with open("processed_sentiment.csv", "r") as w:
    lines = w.readlines()
with open("processed_sentiment.csv", "w") as w:
    for line in lines:
        if ",,,,,,,,," not in line:
            w.write(line)
# Remove rows with NaN values