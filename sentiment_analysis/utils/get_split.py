import pandas as pd
import warnings, os
warnings.simplefilter("ignore")

'''
this script identifies stock splits based on price changes in merged data
'''

for file in os.listdir('/home/dave/Desktop/code_hackathon/merged_data/'):
    try:
        #df = pd.read_csv('/home/dave/Desktop/code_hackathon/merged_data/merged_64768.0.csv')
        df = pd.read_csv(os.path.join('merged_data', file))
        df['split'] = [1]*len(df)

        initial_price = float(df.iloc[0, 2]) 
        print (initial_price)
        from datetime import datetime
        SPLIT = 1
        TOTAL_SPLIT = 1
        for i in range(len(df) -1):
            row = df.iloc[i].tolist()
            next_row = df.iloc[i+1].tolist()
            date, next_date = row[0], next_row[0]
            date, next_date = datetime.strptime(date, '%Y-%m-%d'), datetime.strptime(next_date, '%Y-%m-%d')
            #print(round((next_date-date).days/30, 0),)
            if round((next_date-date).days/30, 0) == 1:

                initial_price = initial_price * (1 + (next_row[3]))
            else:
                TOTAL_SPLIT = 1 # reset split multiplier
                #print("GAPS", round((next_date-date).days/30, 0))
                initial_price = next_row[2]
            #print(row[2], initial_price, date, row[3])
            if (abs(next_row[2] - initial_price) > 0.5):
                if round(initial_price/next_row[2], 2) / TOTAL_SPLIT == 1:
                    continue
                SPLIT = round(initial_price/next_row[2], 2) / TOTAL_SPLIT
                TOTAL_SPLIT = TOTAL_SPLIT * SPLIT
                #print(df['split'][[i-1]], df['split'][[i]])
                #print(TOTAL_SPLIT, SPLIT)
                if i>0 and int(df['split'][[i-1]]) != SPLIT and SPLIT != 1:
                    df['split'][[i]] = SPLIT
                    print("Mismatch", next_row[2], initial_price, next_date, next_row[3], round(initial_price/next_row[2], 2))
                SPLIT = 1
        df.to_csv('merged_data_with_split/' + file, index=False)
    except Exception as e:
        print(e)
        continue