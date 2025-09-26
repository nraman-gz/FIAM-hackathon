### Meeith with Qingnan and Tony 


training AI agents is not necessary 
time series forecasting with training a transformer 

if we use deep learning there is no need to do feature selection because it will do it automatically (this is a black box we will not be able to explain it) 
we can send him the code and he will train the model? what a W guy 

pythorch 
  nikhil used neuro networks and optimizers (which is enough) 

Pandas: 
  to clean the data 
  the model only takes numbers 
  USA = 1, Canada = 2 (but he doesnt recommend this), to feed countries into the model he said drop that row idk or we will have buggs 
  they have 4 samples (for examplejan - may) and they try to estimate price in may, so they make each sample that has 4 timestamp inputs (then he drew a graph) 

x and y - here these months are the four lags (this is a sliding window) MAKE SURE TRAINING NOT IN VALIDATION 
1 - jan
estimate for may 
2 - feb
new estimate for may 
3 - march
new estimate for may 
4 - april 
new estimate for may 

or without using time series models, do it just for january, you can also make the input in 1 for example as the return 

Make sure: in the training data set we do not only fit the one time stamp information - krishna understood this I did not (training data set is not the same as validation data set) 
   we haev 20 years of data, we can take first 18 years as training as last 2 as validation 
  he used the sequence: 2, 4, 6 
  in deep learning, when it automatically chooses features, you still have to manually choose/put in the lagged series  (but this is in pre processing step of data set) 
  how do you evaluate which lag is best? they used mean squared value error because this is a regression 
  neuro networks are non linear 
  bayesian information criterion? like log likelihood 
  L1loss in pythorch - absolute error 

*** end of prediction part

reinforcement learning didnt work for portfolio optimization 
they just used the provided code for portfolio optimization (they just feed the prediction part into that) 

note: transformers do not have a mechanics that counts sequence, and also transformers need alot of data to perform well, hes not surw of we have enough data for that, so he recommends we start with LSTM (nikhil knows - long short term memory) and you try different (ssians??? Nikhil knows) and how to build an MLP 
their model is CNN LSTM with AM 
CNN chooses features along feature space and then feed that into LSTM (lstm can learn info from time sequence) and this is from the table of x and y above, LSTM learns whats mpst important and it gives us weighting from thetension volume, add it up we get final embedding, and the last step is to  pass with linear layer 
part 2: financial intuition based on prediction on maximizing portfolio 

Q: did they train 1 model for evry stock? 
  what they did is they trained 1 model that just works for every stock train one model for each sector? this is doable 

*** Presnetation: 
  come up w fancy stuff you dont need to prove it worked
  they j lied 
  they did not take into account trading costs liel broker fees 
  the thought process behind it should be well explained (thats his recommendation) 
  and if u have a cool idea but doesnt work then just use part of it 
  they are looking at your methodology and how you are thinking 
  AI approach can stand out (many ppl in quant but not AI or really shine in the math or ideally use both) 

the guys name is Tony Xu 
  they just picked top 50 (they grouped it into 50’s) they went into the groups and took the best group (it might not have been the top group) 
  the portfolio optimization was just the template code, they didnt do any more optimization 
  they did not short the 50 lowest because they found that overall was not that good 

text idea: process for extrotidanry events (mentioned in footnotes - but this is often already priced into the markets)  that ppl have missed (but he doesnt know this is hard) find correlation with next quarter price (what he did is rank via google API and assign a sentiment score and that had a small correlation to the next quarter price) - or we already this  - something that management downlpalyed the riks and then had it the next quarter (but this would be an event driven strategy) 
we would need natural language undestnading midel to pull this out 

also to differentiate yourself, if you can think of another strategy or take external data that would be very good  - like some macro strategies maybe rates (how conflicts affect enrgy pirces and look into un the deep level) 



### Meeting with Antoine
* Multiprocessed data
* Firedux (does multiprocessing for you)
* Make the model as simple as you can (stick to an idea and dont try to
  incorporate everything)
* His idea
  * filtered most of the feature data, stock by stock and specialized
  * one model for each stock
  * used XGBoost (random forest is also good)
  * predicted returns, then inverse filtered to get back to data
  * way to process features
  * need MVP, can find more features if we have time
  * data they gave us already has predictive power
* should have explanation for how we process features, transform features to see
  if they add anything (maybe down the line)
* keep the structure of the code they gave (predicts returns, then constructs
  portfolio)
* different regimes have different viable factors
* quality of slides + model that is original enough but not too complicated
* checking for consistency of features over time, dropping could end up leaving
  the ones that are not important
* non parametric dimension reduction
* he stood out by feature processing with fourier transform
* denoised features have a lot of predictive power
* parquet

### Meeting 9/20
* Leading vs. lagging indicators, should have a mixture?
* macro forecasting (DSGE) for fundamentals, unemployment, inflation
* regime detection
* DCM
  * $\alpha$^2 fund
  * based quantitative model on goyenko's paper
  * not that much hope in using text data for $\alpha$

### Meeting with Lucas
* Judges are not too technical
* They are aware of new technologies but surface level (judges care bout techincals and P/L, they like numbers)
* Glaze Rus, research judges and find their biases (but you must make sure your data has no bias)
* The Pitch is the most important part (what Lucas did is he started with the slides and the idea, and after he made the algo) 
* Regime detection will determine where to cluster or not (helps determine what market we are in) 
* PPO (type of RL) where we optimize sharpe. RL agent learns how to trade. This
  can be a layer in the model
  * RL to find which features are important, smart way to decompose data
  * e.g. extract signals from data (over different forecast horizons, ) 
  * importance is in data manipulation and interpretation of the forecasts
  * kalman filtering
  * look into signal processing
  * you can have repition sequences in signal capturing (days, months, years) and then you get a more accurate prediction
* multiple sub-models that fed into one big model (easy to explain)
* story, managing time, managing resources
* OUR GOAL IS TO SELL A STORY: it must be interpreteable, forecast data and data manipulation (clean + model), what you want to sell is elegance, simplicity and creativity - you should be able to easily explain your approach 
* something they forgot to consider is liquidity
* baysian optimization
* all other teams used neuro networks: we should stand out
* use simple architecture but make it optimized
* "temporal fusion" transformer (latest model from google deepmind)
  * had interpretability
  * transformers with sequential layers
  * make forecasts over longer horizons
  * they used pythorch 
* finding good alternative data
  * WRDS, edgar (SEC gov data), satellite data (image data), betting markets (manifest and crowd source alpha there)
  * weather (for forecasting)
  * showing how retail vs. insititution expectations diverge
  * electric grid transformers report heat data, use to predict price of
    electricity
  * e.g. counting boats in a dock to predict macro movements, "leading
    indicator"
  * weather -> futures -> bonds
* they are looking for us to come up with something and for them to implement it
  * either a model, or a feature. models are hard to implement but features only
    rely on your creativity
  * e.g. the amount of oil in a ship determines how much it floats. use length
    of shadows to determine how much oil, and predict quantity of oil. front-run
    oil market
  * find a place that analysts aren't gonna look
* different places to apply creativity, could be in the data side or in the
  models/approach
* text data: take note of what isn't there, how complex is it (if it's more
  complex they're usually trying to hide something and company is less
  financially healthy)
  * how stressed was the person, how much did they mumble, how many commas are in the text, how long are the sentances (management is a very good indicator - like jerome opening with good morning vs gentlemen) you can also look at deviation and variation and time of when reports are released 
* very high dimensional space, we have to extract signal and denoise
* another alternative source: audio data
* use data to predict volatility regimes
* signal in AI: where things are real vs. not real, what is valuable information
  vs. what isn't. we need to try to remove those useless fluctuations, find the
  most predictive, without overfitting (30 - 60 powerful features)
  * sequences of those features, sliding window vs. expanding window, we should
    choose sliding window
* look at sponsor firms 
* key takeaways: interpretable
* Lucas made 5 models and thats what made him stand out, you want a good thesis and keep it simple - do something that people are scared of doing and try something new
* Nick's firm is trying to act fast on high frequency data 



### Meeting notes (9/14)

**Main Problem** we're trying to solve: portfolio optimization by MVO. For a given
return, we need to minize risk. The problem is formulated as \
$\min_w w^T \Sigma w$ subject to 
$\sum_{i=1}^n w_i = 1 $

We could also have $w_i$
sum to less than 0 if short selling, or less than a leverage ratio $L$. The other constraint is that $w^T \mu = \mu_{target}$.

Where can we apply Machine Learning to MVO?
1. predicting this target return (OOS prediction problem)
2. estimating $\Sigma$, the covariance matrix. 
3. clustering the entities (stocks) to find common relationships

**Another Approach:** Maximizing Sharpe Ratio (SR) directly. The SR is 
$$
\frac{E[r_p] - r_f}{\sigma_p}  \sigma_p 
$$
i.e. standardizes you return based on how much risk you are actually taking.
$\alpha$ is the measure of how much you outperform the maximum SR obtained by
following the market. The market for example can be the return of the S&P 500.

**Prediction problem**
* Elastic Net
* Choosing hyperparameters

**Max SR vs MVO**
* Maximizing SR depends on correctly predicting the target return, and then
  directly finds the highest SR
* MVO provides more leeway by 

**Liz's Colleague**
* Suggested to group stocks into 8 groups based on valuation (not based on sector)
    * Random forest, decision tree are ways of partitioning the stocks
* https://github.com/sinhapgit/Algorithmic-Trading-Strategy-Using-Elastic-Net-and-Random-Forests



### To Do for Tuesday:
1. Talk to guy on tuesday, figure out how he broke down problem
2. Kevin talk to Archer about how to choose ML model for prediction
3. Nikhil can talk to john about approaching problem

---

### To Do:
1. Research which machine learning (factor reduction techniques, e.g. LASSO)
2. Examine data for formatting
