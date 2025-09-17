### Meeting with Lucas
* Judges are not too technical
* They are aware of new technologies but surface level (judges care bout techincals and P/L, they like numbers)
* Glaze Rus, research judges and find their biases
* Regime detection will determine where to cluster or not
* PPO (type of RL) where we optimize sharpe. RL agent learns how to trade. This
  can be a layer in the model
  * RL to find which features are important, smart way to decompose data
  * e.g. extract signals from data (over different forecast horizons, ) 
  * importance is in data manipulation and interpretation of the forecasts
  * kalman filtering
  * look into signal processing
* multiple sub-models that fed into one big model (easy to explain)
* story, managing time, managing resources
* use simple architecture but make it optimized
* "temporal fusion" transformer (latest model from google deepmind)
  * had interpretability
  * transformers with sequential layers
  * make forecasts over longer horizons
* finding good alternative data
  * WRDS, edgar (SEC gov data), satellite data, betting markets
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
* very high dimensional space, we have to extract signal and denoise
* another alternative source: audio data
* use data to predict volatility regimes
* signal in AI: where things are real vs. not real, what is valuable information
  vs. what isn't. we need to try to remove those useless fluctuations, find the
  most predictive, without overfitting (30 powerful features)
  * sequences of those features, sliding window vs. expanding window, we should
    choose sliding window
* look at sponsor firms 
* key takeaways: interpretable



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
