### Meeting notes (9/14)

**Main Problem** we're trying to solve: portfolio optimization by MVO. For a given
return, we need to minize risk. The problem is formulated as
$$\min_w w^T \Sigma w$$ 

subject to 
$$\sum_{i=1}^n w_i = 1 $$
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
