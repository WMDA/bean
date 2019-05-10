# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:18:24 2019

@author: k1812017


========================
Bean Statistical libary

based on anaconda

=======================

"""


from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def functions():
    print('Bean Statistical Library functions','\n','-'*140,'\n', 'ttest(): Needs arguments X and Y' , '\n' ,'mannwhitneyu(): Needs arguments X and Y' , '\n' ,'levene(): Levene test for homogenetiy, Need argument X', '\n', 'shapiro(): Shapiro test for normalicy, Needs argument X', '\n', 'ols(): OLS linear regression, Needs a target(deoendent variable) and Covariates (Independent variables), returns regression and assumption checking ', '\n', 'RLM(): Robust regression, Needs a target(deoendent variable) and Covariates (Independent variables), returns regression and assumption checking', '\n', 'whichtest(): returns if a parametric or non parametric test should be used','\n', 'difference(): similar to whichtest but will perform an indepdent t test or mannwhitney U dependent upon test results','\n','-'*140)

def ttest(x, y):
    stat, p = stats.ttest(x, y)
    print('Independent samples t test:  ', '\n' 'T-value= %.3f,' 'p= %.3f' % (stat, p),'\n')

def mannwhitneyu (x, y):
    stat, p = stats.mannwhitneyu(x, y)
    print('Mann-Whitney U:' ,'\n', 'Statistics= %.3f ,' 'p= %.3f' % (stat, p),'\n')
    
def levene(x, y):
    stat, p = stats.levene(x, y)
    print('Levene test for Homogeneity:  ' '\n' 'Statistics= %.3f ,' 'p= %.3f' % (stat, p),'\n')

def shapiro(x):
    stat, p =  stats.shapiro(x)
    print('Shapiro for normaility :  ' 'Statistics= %.3f ,' 'p= %.3f' % (stat, p),'\n')

def whichtest(x, y):
    stat1, p1=  stats.levene(x, y)
    stat2, p2 = stats.shapiro(x)
    stat3, p3 =stats.shapiro(y)
    if p1 and p1 and p3 >0.05:
        print('Use parametric test')
    else:
        print('Use non Parametric test')
 
def difference(x, y):
    stat1, p1=  stats.levene(x, y)
    stat2, p2 = stats.shapiro(x)
    stat3, p3 =stats.shapiro(y)
    if p1 and p1 and p3 >0.05:
        stat4, p4 = stats.ttest_ind(x, y)
        print('Used Independent T-Test, based upon testing','\n',' Indepndent Sample Test:' ,'\n', 'T-Statsistic= %.3f,' ' p= %.3f' % (stat4, p4),'\n')    
    else:
       stat5, p5 = stats.mannwhitneyu(x, y)
       print('Used  mann-whitney-U, based upon testing','\n',' Mann-Whitney U:' ,'\n', 'Statsistic= %.3f,' ' p= %.3f' % (stat5, p5),'\n')    

'''
Linear regression
'''
def ols(target, covariates,):
    covariates = sm.add_constant(covariates)
    model = sm.OLS(target, covariates).fit()
    predictions = model.predict(covariates)
    print(model.summary())
    residual = sm.regression.linear_model.RegressionResults.resid(model)
    lm, lm_p, fvalue, f_pvalue = sm.stats.diagnostic.het_breuschpagan(residual, covariates)
    print('\n','Heteroskedasticity', '\n', 'lagrange multiplier statistic = %.3f , p= %.3f , F value= %.3f, p value for f value = %.3f' % (lm, lm_p, fvalue, f_pvalue),'\n')
    k2, p =stats.normaltest(residual)
    print('\n', 'Normality of residuals', '\n', 'K value= %.3f, p=%.3f' % (k2, p))
    print('\n','R2 Score: ', '\n', r2_score(target, predictions), '\n')
    vif = pd.DataFrame()
    vif["features"] = covariates.columns
    vif["VIF Factor"] = [variance_inflation_factor(covariates.values, i) for i in range(covariates.shape[1])]
    print('VIF ', '\n', vif)

def RLM(target, covariates,):
    covariates = sm.add_constant(covariates)
    model = sm.RLM(target, covariates).fit()
    predictions = model.predict(covariates)
    print(model.summary())
    residual = sm.regression.linear_model.RegressionResults.resid(model)
    lm, lm_p, fvalue, f_pvalue = sm.stats.diagnostic.het_breuschpagan(residual, covariates)
    print('\n','Heteroskedasticity', '\n', 'lagrange multiplier statistic = %.3f , p= %.3f , F value= %.3f, p value for f value = %.3f' % (lm, lm_p, fvalue, f_pvalue),'\n')
    k2, p =stats.normaltest(residual)
    print('\n', 'Normality of residuals', '\n', 'K value= %.3f, p=%.3f' % (k2, p))
    print('\n','R2 Score: ', '\n', r2_score(target, predictions), '\n')
    vif = pd.DataFrame()
    vif["features"] = covariates.columns
    vif["VIF Factor"] = [variance_inflation_factor(covariates.values, i) for i in range(covariates.shape[1])]
    print('VIF ', '\n', vif)

