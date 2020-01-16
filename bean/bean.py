#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import math
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression, RANSACRegressor
import warnings
warnings.filterwarnings('ignore')




def functions():
    print('Bean Statistical Library functions','\n','-'*140,'\n', 'ttest(): Needs arguments X and Y' , '\n' ,'mannwhitneyu(): Needs arguments X and Y' , '\n' ,'levene(): Levene test for homogenetiy, Need argument X', '\n', 'shapiro(): Shapiro test for normalicy, Needs argument X', '\n', 'ols(): OLS linear regression, Needs a target(deoendent variable) and Covariates (Independent variables), returns regression and assumption checking ', '\n', 'RLM(): Robust regression, Needs a target(deoendent variable) and Covariates (Independent variables), returns regression and assumption checking', '\n', 'whichtest(): returns if a parametric or non parametric test should be used','\n', 'difference(): similar to whichtest but will perform an indepdent t test or mannwhitney U dependent upon test results','\n','loools: leave one out ols regression, takes, target, features, guess wiith output if model_print=True','\n','-'*140)



def ttest(x, y):
    stat, p = stats.ttest_ind(x, y)
    print('Independent samples t test:  ', '\n' 'T-value= %.3f ,' 'p= %.3f' % (stat, p),'\n')

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

def ANOVA(Indyvariable, dependent, df, verbose=False):
    if verbose==False:
        results = ols('{Indyvariable} ~ {dependent}'.format(Indyvariable=Indyvariable,dependent=dependent), data=df).fit()
        aov_table= sm.stats.anova_lm(results, typ=2)
        def anova_table(aov):
            aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
            aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
            aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
            cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
            aov = aov[cols]
            return aov
        results= anova_table(aov_table)
        print(results)
        return results
    else:
        print('One Way ANOVA based on statsmodel. Needs independent and dependent variable in string. Prints out ANOVA table, however if assigned to a varable will return ANOVA table')
'''
Linear regression
'''
def ols(target, covariates):
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

'''
Machine Learning

'''

def radsvr(target, features, guess, splits=10, random_state=0, model_print=False, shuffle=False):
    best_svr =SVR(kernel='rbf')
    cv = KFold(splits, shuffle, random_state)
    for train_index, test_index in cv.split(features):
        X_train, X_test, y_train, y_test= features[train_index], features[test_index], target[train_index], target[test_index]
        best_svr.fit(X_train,y_train)
    model = best_svr.predict(guess)
    y_pred= cross_val_predict(best_svr, features, target, cv=10)
    print('Variance Score:', explained_variance_score(target, y_pred), '\n')
    mse=  mean_squared_error(target, y_pred)
    print('Mean Squared error :', mse, '\n')
    print('Root Mean Squared Error :' ,math.sqrt(mse))
    
    if model_print ==False:
        print('\n')
    elif model_print==True:
            print ('\n','Predicitions:' '\n', model,'\n')
    else:
         raise NameError ('Needs True or False')
        


def linearsvr(target, features, guess, splits=10, random_state=0, model_print=False, shuffle=False):
    best_svr =SVR(kernel='linear')
    cv = KFold(splits, shuffle, random_state)
    for train_index, test_index in cv.split(features):
        X_train, X_test, y_train, y_test= features[train_index], features[test_index], target[train_index], target[test_index]
        best_svr.fit(X_train,y_train)
    model = best_svr.predict(guess)
    y_pred= cross_val_predict(best_svr, features, target, cv=10)
    print('Variance Score:', explained_variance_score(target, y_pred), '\n')
    mse=  mean_squared_error(target, y_pred)
    print('Mean Squared error :', mse, '\n')
    print('Root Mean Squared Error :' ,math.sqrt(mse))
    
    if model_print ==False:
        print('\n')
    elif model_print==True:
            print ('\n','Predicitions:' '\n', model,'\n')
    else:
         raise NameError ('Needs True or False')
         

def polysvr(target, features, guess, splits=10, random_state=0, model_print=False, shuffle=False):
    best_svr =SVR(kernel='poly')
    cv = KFold(splits, shuffle, random_state)
    for train_index, test_index in cv.split(features):
        X_train, X_test, y_train, y_test= features[train_index], features[test_index], target[train_index], target[test_index]
        best_svr.fit(X_train,y_train)
    model = best_svr.predict(guess)
    y_pred= cross_val_predict(best_svr, features, target, cv=10)
    print('Variance Score:', explained_variance_score(target, y_pred), '\n')
    mse=  mean_squared_error(target, y_pred)
    print('Mean Squared error :', mse, '\n')
    print('Root Mean Squared Error :' ,math.sqrt(mse))
    
    if model_print ==False:
        print('\n')
    elif model_print==True:
            print ('\n','Predicitions:' '\n', model,'\n')
    else:
         raise NameError ('Needs True or False')
         

def linear(target, features, guess, splits=10, random_state=0, model_print=False, shuffle=False):
    ols =LinearRegression()
    cv = KFold(splits, shuffle, random_state)
    for train_index, test_index in cv.split(features):
        X_train, X_test, y_train, y_test= features[train_index], features[test_index], target[train_index], target[test_index]
        ols.fit(X_train,y_train)
    model = ols.predict(guess)
    y_pred= cross_val_predict(ols, features, target, cv=10)
    print('Variance Score:', explained_variance_score(target, y_pred), '\n')
    mse=  mean_squared_error(target, y_pred)
    print('Mean Squared error :', mse, '\n')
    print('Root Mean Squared Error :' ,math.sqrt(mse))
    
    if model_print ==False:
        print('\n')
    elif model_print==True:
            print ('\n','Predicitions:' '\n', model,'\n')
    else:
         raise NameError ('Needs True or False')

def loools(target, features, guess, model_print=False):
    ols =LinearRegression()
    loo= LeaveOneOut()
    loo.get_n_splits(features)
    for train_index, test_index in loo.split(features):
       
       X_train, X_test = features[train_index], features[test_index]
       y_train, y_test = target[train_index], target[test_index]
       ols.fit(X_test, y_test)
    model= ols.predict(guess)
    y_pred= cross_val_predict(ols, features, target)
    print('Variance Score:', explained_variance_score(target, y_pred), '\n')
    mse=  mean_squared_error(target, y_pred)
    print('Mean Squared error :', mse, '\n')
    print('Root Mean Squared Error :' ,math.sqrt(mse), '\n')
  
    if model_print ==False:
        print('\n')
    elif model_print==True:
        print ('\n','Predicitions:' '\n', model,'\n')
    else:
         raise NameError ('Needs True or False')
