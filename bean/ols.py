#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLS regression based on statsmodel.api
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy.stats


class OLS:
    def __init__(self,x,y):
           self.y = y 
           self.x = x
           self.intercept = np.ones(len(x))

    
    def fit(self, fit_intercept=True, inverse='pseudo'):
        
        '''
        Fits the OLS model. Calculates ^β, ^y, residuals, error_term, t score, p values and r2.
        Each can be accessed individually or for more readable format use .results() method.
        
        Example
        
        model = OLS(x,y).fit()
        pvals = model.pvals
        t_score= model.t_score
        betas = model.beta_hat
        resid = model.resid
        rsquared = model.r2
        
        Parameters
        -----------
        X (design matrix) and Y (dependent variable)
        
        
        Returns
        --------
        
        self
        
        '''
        
        if fit_intercept == True:
            self.X = np.c_[self.intercept, self.x]
        else:
            self.X = self.x
        self.beta_hat = self.__betas(self.X, self.y,inverse)
        predict= self.__prediction(self.beta_hat, self.X, self.y,)
        self.residuals= predict['residuals']
        self.fitted_response = predict['yhat']
        self.error_term = self.__error(self.residuals,self.X)
        self.t_score = self.__tscore(self.beta_hat, self.error_term)
        self.resid_dof = self.__dof_resid(self.X)
        self.p_values = self.__pval(self.t_score,  self.resid_dof)
        self.squares = self.__sss(self.y, self.fitted_response, fit_intercept)
        self.r2 = self.__r2score(self.squares['rss'],self.squares['tss'])
        self.r2_adjusted = self.__adjusted_r2(self.r2, self.X,fit_intercept)
        self.model_dof = self.__dof_model(self.X, fit_intercept)
        self.mse_model = self.__mse_model(self.squares['ess'], self.model_dof)
        self.mse_resid = self.__mse_residuals(self.squares['rss'], self.resid_dof)
        self.fval = self.__fvalue(self.mse_model, self.mse_resid)
        self.model_pval = self.__pval_for_fval(self.fval, self.model_dof, self.resid_dof)
        
        return self
    
    def results(self):
        
        '''
        Results wrapper in form of pandas dataframe for betas, standard error, t score and pvals.
        Covariates are index (if a numpy array is entered into the model then covariates will be 0,1 etc)
        Needs .fit() method. 
        
        Parameters
        -----------
        self
        
        Returns
        ----------
        pd.DataFrame of betas, standard error, t and p values
        
        '''
        
        covariates=pd.DataFrame(self.x)
        check_x = np.array(self.x)
        check_X = np.array(self.X)
        if (len(check_x[0])) != (len(check_X[0])):
           covariates['intercept']= 1
           cols = covariates.columns.tolist()
           cols = cols[-1:] + cols[:-1]
           covariates = covariates[cols]
        else:
            covariates = covariates
        df = pd.DataFrame({'beta':self.beta_hat, 'error': self.error_term, 't_score': self.t_score, 'pvals':self.p_values},[covariates.columns])
        return df
    
    def diagnostics(self):
        name = pd.DataFrame(self.y).columns
        print('\n','-'*75)
        print(f'\nDiagnostics for {name[0]} model')
        hetero = self.__hetero_ftest(self.x, self.residuals)
        print('\nTest for Heteroskedasticity: fstat',hetero['fstat'], 'pval',hetero['pval'])
        k2, p =scipy.stats.normaltest(self.residuals)
        print('\n', f'Normality of residuals \n K value= {k2}, pval={p}')
        vif =self.__vif(self.x)
        print('\n',vif)
        print('\n','-'*75)
        return self 
    
    def autocorrelation(self):
        plt.figure()
        pd.plotting.autocorrelation_plot(self.residuals)
        plt.show()
        
    def residual_plot(self):
        naming_df = pd.DataFrame(self.y).columns
        data = {f'{naming_df[0]} residuals':self.residuals,f'{naming_df[0]} fitted response':self.fitted_response}
        plotting_df= pd.DataFrame(data)
        plot = plotting_df.columns[0].replace('residuals','')
        import seaborn as sns #lazy import
        sns.jointplot(x=f"{plot}residuals", y= f"{plot}fitted response", data=plotting_df, kind='reg', ci=None)
     
        
    def __betas(self,X, y, inverse='pseudo'):
        
        '''
        Private method. 
        This uses the OLS formula to find  ^β.
        Can be done either using the moore-penrose inverse (^β= V^TSUy ) 
        or the standard inverse (^β = (X^T X)^-1 X^T y)
               
        If model has intercept will calculate betas for the intercept
        
        Parameters
        ----------
        X (design matrix), y (dependent variable)
        
        Returns
        ---------
        ^β
        
        '''
        
        if inverse=='pseudo':
            X = np.array(X)
            u, s, vt = np.linalg.svd(X, 0)
            cutoff = 1e-15 * np.maximum.reduce(s)
            s = np.asarray(list(map(lambda x: 1/x if x > cutoff  else x ==0,s)))
            b_xTx_inverse_xT = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis], np.transpose(u)))
            b_hat = np.dot(b_xTx_inverse_xT, y)
            return b_hat
        elif inverse =='inverse':
            Xt = X.T
            b_xTx_inverse_xT = np.linalg.pinv(Xt.dot(X))
            b_hat = np.dot(b_xTx_inverse_xT, y)
            return b_hat
        else:
            raise ValueError ('inverse has to be set to inverse or pseudo')
        
    def __prediction(self, beta_hat, X, y):
        y_hat= np.dot(beta_hat,X.T)
        residuals = y - y_hat
        prediction = {'yhat':y_hat,'residuals':residuals}
        return prediction
    
    def __error(self,residuals, X):
        sigma2 = np.dot(residuals.T,residuals)/(len(X)-len(X.T))
        vcov_betas = sigma2 * np.linalg.inv(X.T.dot(X))
        standard_error = np.sqrt(np.diag(vcov_betas))
        return standard_error
    
    def __tscore(self,beta_hat,standard_error):
        t= beta_hat/standard_error
        return t
    
    
    def __pval(self, t_score, dof):
        p=2*(1-scipy.stats.t.cdf(np.abs(t_score),dof))
        return p
            
    def __sss(self, y, yhat, fit_intercept=True):
       
        '''
        Private method.
        Calculates total sum of squares (tss), residual sum of squares (rss) and explained sum of squares 
       
        rss = Σ((y - y_hat)**2)
       
        tss changes with intercept:
       
        tss = Σ((y - mean(y))**2)
        
        and without intercept:
       
        tss = Σ((y)**2)
        
        ess = tss - rss
        
        
        Parameters
        ----------
        
        y, yhat, fit_intercept
        
        Returns
        ----------
        
        dictionary of ess, rss,tss
        
        '''
        rss = np.sum((y - yhat)**2)   
        if fit_intercept==True:
            tss = np.sum((y- np.mean(y))**2)
        else:
            print('warning, R2 is computed without centering due to no intercept')
            tss = np.sum((y**2))
        ess = tss - rss
        squares={'ess':ess,'rss':rss,'tss':tss}
        return squares
        
    def __r2score(self, rss, tss):
        
        '''
        Private method.
  
        r2 = 1-rss/tss
        
        Parameters
        ----------
        
        y, yhat,
        
        Returns
        ----------
        
        Rsquared 
        
        '''
        r2 = 1 - (rss/tss)
        return r2
        
    def __adjusted_r2(self,r2, X, fit_intercept=True):
        
        '''
        Private method. Adjusted R squared using:
            
        adjusted_r2 = 1 - ((1-r2) * (n-1))/(n - k - 1)
        
        r2 = rsquared
        n = number of observation in X
        k = number of covariates minus intercept
        
        Parameters
        ---------
        X, r2
        
        Returns
        ---------
        adjusted r2
        '''
        
        n = len(X)
        if fit_intercept==True:
            k = np.linalg.matrix_rank(X) -1
        else:
            k = np.linalg.matrix_rank(X)
        adjusted_r2 = 1 - ((1-r2) * (n-1))/(n - k - 1)
        return adjusted_r2
    
    def __dof_resid(self, X):
        dof_resid = len(X)-np.linalg.matrix_rank(X)
        return dof_resid
        
    def __dof_model(self,X, fit_intercept=True):
        if fit_intercept ==True:
            model_df = np.linalg.matrix_rank(X) -1
        else:
            model_df = np.linalg.matrix_rank(X)
        return model_df
    
    def __mse_model(self, ess, model_df):
        mse_model = ess / model_df
        return mse_model
    
    def __mse_residuals(self, rss, dof_resid):
        mse_resid = rss /dof_resid
        return mse_resid
    
    def __fvalue(self,mse_model,mse_resid):
        fvalue_for_model = mse_model/mse_resid
        return fvalue_for_model
    
    def __pval_for_fval(self,fval,dof_model,dof_resid):
        pval_of_model = scipy.stats.f.sf(fval,dof_model,dof_resid)
        return pval_of_model
    
    def __hetero_ftest(self,X,residuals):
        residuals= residuals **2
        heterosk_model = OLS(X,residuals).fit()
        hetero = {'fstat':heterosk_model.fval, 'pval':heterosk_model.model_pval}
        return hetero
    
    def __vif(self, x):
        X = np.array(x)
        r2_list = [OLS(np.delete(X, i ,axis=1), X[:,i]).fit().r2  for i in range(len(X[1]))]
        vif_list = [1 / (1 - r2) for r2 in r2_list]
        X_df = pd.DataFrame(x)
        vif_df = pd.DataFrame((vif_list),index=[X_df[:0]], columns=['VIF'])
        return vif_df
        
        
        
    
    
        
  


