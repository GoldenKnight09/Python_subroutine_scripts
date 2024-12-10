# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:56:31 2024

@author: chris
"""

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

def PLS_CV(factor_list,
           response_list,
           input_data,
           response_var,
           scaler = 'robust',
           split_test_size = 0.15,
           relative_improvement_tol = 0.005, # ratio of (n-(n-1))/n scores for PLS model fits
           max_number_of_comps = 20, # max number of components to fit
           split_random_state = 17,
           cv_random_state = 17):
    '''
    This function is used to transform and input dataframe using partial least squares
    It also splits the input data into training & validation sets
    Inputs:
        factors_list: list of factor (column) names
        response_list: list of response (column) names
        model_data: dataframe of input data (contains both factors & response(s) columns)
        response_var: string that is the (column) name of the response to be used for the transformation
        scaler: string that specifies the factor scaling method to be used; options are:
            minmax: uses MinMaxScaler() (minimum & maximum extremes) to scale the input data
            standard: uses StandardScaler() (removes the mean & scales to unit variance) to scale the input data
            robust: uses RobustScaler() (interquartile range) to scale the input data
        split_test_size: fraction of the data in the validation data set
        relative_improvement_tol: tolerance ratio for improvement between subsequent latent variables
        max_number_of_comps: maximum number of potential latent variables to fit
        split_random_state: random state used for train_test_split
        cv_random_state: random state used for cross-validation step to determine the number of latent variables
    Outputs:
        CV_results: dictionary of raw cross-validation results for each latent variable
        response_num_comps: number of latent variables in the transformation
        X_train_trans: dataframe of transformed input data in the training set
        y_train: dataframe of reponse data in the training set
        X_valid_trans: dataframe of transformed input data in the validation set
        y_valid: dataframe of response data in the validation set
    '''
    response = input_data[response_var]
    factors_df = input_data.drop(columns = response_list)
    X_train, X_valid, y_train, y_valid = train_test_split(factors_df,
                                                          response,
                                                          test_size = split_test_size,
                                                          random_state = split_random_state,
                                                          shuffle = True)
    if scaler == 'minmax':
        pipe = Pipeline(steps = [('scaler',MinMaxScaler())])
    if scaler == 'standard':
        pipe = Pipeline(steps = [('scaler',StandardScaler())])
    if scaler == 'robust':
        pipe = Pipeline(steps = [('scaler',RobustScaler())])
    pipe.fit(X_train, y_train)
    X_train_scaled = pipe.transform(X_train)
    X_valid_scaled = pipe.transform(X_valid)
    relative_improvement = 1 # dummy initial value
    n_comps = 1
    CV_results = {}
    kfold_cv = KFold(shuffle = True,
                     random_state = cv_random_state)
    while relative_improvement > relative_improvement_tol and n_comps < max_number_of_comps + 1:
        pls_n = PLSRegression(n_components = n_comps,
                              scale = False)
        CV_results[n_comps] = cross_val_score(estimator = pls_n,
                                              X = X_train_scaled,
                                              y = y_train,
                                              scoring = 'r2',
                                              cv = kfold_cv,
                                              n_jobs = -3).mean()
        if n_comps > 1:
            relative_improvement = (CV_results[n_comps] - CV_results[n_comps - 1]) / CV_results[n_comps]
        n_comps += 1
    response_num_comps = n_comps - 1
    pls = PLSRegression(n_components = response_num_comps,
                        scale = False)
    X_train_trans, _ = pls.fit_transform(X_train_scaled, y_train)
    X_valid_trans = pls.transform(X_valid_scaled)
    return [CV_results,
            response_num_comps,
            X_train_trans,
            y_train,
            X_valid_trans,
            y_valid]

def main():
    pass

if __name__ == '__main__':
    main()