# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:32:42 2024

@author: chris
"""

import matplotlib.pyplot as plt

def corr_plot(factor_list, response_list, input_data):
    '''
    Generates correlation plots of the input data; can filter for specific factors and/or response(s)
    Inputs:
        factor_list: list of factors
        response_list: list of response(s)
        input_data: dataframe of data to be visualized
    '''
    input_data_corr = input_data.loc[:, factor_list + response_list].corr()
    # maybe use floor division (//) to automatically determine dimension based on number of factors + response(s)
    if len(factor_list + response_list) < 10:
        fig = plt.figure(figsize = [8,5])
        ax = fig.add_subplot()
    elif (len(factor_list + response_list) >= 10) and (len(factor_list + response_list) < 20):
        fig = plt.figure(figsize = [20,16])
        ax = fig.add_subplot()
    elif (len(factor_list + response_list) >= 20) and (len(factor_list + response_list) < 35):
        fig = plt.figure(figsize = [30,24])
        ax = fig.add_subplot()
    elif (len(factor_list + response_list) >= 35) and (len(factor_list + response_list) < 60):
        fig = plt.figure(figsize = [36,30])
        ax = fig.add_subplot()
    else:
        fig = plt.figure(figsize = [50,32])
        ax = fig.add_subplot()
    cax = ax.matshow(input_data_corr, cmap = 'coolwarm', vmin = -1, vmax = 1)
    for i, row in enumerate(factor_list + response_list):
        for j, column in enumerate(factor_list + response_list):
            ax.text(j, i, '{:0.2f}'.format(input_data_corr.loc[row,column]), ha = 'center', va = 'center')
    fig.colorbar(cax)
    plt.xticks(range(input_data_corr.shape[1]),input_data_corr.columns, rotation = 90)
    plt.yticks(range(input_data_corr.shape[0]),input_data_corr.columns)
    plt.show()

def hist_plot(input_data, response_col, bins = 30):
    '''
    Short function to plot histograms of input data for a single given response
    Inputs:
        input_data: dataframe of input data
        response_col: string of response column name in input data
        bins: number of bins to use for histogram plot
    '''
    fig, ax = plt.subplots()
    ax.hist(x = input_data[response_col], bins = bins)
    plt.xlabel(response_col)
    plt.ylabel('Count')
    plt.show()
    
def main_Data_vis():
    pass

if __name__ == '__main__':
    main_Data_vis()