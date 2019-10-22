#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:15:25 2019

@author: nitinsinghal
"""

# Preddict Asset prices and their direction using macroeconomic data using Regression algortihms
# For US, UK, EU

#Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Import the macro and asset data for each country
# Macro data - US, UK, EU.
usmacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/usmacrodata_sep19.csv')
eumacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/eurmacrodata_sep19.csv')
ukmacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/gbpmacrodata_sep19.csv')
# Asset data
nasdaqdata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/Nasdaq.csv')
sp500data = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/SP500.csv')
daxdata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/DAX.csv')
oilwtidata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/OilWTI.csv')
golddata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/Gold.csv')
Russell2000data = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/Russell2000.csv')
CACdata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/CAC40.csv')
#FTSEdata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/FTSE.csv')
WilshireUSRealEstatedata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/WilshireUSRealEstatePriceIndex.csv')
UST10YrPricedata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/UST10YrPrice.csv')
Treasury10Yrdata = pd.read_csv('/Users/nitinsinghal/Dropbox/MachineLearning/PublishedResearch/RoboAdvisoryAlgo/Data/UST10YrRates.csv')

nasdaqdata['Date'] = pd.to_datetime(nasdaqdata['Date'])
nasdaqdata['Date'] = nasdaqdata['Date'].dt.strftime('%Y-%m-%d')
nasdaqdata = nasdaqdata.sort_values(by=['Date']).reset_index(drop=True)

sp500data['Date'] = pd.to_datetime(sp500data['Date'])
sp500data['Date'] = sp500data['Date'].dt.strftime('%Y-%m-%d')
sp500data = sp500data.sort_values(by=['Date']).reset_index(drop=True)

#FTSEdata['Date'] = pd.to_datetime(FTSEdata['Date'])
#FTSEdata['Date'] = FTSEdata['Date'].dt.strftime('%Y-%m-%d')
#FTSEdata = FTSEdata.sort_values(by=['Date']).reset_index(drop=True)

daxdata['Date'] = pd.to_datetime(daxdata['Date'])
daxdata['Date'] = daxdata['Date'].dt.strftime('%Y-%m-%d')
daxdata = daxdata.sort_values(by=['Date']).reset_index(drop=True)

CACdata['Date'] = pd.to_datetime(CACdata['Date'])
CACdata['Date'] = CACdata['Date'].dt.strftime('%Y-%m-%d')
CACdata = CACdata.sort_values(by=['Date']).reset_index(drop=True)

Russell2000data['Date'] = pd.to_datetime(Russell2000data['Date'])
Russell2000data['Date'] = Russell2000data['Date'].dt.strftime('%Y-%m-%d')
Russell2000data = Russell2000data.sort_values(by=['Date']).reset_index(drop=True)

oilwtidata['Date'] = pd.to_datetime(oilwtidata['DATE'])
oilwtidata['Date'] = oilwtidata['Date'].dt.strftime('%Y-%m-%d')
oilwtidata.drop(['DATE'], inplace=True, axis=1)
oilwtidata = oilwtidata.sort_values(by=['Date']).reset_index(drop=True)

golddata['Date'] = pd.to_datetime(golddata['DATE'])
golddata['Date'] = golddata['Date'].dt.strftime('%Y-%m-%d')
golddata.drop(['DATE'], inplace=True, axis=1)
golddata = golddata.sort_values(by=['Date']).reset_index(drop=True)

UST10YrPricedata['Date'] = pd.to_datetime(UST10YrPricedata['Date'])
UST10YrPricedata['Date'] = UST10YrPricedata['Date'].dt.strftime('%Y-%m-%d')
UST10YrPricedata = UST10YrPricedata.sort_values(by=['Date']).reset_index(drop=True)

Treasury10Yrdata['Date'] = pd.to_datetime(Treasury10Yrdata['Date'])
Treasury10Yrdata['Date'] = Treasury10Yrdata['Date'].dt.strftime('%Y-%m-%d')
Treasury10Yrdata = Treasury10Yrdata.sort_values(by=['Date']).reset_index(drop=True)

# Take macro data from 1999 
usmacrodata = usmacrodata[['date','us_gdp_yoy', 'us_industrial_production','us_inflation_rate', 'us_core_pceinflation_rate',
                           'us_interest_rate','us_retail_sales_yoy', 
                           'us_consumer_confidence', 'us_business_confidence', 'us_unemployment_rate', 'us_manufacturing_production']] 
# 'us_manufacturing_pmi', 'us_non_manufacturing_pmi', 
usmacrodata['date'] = pd.to_datetime(usmacrodata['date'])
usmacrodata = usmacrodata[(usmacrodata['date'] > '31/12/1998')]
usmacrodata['date'] = usmacrodata['date'].dt.strftime('%Y-%m-%d')

eumacrodata = eumacrodata[['date','eu_gdp_yoy', 'eu_industrial_production','eu_inflation_rate', 'eu_core_inflation_rate',
                           'eu_interest_rate','eu_manufacturing_production','eu_retail_sales_yoy',
                           'eu_consumer_confidence','eu_business_confidence','eu_unemployment_rate']]
# 'eu_manufacturing_pmi','eu_services_pmi',
eumacrodata['date'] = pd.to_datetime(eumacrodata['date'])
eumacrodata = eumacrodata[(eumacrodata['date'] > '31/12/1998')]
eumacrodata['date'] = eumacrodata['date'].dt.strftime('%Y-%m-%d')

ukmacrodata = ukmacrodata[['date','uk_gdp_yoy', 'uk_industrial_production','uk_inflation_rate', 'uk_core_inflation_rate',
                           'uk_interest_rate','uk_manufacturing_production','uk_retail_sales_yoy',
                           'uk_consumer_confidence','uk_business_confidence','uk_unemployment_rate']]
# 'uk_manufacturing_pmi', 'uk_services_pmi',
ukmacrodata['date'] = pd.to_datetime(ukmacrodata['date'])
ukmacrodata = ukmacrodata[(ukmacrodata['date'] > '31/12/1998')]
ukmacrodata['date'] = ukmacrodata['date'].dt.strftime('%Y-%m-%d')

# merge us, eu, uk macro data files
macro99data = pd.merge(usmacrodata, eumacrodata, how='left', on='date')
macro99data = pd.merge(macro99data, ukmacrodata, how='left', on='date')

macrogdp = macro99data[['date', 'us_gdp_yoy', 'uk_gdp_yoy', 'eu_gdp_yoy', 
                        'us_interest_rate', 'uk_interest_rate', 'eu_interest_rate']]
macrogdp = macrogdp[(macrogdp.iloc[:, 1] != 0)]
macrogdp.plot(x='date', kind='line', figsize=(8,6))

#Use only High price
nasdaqclose = nasdaqdata.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
nasdaqclose = nasdaqclose.rename({'High':'nasdaq'}, axis='columns')
nasdaqclose['nasdaq'] = nasdaqclose['nasdaq'].str.replace(',', '').astype(float)
sp500close = sp500data.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
sp500close = sp500close.rename({'High':'sp500'}, axis='columns')
sp500close['sp500'] = sp500close['sp500'].str.replace(',', '').astype(float)
daxclose = daxdata.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
daxclose = daxclose.rename({'High':'dax'}, axis='columns')
CACclose = CACdata.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
CACclose = CACclose.rename({'High':'CAC'}, axis='columns')
#FTSEclose = FTSEdata.drop(['Open','Close','Low','Adj Close'],axis=1)
#FTSEclose = FTSEclose.rename({'High':'FTSE'}, axis='columns')
#FTSEclose['FTSE'] = FTSEclose['FTSE'].str.replace(',', '').astype(float)
Russell2000close = Russell2000data.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
Russell2000close = Russell2000close.rename({'High':'Russell2000'}, axis='columns')
Treasury10Yrclose = Treasury10Yrdata.drop(['Open','Close','Low','Adj Close', 'Volume'],axis=1)
Treasury10Yrclose = Treasury10Yrclose.rename({'High':'Treasury10Yr'}, axis='columns')
UST10YrPriceClose = UST10YrPricedata.drop(['Open','Price','Low','Vol.'],axis=1)
UST10YrPriceClose = UST10YrPriceClose.rename({'High':'UST10YrPrice'}, axis='columns')

#Rename Close price column to index name
WilshireUSRealEstateclose = WilshireUSRealEstatedata.rename({'WILLRESIPR':'WilshireUSRealEst', 'DATE':'Date'}, axis='columns')
WilshireUSRealEstateclose['WilshireUSRealEst'] = pd.to_numeric(WilshireUSRealEstateclose['WilshireUSRealEst'], errors='coerce')
oilwticlose = oilwtidata.rename({'DCOILWTICO':'oilwti', 'DATE':'Date'}, axis='columns')
oilwticlose['oilwti'] = pd.to_numeric(oilwticlose['oilwti'], errors='coerce')
goldclose = golddata.rename({'GOLDAMGBD228NLBM':'gold', 'DATE':'Date'}, axis='columns')
goldclose['gold'] = pd.to_numeric(goldclose['gold'], errors='coerce')

#Merge asset, stock, oil, gold, treasur index data
Mergedstockindexdata = pd.merge(sp500close, nasdaqclose, how='left', on='Date')
Mergedstockindexdata = pd.merge(Mergedstockindexdata, daxclose, how='left', on='Date')
Mergedstockindexdata = pd.merge(Mergedstockindexdata, CACclose, how='left', on='Date')
Mergedstockindexdata = pd.merge(Mergedstockindexdata, Russell2000close, how='left', on='Date')
#Mergedstockindexdata = pd.merge(Mergedstockindexdata, FTSEclose, how='left', on='Date')
Mergedstockindexdata = Mergedstockindexdata[(Mergedstockindexdata['Date'] < '2019-09-24')]

MergedAssetdata = pd.merge(Mergedstockindexdata, WilshireUSRealEstateclose, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, goldclose, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, oilwticlose, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, Treasury10Yrclose, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, UST10YrPriceClose, how='left', on='Date')
MergedAssetdata = MergedAssetdata.fillna(0)

#Get all merged data from 1999, as most indices have data from then 
MergedAsset99data = MergedAssetdata[(MergedAssetdata['Date'] > '1998-12-31')]
MergedAsset99data.drop_duplicates(subset='Date', keep='first', inplace=True)

MergedAsset99Pricedata = MergedAsset99data.drop(['Treasury10Yr'], axis=1).reset_index(drop=True)

#Plot all asset data with subplot to plot against their own scale on x-axis, same y-axis date
MergedAsset99Pricedata.plot(x='Date', subplots=True, figsize=(8,20))

# Merge each asset price with us macro data
macro99 = macro99data
macro99['Date'] = pd.to_datetime(macro99['date'],format='%Y-%d-%m')
macro99 = macro99.drop(['date'], axis=1)
macro99 = macro99.fillna(0)
# Select macro data upto t-1 for prediction asset price for t
macrodata = macro99[(macro99['Date'] <= '2019-08-01')]
X_validation = macro99[(macro99['Date'] == '2019-08-01')]
X_validation.drop(['Date'], inplace=True, axis=1)
X_validation.reset_index(drop=True, inplace=True)

# Resample asset price data monthly to calculate the mean
indexdata = MergedAsset99Pricedata[~MergedAsset99Pricedata.eq(0).any(1)]
indexdata.set_index(pd.DatetimeIndex(indexdata['Date']), inplace=True)
indexdata.drop(['Date'],inplace=True, axis=1)
indexmthlydataall = indexdata.resample('M').mean()
indexmthlydataall['Date'] = indexmthlydataall.index
indexmthlydataall['Date'] = pd.to_datetime(indexmthlydataall['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
#indexmthlydata = indexmthlydata[1:]
indexmthlydataall.reset_index(drop=True, inplace=True)

# Select individual asset from the merged assetpricedata for prediction
yval = indexmthlydataall[(indexmthlydataall['Date'] == '2019-09-01')]['sp500']
# Take asset price data upto t-1 for training the model
indexmthlydata = indexmthlydataall[(indexmthlydataall['Date'] < '2019-09-01')]
assetmthlypricedata = indexmthlydata[['sp500']]
macromthlydata = macrodata.drop(['Date'], axis=1)
mergedassetmacrodata = pd.concat([macromthlydata, assetmthlypricedata], axis=1)

# Add time lag as required - 6M, 12M, etc
assetlag = assetmthlypricedata#[12:]
assetlag.reset_index(drop=True, inplace=True)
macrolag = macromthlydata#[:-12]
macrolag.reset_index(drop=True, inplace=True)

mergedassetmacrolag = pd.concat([macrolag, assetlag], axis=1)

# Create dependent and independent variables dataframes
Xasset = mergedassetmacrolag.iloc[:, 0:len(mergedassetmacrolag.columns)-1].values
yasset = mergedassetmacrolag.iloc[:,len(mergedassetmacrolag.columns)-1].values

# Train test split for running GridSearchCV on different estimators
X_train, X_test, y_train, y_test = train_test_split(Xasset, yasset, test_size = 0.25)


######## Run Random Forest Algo on Assets with macro data ##########
rfreg = RandomForestRegressor(n_estimators=500, criterion='mse', min_samples_leaf=2, max_depth=15,  
                                min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
rfreg.fit(X_train, y_train)

# Predicting asset value result with Random Forest Regression
rf_pred = rfreg.predict(X_test)

# Check the importance of each feature
rfimp = rfreg.feature_importances_

rfmse = mean_squared_error(y_test , rf_pred)
rfrmse = np.sqrt(rfmse)
rfmae = mean_absolute_error(y_test , rf_pred)
rfr2 = r2_score(y_test , rf_pred)
print('MSE:%.4f, RMSE:%.4f, MAE:%.4f, R2:%.4f' %(rfmse, rfrmse, rfmae, rfr2))
plt.title('Predicted Vs Test Variation')
plt.xlabel('Test Sample Size')
plt.ylabel('Asset Price')
plt.plot(y_test, color='red', label='Test values')
plt.plot(rf_pred, color='blue', label='Predicted values')
plt.legend()
plt.show()

testpreddiff = np.sort(rf_pred-y_test, kind='quicksort')
testpreddiffpct = np.sort((rf_pred-y_test)/rf_pred, kind='quicksort')
plt.title('Predicted Vs Test % Difference')
plt.xlabel('Test Sample Size')
plt.ylabel('% Difference')
#plt.plot(testpreddiff, color='red')
plt.plot(testpreddiffpct, color='blue')
plt.show()

featimp = pd.DataFrame(rfimp)
featimp = featimp.rename({0:'FeatImp'}, axis='columns')
featimp.index = mergedassetmacrolag.columns[0:len(mergedassetmacrolag.columns)-1]
featimp = featimp.sort_values(by=['FeatImp'])
featimp.plot(kind='bar')

rf_validationpred = rfreg.predict(X_validation.iloc[:, 0:len(X_validation.columns)].values)
rf_validationact = yval
valpreddiff = (rf_validationpred-rf_validationact)
valpreddiffpct = ((rf_validationpred-rf_validationact)/rf_validationpred)
print('act:%.3f, pred:%.3f, diff:%.3f, diffpct:%.3f' % 
      (rf_validationact, rf_validationpred, valpreddiff, valpreddiffpct))

######## Run XGBoost Algo on Assets with macro data ##########
xgbreg = XGBRegressor(objective ='reg:squarederror', learning_rate=0.1, max_depth=4, seed=1)
xgbreg.fit(X_train, y_train)

# Predicting asset value result with Random Forest Regression
xgb_pred = xgbreg.predict(X_test)

# Check the importance of each feature
xgbimp = xgbreg.feature_importances_

xgbmse = mean_squared_error(y_test , xgb_pred)
xgbrmse = np.sqrt(xgbmse)
xgbmae = mean_absolute_error(y_test , xgb_pred)
xgbr2 = r2_score(y_test , xgb_pred)
print('MSE:%.4f, RMSE:%.4f, MAE:%.4f, R2:%.4f' %(xgbmse, xgbrmse, xgbmae, xgbr2))
plt.title('Predicted Vs Test Variation')
plt.xlabel('Test Sample Size')
plt.ylabel('Asset Price')
plt.plot(y_test, color='red', label='Test values')
plt.plot(xgb_pred, color='blue', label='Predicted values')
plt.legend()
plt.show()

testpreddiff = np.sort(xgb_pred-y_test, kind='quicksort')
testpreddiffpct = np.sort((xgb_pred-y_test)/xgb_pred, kind='quicksort')
plt.title('Predicted Vs Test % Difference')
plt.xlabel('Test Sample Size')
plt.ylabel('% Difference')
#plt.plot(testpreddiff, color='yellow')
plt.plot(testpreddiffpct, color='blue')
plt.show()

featimp = pd.DataFrame(xgbimp)
featimp = featimp.rename({0:'FeatImp'}, axis='columns')
featimp.index = mergedassetmacrolag.columns[0:len(mergedassetmacrolag.columns)-1]
featimp = featimp.sort_values(by=['FeatImp'])
featimp.plot(kind='bar')

xgb_validationpred = xgbreg.predict(X_validation.iloc[:, 0:len(X_validation.columns)].values)
xgb_validationact = yval
valpreddiff = (xgb_validationpred-xgb_validationact)
valpreddiffpct = ((xgb_validationpred-xgb_validationact)/xgb_validationpred)
print('act:%.3f, pred:%.3f, diff:%.3f, diffpct:%.3f' % 
      (xgb_validationact, xgb_validationpred, valpreddiff, valpreddiffpct))

# Create the pipeline t0 run gridsearchcv for best estimator and hyperparameters
pipe_rf = Pipeline([('rgr', RandomForestRegressor(random_state=42))])

pipe_svr = Pipeline([('rgr', SVR())])

pipe_mlr = Pipeline([('rgr', LinearRegression())])

pipe_xgb = Pipeline([('rgr', XGBRegressor(objective ='reg:squarederror'))])

# Set grid search params
grid_params_rf = [{'rgr__n_estimators' : [500],
                   'rgr__criterion' : ['mse'], 
                   'rgr__min_samples_leaf' : [2,3,4], 
                   'rgr__max_depth' : [15,16,17],
                   'rgr__min_samples_split' : [2,3,4],
                   'rgr__max_features' : ['sqrt', 'log2']}]

grid_params_svr = [{'rgr__kernel' : ['rbf','sigmoid'],
                    'rgr__gamma' : ['scale'],
                    'rgr__C' : [10,11,12]}]

grid_params_mlr = [{'rgr__fit_intercept' : ['True', 'False'],
                    'rgr__normalize' : ['False', 'True']}]

grid_params_xgb = [{'rgr__learning_rate' : [0.05,0.1,0.2],
                    'rgr__max_depth' : [3,4,5],
                    'rgr__seed' : [1,2,3]}]

# Create grid search
gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=grid_params_rf,
                     scoring='neg_mean_squared_error',
                     iid=False,
                     cv=10,
                     n_jobs=-1)

gs_svr = GridSearchCV(estimator=pipe_svr,
                      param_grid=grid_params_svr,
                      scoring='neg_mean_squared_error',
                      iid=False,
                      cv=10,
                      n_jobs=-1)

gs_mlr = GridSearchCV(estimator=pipe_mlr,
                      scoring='neg_mean_squared_error',
                      param_grid=grid_params_mlr,
                      iid=False,
                      cv=10,
                      n_jobs=-1)

gs_xgb = GridSearchCV(estimator=pipe_xgb,
                      param_grid=grid_params_xgb,
                      scoring='neg_mean_squared_error',
                      iid=False,
                      cv=10,
                      n_jobs=-1)

# List of grid pipelines
grids = [gs_rf, gs_svr, gs_mlr, gs_xgb] 
# Grid dictionary for pipeline/estimator
grid_dict = {0:'RandomForestRegressor', 1:'SupportVectorRegression', 2: 'MultipleLinearRegression', 
             3: 'XGBoostRegressor'}

# Fit the pipeline of estimators using gridsearchcv
print('Fitting the gridsearchcv to pipeline of estimators...')
mse=0.0
rmse=0.0
mae=0.0
r2 = 0.0
resulterrorgrid = {}
testpreddiffvals = {}
testpreddiffpctvals = {}

for gsid,gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[gsid])
    gs.fit(X_train, y_train)
    print('\n Best score : %.5f' % gs.best_score_)
    print('\n Best grid params: %s' % gs.best_params_)
    y_pred = gs.predict(X_test)
    mse = mean_squared_error(y_test , y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test , y_pred)
    r2 = r2_score(y_test , y_pred)
    resulterrorgrid[gsid,'mse'] = mse
    resulterrorgrid[gsid,'rmse'] = rmse
    resulterrorgrid[gsid,'mae'] = mae
    resulterrorgrid[gsid,'r2'] = r2
    testpreddiff = np.sort(y_pred-y_test, kind='quicksort')
    testpreddiffpct = np.sort((y_pred-y_test)/y_pred, kind='quicksort')
    testpreddiffvals[gsid] = testpreddiff
    testpreddiffpctvals[gsid] = testpreddiffpct
    print('\n Test set accuracy for best params MSE:%.4f, RMSE:%.4f, MAE:%.4f, R2:%.4f' 
          %(mse, rmse, mae, r2))
    plt.plot(testpreddiffpct, color='blue')


# Check correlation for different 10 year buckets for monthly average price of assets
assetmacro = mergedassetmacrodata
assetmacro9908 = assetmacro[(assetmacro['Date'] <= '2008-12-01')]
assetmacro0919 = assetmacro[(assetmacro['Date'] > '2008-12-01')]

# Calculate correlation of mthly avg price with main macro data
assetmacroR = assetmacro.corr(method='pearson')
assetmacro9908R = assetmacro9908.corr(method='pearson')
assetmacro0919R = assetmacro0919.corr(method='pearson')
assetmacroR2 = assetmacroR.pow(2)
assetmacroCov = assetmacro.cov()
assetmacro9908R2 = assetmacro9908R.pow(2)
assetmacro0919R2 = assetmacro0919R.pow(2)
sns.heatmap(assetmacroR, annot=True, fmt=".2f")
sns.heatmap(assetmacro9908R, annot=True, fmt=".2f")
sns.heatmap(assetmacro0919R, annot=True, fmt=".2f")

# Plot main asset monthly prices vs main macro
assetmacro.plot(x='Date')

macrogold = assetmacro[['Date', 'sp500', 'us_interest_rate', 'uk_interest_rate', 'eu_interest_rate']]
macrogold = macrogold[(macrogold.iloc[:, 1] != 0)]
macrogold.plot(x='Date', kind='line', subplots=True, figsize=(8,6))
sns.heatmap(macrogold.corr(method='pearson'), annot=True, fmt=".2f")

macrogold9908 = assetmacro9908[['Date', 'sp500', 'us_interest_rate', 'uk_interest_rate', 'eu_interest_rate']]
macrogold9908 = macrogold9908[(macrogold9908.iloc[:, 1] != 0)]
macrogold9908.plot(x='Date', kind='line', subplots=True, figsize=(8,6))

macrogold0919 = assetmacro0919[['Date', 'sp500', 'us_interest_rate', 'uk_interest_rate', 'eu_interest_rate']]
macrogold0919 = macrogold0919[(macrogold0919.iloc[:, 1] != 0)]
macrogold0919.plot(x='Date', kind='line', subplots=True, figsize=(8,6))

macrogold['avg_int_rate'] = (macrogold['us_interest_rate']+macrogold['uk_interest_rate']+macrogold['eu_interest_rate'])/3
macrogold.plot(x='Date', kind='line', subplots=True, figsize=(8,6))





