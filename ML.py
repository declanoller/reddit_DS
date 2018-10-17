import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import FileSystemTools as fst
import os
from tabulate import tabulate
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNetCV, LassoCV, LassoLarsCV, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
#import xgboost as xgb
import lightgbm as lgb


# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
def custom_metric(y_pred, train_data):
    y_true = train_data.get_label()
    diff = np.abs(y_true - y_pred)
    diff[diff>12] = 24 - diff
    err=np.mean(diff)
    return 'custom diff error',err,False


class ML:

    def __init__(self, data_dir):

        self.dir = data_dir
        #Right now I've just made the main dataframe .csv (with the data for all subreddits)
        #start with 'aggregate',
        #so it will look for that, but we'll probably want to change that in the
        #future.
        file_list = glob.glob(fst.addTrailingSlashIfNeeded(self.dir) + 'aggregate' + '*')
        print(file_list)

        if len(file_list) == 1:
            self.csv_file = file_list[0]
        else:
            print('Either no aggregate file, or too many. Exiting.')
            exit(0)

        self.df = pd.read_csv(self.csv_file, index_col=0)
        #self.prettyPrintDB(self.df)

        #There's probably a cleaner way to do this, but it should work for now.
        #It splits the file name, then looks for the part with 'bins', then takes the number before it.
        fname_parts = fst.fnameFromFullPath(self.csv_file).split('_')
        bins_part = [x for x in fname_parts if 'bins' in x]
        assert len(bins_part)==1, print('bins_part is length {}, need len 1.'.format(len(bins_part)))
        bins_part = bins_part[0]
        self.N_bins = int(bins_part.replace('bins',''))
        print(self.N_bins)

        self.bin_names_ordered = ['bin'+str(i) for i in range(self.N_bins)]

        '''
        for now, self.df will be the main DF that's read in. Let's keep it like that,
        and if you want to make any functions that clean/preprocess it before doing
        fitting, make a new df (a copy of it) called self.df_PP or something.

        '''

        #number of users in comment following time zone
        self.region_tz_dict = {
        'anchorange' : -9,
        'losangeles' : -8, #124k
        'sanfrancisco' : -8,
        'vancouver' : -8,
        'denver' : -7, #77k
        'dallas' : -6,
        'chicago' : -6, #124k
        'mexicocity' : -6,
        'nyc' : -5, #154k
        'boston' : -5, #100k
        'washingtondc' : -5, #63k
        'puertorico' : -4,
        'buenosaires' : -3, #2k
        'london' : 0,
        'unitedkingdom' : 0, #210k
        'ireland' : 0,
        'paris' : 1, #19k
        'france' : 1, #220k
        'geneva': 1,
        'spain' : 1, #19k
        'italy' : 1, #122k
        'greece' : 2, #43k
        'romania' : 2,
        'saudiarabia' : 3, #7k
        'pakistan' : 5, #19k
        'india' : 5.5, #153k
        'thailand' : 7,
        'vietnam' : 7, #18k
        'indonesia' : 7, #31k
        'bali' : 8, 
        'beijing' : 8,
        'shanghai' : 8, #10k
        'perth' : 8.75,
        'taiwan' : 8, #20k
        'japan' : 9, #138k
        'korea' : 9, #60k
        'sydney' : 10,
        'melbourne' : 10,
        'newzealand' : 12 #92k
        }



    def prettyPrintDB(self,df):

        print(tabulate(df.head(), headers=df.columns.values, tablefmt='psql'))
        print('\n')
        print('columns: ', df.columns)


    def preprocessing():
        pass


    def addTzCol(self):

        if 'tz' not in self.df.columns.values:
            region_vals = self.df['subreddit'].unique()
            for reg in region_vals:
                assert reg in self.region_tz_dict.keys(), print('Region {} in dataframe not in region_tz_dict!'.format(reg))

            self.df['tz'] = self.df['subreddit'].apply(lambda x: self.region_tz_dict[x])

        else:
            print('Dataframe already has tz column!')


    def trainTestSplit(self):

        self.addTzCol()
        #This does a TT split, and returns COPIES of the split DF's, so use them locally like that.
        #It's not doing any K-fold cross val stuff, so we should do that later. This is just quick and dirty.
        X_tr, X_test, y_tr, y_test = train_test_split(self.df.drop(['user','subreddit','tz'], axis=1), self.df['tz'], test_size=0.3, random_state=42)

        print("shape of X_tr: {}".format(X_tr.shape))
        print("shape of X_test: {}".format(X_test.shape))
        print("shape of y_tr: {}".format(y_tr.shape))
        print("shape of y_test: {}".format(y_test.shape))

        return(X_tr, X_test, y_tr, y_test)



    def simpleLinReg(self):

        X_tr, X_test, y_tr, y_test = self.trainTestSplit()

        lr = LinearRegression()
        lr.fit(X_tr, y_tr)

        print("\nLR train score: {}".format(lr.score(X_tr,y_tr)))
        print("LR test score: {}".format(lr.score(X_test,y_test)))



    def run_lightgbm(self):

        x_train,x_val,y_train,y_val = self.trainTestSplit()
        lgb_train = lgb.Dataset(x_train, label=y_train)
        lgb_test = lgb.Dataset(x_val, label=y_val)

        evals_result={}
        lgb_params = {
                       'objective': 'mse',
                       'metric': 'custom_metric',
                       'nthread':4, 
                       'learning_rate': 0.03, 
                       'verbose':1,
                       'min_data':2,
                       'min_data_in_bin':1,
                      }

        num_boost_round = 300
        verbose_eval = int(num_boost_round/5)
        model = lgb.train(lgb_params, 
                          lgb_train,
                          valid_sets=[lgb_train, lgb_test],
                          valid_names=['train','eval'],
                          num_boost_round=num_boost_round,
                          evals_result=evals_result,
                          early_stopping_rounds=100,
                          verbose_eval=verbose_eval,
                          feval=custom_metric,
                          )

        print('Plot metrics recorded during training...')

        ax = lgb.plot_metric(evals_result, metric='custom diff error')
        #if(saveplots):plt.savefig(saveFolder+"/"+"lgb_plot_metric_"+saveName+".pdf")

        print('Plot feature importances...')
        ax = lgb.plot_importance(model, max_num_features=x_val.shape[1])
        # if(saveplots):plt.savefig(saveFolder+"/"+"lgb_plot_importance_"+saveName+".pdf")

        plt.show()



    def plotUserPostTimesSub(self, subreddit):

        fig, ax = plt.subplots(1,1,figsize=(8,8))

        df_subreddit = self.df[self.bin_names_ordered][self.df['subreddit']==subreddit]

        for i in range(len(df_subreddit)):
            dat = df_subreddit.iloc[i].values
            ax.plot(dat)

        ax.set_xlim(0, self.N_bins)
        ax.set_xlabel('Hour (24H)')
        ax.set_ylabel('Post frequency')
        ax.set_title('Post-time maxs dist. for subreddit: ' + subreddit)
        plt.show()
        #plt.savefig('savefigs/{}_posttimes_{}.png'.format(subreddit,fst.getDateString()))


    def postAvgTimesByRegion(self, type='horizontal'):

        #This plots the df, plotting the distributions for each subreddit, averaged over all
        #the users for that subreddit.

        fig, ax = plt.subplots(1,1,figsize=(12,8))

        unique_subs = self.df['subreddit'].unique()

        for sub in unique_subs:

            df_subreddit = self.df[self.bin_names_ordered][self.df['subreddit']==sub]
            df_subreddit_sum = df_subreddit.sum()
            ax.plot((df_subreddit_sum/df_subreddit_sum.sum()).values, label=sub+' ('+str(self.region_tz_dict[sub])+')')

        ax.legend()
        ax.set_xlim(0, self.N_bins)
        ax.set_xlabel('Hour (24H)')
        ax.set_ylabel('Post frequency')
        #plt.savefig('savefigs/{}_posttimes_{}.png'.format('_'.join(region_list),fst.getDateString()))
        plt.show()



    def plotUserPostTimes(self, user):

        #You can pass it either a single user, or a list of users.

        fig, ax = plt.subplots(1,1,figsize=(8,6))

        if len(user)==0:
            dat = self.df[self.bin_names_ordered][self.df['user']==user].values[0]
            ax.plot(dat, color='tomato', marker='o')
            ax.set_title('Post-time dist. for user ' + user)
        else:
            for u in user:
                dat = self.df[self.bin_names_ordered][self.df['user']==u].values[0]
                ax.plot(dat)
            ax.set_title('Post-time dist. for users ' + ', '.join(user))


        ax.set_xlim(0, self.N_bins)
        ax.set_xlabel('Hour (24H)')
        ax.set_ylabel('Post frequency')
        #plt.savefig('savefigs/{}_posttimes.png'.format(user))
        plt.show()


































#
