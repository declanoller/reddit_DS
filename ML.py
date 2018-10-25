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
from sklearn.metrics import make_scorer
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
            print('Either no aggregate file, or too many.')
            print('Attempting to make aggregate file...')
            self.csv_file = self.createAggFile()

        self.df = pd.read_csv(self.csv_file, index_col=0)
        #self.prettyPrintDB(self.df)

        '''#There's probably a cleaner way to do this, but it should work for now.
        #It splits the file name, then looks for the part with 'bins', then takes the number before it.
        fname_parts = fst.fnameFromFullPath(self.csv_file).split('_')
        bins_part = [x for x in fname_parts if 'bins' in x]
        assert len(bins_part)==1, print('bins_part is length {}, need len 1.'.format(len(bins_part)))
        bins_part = bins_part[0]
        self.N_bins = int(bins_part.replace('bins',''))
        print(self.N_bins)'''

        self.N_bins = len([x for x in self.df.columns.values if 'bin' in x])


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
        'seattle' : -8,
        'saltlakecity' : -8,
        'vancouver' : -8,
        'denver' : -7, #77k
        'dallas' : -6,
        'houston' : -6,
        'chicago' : -6, #124k
        'mexicocity' : -6,
        'nyc' : -5, #154k
        'boston' : -5, #100k
        'washingtondc' : -5, #63k
        'detroit' : -5,
        'puertorico' : -4,
        'venezuela' : -4,
        'brazil': -3, #(-4,3)
        'buenosaires' : -3, #2k
        'riodejaneiro' : -3,
        'london' : 0,
        'unitedkingdom' : 0, #210k
        'ireland' : 0,
        'dublin' : 0,
        'paris' : 1, #19k
        'france' : 1, #220k
        'geneva': 1,
        'spain' : 1, #19k
        'italy' : 1, #122k
        'geneva' : 1,
        'greece' : 2, #43k
        'romania' : 2,
        'saudiarabia' : 3, #7k
        'turkey' : 3,
        'moscow' : 3,
        'oman' : 4,
        'pakistan' : 5, #19k
        'india' : 5.5, #153k
        'kazakhstan' : 6,
        'thailand' : 7,
        'bangkok' : 7,
        'vietnam' : 7, #18k
        'indonesia' : 7,#(7,8,9), #31k
        'bali' : 8, 
        'beijing' : 8,
        'shanghai' : 8, #10k
        'perth' : 8.75,
        'taiwan' : 8, #20k
        'japan' : 9, #138k
        'korea' : 9, #60k
        'seoul' : 9,
        'sydney' : 10,
        'melbourne' : 10,
        'tasmania' : 11,
        'newzealand' : 12 #92k
        }



    def createAggFile(self, dir=None):

        #In the future, all data collection runs should create an aggregate file, but for older
        #ones, or maybe combining individual runs together, this function will take several CSV files
        #together and create an aggregate one.

        if dir is None:
            dir = self.dir

        file_list = glob.glob(fst.addTrailingSlashIfNeeded(dir) + '*.csv')
        #print(file_list)

        assert len(file_list)>=1, 'No CSV files to create agg file in dir, exiting.'

        agg_df = pd.read_csv(file_list[0], index_col=0)

        for i, file in enumerate(file_list):
            if i>0:
                next_df = pd.read_csv(file, index_col=0)
                agg_df = agg_df.append(next_df, ignore_index=True)

        csv_file = fst.combineDirAndFile(dir, 'aggregate.csv')
        agg_df.to_csv(csv_file)
        print('aggregate file {} created.'.format(csv_file))
        return(csv_file)



    def prettyPrintDB(self, df):

        self.addTzCol()

        print(tabulate(df.head(), headers=df.columns.values, tablefmt='psql'))
        print('\n')

        print('\nColumns: ', df.columns)

        print('\nUnique subreddits: ',df['subreddit'].unique())

        print('\n\nUsers per subreddit: ')
        sub_grouped = df.groupby(['subreddit'])[['subreddit']].count()
        print(tabulate(sub_grouped, headers=['subreddit', 'counts'], tablefmt='psql'))

        print('\n\nUsers per timezone: ')
        sub_grouped = df.groupby(['tz'])[['tz']].count()
        print(tabulate(sub_grouped, headers=['tz', 'counts'], tablefmt='psql'))


    def preprocessing():
        pass


    def addTzCol(self):

        #This adds the timezone to the df, using the dict in init(). It's okay to call it repeatedly
        #because it first checks if it's already there.

        if 'tz' not in self.df.columns.values:
            region_vals = self.df['subreddit'].unique()
            for reg in region_vals:
                assert reg in self.region_tz_dict.keys(), print('Region {} in dataframe not in region_tz_dict!'.format(reg))

            self.df['tz'] = self.df['subreddit'].apply(lambda x: self.region_tz_dict[x])

        else:
            print('Dataframe already has tz column!')



    def trainTestSplit(self, test_size=0.3):

        self.addTzCol()
        #This does a TT split, and returns COPIES of the split DF's, so use them locally like that.
        #It's not doing any K-fold cross val stuff, so we should do that later. This is just quick and dirty.
        X_tr, X_test, y_tr, y_test = train_test_split(self.df.drop(['user','subreddit','tz'], axis=1), self.df['tz'], test_size=test_size, random_state=42)

        '''print("shape of X_tr: {}".format(X_tr.shape))
        print("shape of X_test: {}".format(X_test.shape))
        print("shape of y_tr: {}".format(y_tr.shape))
        print("shape of y_test: {}".format(y_test.shape))'''

        return(X_tr, X_test, y_tr, y_test)


    def customCyclicMetric(self, y_pred, y_true):
        #y_true = train_data.get_label()
        diff = abs(y_true - y_pred)
        if type(y_pred).__name__=='Tensor':
            diff[diff>12] = 24 - diff[diff>12]
        else:
            diff[diff>12] = 24 - diff
        #err = np.mean(diff)
        err = sum(diff**2)/len(diff)
        return(err)


    def cyclicMetricSGD(self, show_plot=False, alpha=10**-4, timesteps=10**3):

        import torch
        torch_dtype = torch.float32
        torch.set_default_dtype(torch_dtype)

        X_tr, X_test, y_tr, y_test = self.trainTestSplit()

        #This is the set of linear weights.
        #Not doing any regularization right now, which we might want to do.
        #Also, it technically doesn't have a bias term, I think...should probably add that.
        '''W = torch.zeros((self.N_bins + 1,1), requires_grad=True)

        X_tr_tensor = torch.tensor(np.concatenate((X_tr.values,np.ones((X_tr.values.shape[0],1))),axis=1), requires_grad=False, dtype=torch_dtype)
        X_test_tensor = torch.tensor(np.concatenate((X_test.values,np.ones((X_test.values.shape[0],1))),axis=1), requires_grad=False, dtype=torch_dtype)'''


        W = torch.zeros((self.N_bins,1), requires_grad=True)

        X_tr_tensor = torch.tensor(X_tr.values, requires_grad=False, dtype=torch_dtype)
        X_test_tensor = torch.tensor(X_test.values, requires_grad=False, dtype=torch_dtype)



        y_tr_tensor = torch.tensor(y_tr.values, requires_grad=False, dtype=torch_dtype)
        y_test_tensor = torch.tensor(y_test.values, requires_grad=False, dtype=torch_dtype)

        y_tr_tensor = y_tr_tensor.unsqueeze(dim=1)
        y_test_tensor = y_test_tensor.unsqueeze(dim=1)


        #From some short testing, using alpha from 10^-3 to 10^-6 seems good.
        #alpha = 10**-3
        w_history = np.array(W.detach().numpy())
        print('\n')
        t_range = timesteps
        for t in range(t_range):

            y_pred_tensor = X_tr_tensor.mm(W)
            loss = self.customCyclicMetric(y_pred_tensor, y_tr_tensor)
            loss.backward()

            if t%int(t_range/10)==0:
                print('iteration {}/{}, loss: {:.2f}'.format(t, t_range, loss.item()))

            with torch.no_grad():
                W -= alpha*W.grad
                W.grad.zero_()

            w_history = np.concatenate((w_history,W.detach().numpy()), axis=1)


        print('\n\nfinal W values: {}'.format(W.squeeze().detach().numpy()))

        y_tr_pred_tensor = X_tr_tensor.mm(W)
        y_test_pred_tensor = X_test_tensor.mm(W)
        loss = self.customCyclicMetric(y_test_pred_tensor, y_test_tensor)

        print('\nLoss from test data set: {:.4f}'.format(loss.item()))

        fig, ax = plt.subplots(1, 1, figsize=(16,8))
        #for i in range(5):
        for i in range(w_history.shape[0]):
            plt.plot(w_history[i,:],label='w'+str(i+1))

        plt.legend()
        plt.title('alpha = '+str(alpha))
        plt.xlabel('SGD iterations')
        plt.ylabel('weight values')
        plt.savefig('SGD_weights_converge_alpha{}_{}steps.png'.format(alpha, t_range))
        if show_plot:
            plt.show()

        y_tr_pred = y_tr_pred_tensor.squeeze().detach().numpy()
        y_tr_true = y_tr_tensor.squeeze().detach().numpy()
        y_test_pred = y_test_pred_tensor.squeeze().detach().numpy()
        y_test_true = y_test_tensor.squeeze().detach().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16,8))
        ax_train = axes[0]
        ax_test = axes[1]

        df_tr = pd.DataFrame({'true':y_tr_true, 'pred':y_tr_pred})
        df_test = pd.DataFrame({'true':y_test_true, 'pred':y_test_pred})

        mean_tr = df_tr.groupby(['true']).mean()
        tr_bins = mean_tr.index.values
        mean_tr_pred = mean_tr.values[:,0]

        mean_test = df_test.groupby(['true']).mean()
        test_bins = mean_test.index.values
        mean_test_pred = mean_test.values[:,0]

        ideal = np.arange(-12, 13, 1)
        ax_train.plot(y_tr_true, y_tr_pred, color='tomato', marker='+', linestyle='None')
        ax_train.plot(ideal, ideal, color='lightgray', label='ideal')
        ax_train.plot(tr_bins, mean_tr_pred, color='black', marker='+', markersize=15, linestyle='dashed', label='bin avg')
        ax_train.set_xlabel('true train y values (time zone)')
        ax_train.set_ylabel('pred train y values (time zone)')
        ax_train.set_xlim((-13,13))
        ax_train.legend()

        ax_test.plot(y_test_true, y_test_pred, color='cornflowerblue', marker='+', linestyle='None')
        ax_test.plot(ideal, ideal, color='lightgray', label='ideal')
        ax_test.plot(test_bins, mean_test_pred, color='black', marker='+', markersize=15, linestyle='dashed', label='bin avg')
        ax_test.set_xlabel('true test y values (time zone)')
        ax_test.set_ylabel('pred test y values (time zone)')
        ax_test.set_xlim((-13,13))
        ax_test.legend()

        plt.savefig('SGD_predictions_alpha{}_{}steps.png'.format(alpha, t_range))
        if show_plot:
            plt.show()


    def plotTrainTestResults(self, y_tr_true, y_tr_pred, y_test_true, y_test_pred, show_plot=False, save_plot=False, plot_title=""):

        fig, axes = plt.subplots(1, 2, figsize=(16,8))
        ax_train = axes[0]
        ax_test = axes[1]

        df_tr = pd.DataFrame({'true':y_tr_true, 'pred':y_tr_pred})
        df_test = pd.DataFrame({'true':y_test_true, 'pred':y_test_pred})

        mean_tr = df_tr.groupby(['true']).mean()
        tr_bins = mean_tr.index.values
        mean_tr_pred = mean_tr.values[:,0]

        mean_test = df_test.groupby(['true']).mean()
        test_bins = mean_test.index.values
        mean_test_pred = mean_test.values[:,0]

        ideal = np.arange(-12, 13, 1)
        ax_train.plot(y_tr_true, y_tr_pred, color='tomato', marker='+', linestyle='None')
        ax_train.plot(ideal, ideal, color='lightgray', label='ideal')
        ax_train.plot(tr_bins, mean_tr_pred, color='black', marker='+', markersize=15, linestyle='dashed', label='bin avg')
        ax_train.set_xlabel('true train y values (time zone)')
        ax_train.set_ylabel('pred train y values (time zone)')
        ax_train.set_xlim((-13,13))
        ax_train.legend()

        ax_test.plot(y_test_true, y_test_pred, color='cornflowerblue', marker='+', linestyle='None')
        ax_test.plot(ideal, ideal, color='lightgray', label='ideal')
        ax_test.plot(test_bins, mean_test_pred, color='black', marker='+', markersize=15, linestyle='dashed', label='bin avg')
        ax_test.set_xlabel('true test y values (time zone)')
        ax_test.set_ylabel('pred test y values (time zone)')
        ax_test.set_xlim((-13,13))
        ax_test.legend()

        if save_plot:
            plt.savefig(plot_title + '.png')
        if show_plot:
            plt.show()



    def NN1(self):

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        torch_dtype = torch.float32
        torch.set_default_dtype(torch_dtype)

        X_tr, X_test, y_tr, y_test = self.trainTestSplit()

        X_tr_tensor = torch.tensor(X_tr.values, requires_grad=False, dtype=torch_dtype)
        X_test_tensor = torch.tensor(X_test.values, requires_grad=False, dtype=torch_dtype)

        y_tr_tensor = torch.tensor(y_tr.values, requires_grad=False, dtype=torch_dtype)
        y_test_tensor = torch.tensor(y_test.values, requires_grad=False, dtype=torch_dtype)

        y_tr_tensor = y_tr_tensor.unsqueeze(dim=1)
        y_test_tensor = y_test_tensor.unsqueeze(dim=1)

        class DQN(nn.Module):

            def __init__(self,D_in,H,D_out,NL_fn=torch.tanh,softmax=False):
                super(DQN, self).__init__()

                self.lin1 = nn.Linear(D_in,H)
                self.lin2 = nn.Linear(H,D_out)
                self.NL_fn = NL_fn
                self.softmax = softmax

            def forward(self, x):
                x = self.lin1(x)
                #x = F.relu(x)
                #x = torch.tanh(x)
                x = self.NL_fn(x)
                x = self.lin2(x)
                if self.softmax:
                    x = torch.softmax(x,dim=1)
                return(x)

        N_hidden_layer_nodes = 60
        self.tz_predict_NN = DQN(24, N_hidden_layer_nodes, 1, NL_fn=F.relu)
        self.optimizer = optim.RMSprop(self.tz_predict_NN.parameters())

        loss_history = []
        N_steps = 2000
        for i in range(N_steps):

            y_pred_tensor = self.tz_predict_NN(X_tr_tensor)
            loss = self.customCyclicMetric(y_pred_tensor, y_tr_tensor)
            loss_history.append(loss.item())

            if i%int(N_steps/10)==0:
                print('iteration {}/{}, loss: {:.2f}'.format(i, N_steps, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



        y_tr_pred_tensor = self.tz_predict_NN(X_tr_tensor)
        y_test_pred_tensor = self.tz_predict_NN(X_test_tensor)
        loss = self.customCyclicMetric(y_test_pred_tensor, y_test_tensor)

        print('\nLoss from test data set: {:.4f}'.format(loss.item()))
        plt.plot(loss_history)
        plt.title('NN_loss_{}steps_{}HLN_{}'.format(N_steps, N_hidden_layer_nodes, fst.getDateString()))
        plt.xlabel('NN SGD iterations')
        plt.ylabel('loss')
        #plt.savefig('NN_loss_{}steps_{}HLN_{}.png'.format(N_steps, N_hidden_layer_nodes, fst.getDateString()))
        plt.show()

        y_tr_pred = y_tr_pred_tensor.squeeze().detach().numpy()
        y_tr_true = y_tr_tensor.squeeze().detach().numpy()
        y_test_pred = y_test_pred_tensor.squeeze().detach().numpy()
        y_test_true = y_test_tensor.squeeze().detach().numpy()

        self.plotTrainTestResults(y_tr_true, y_tr_pred, y_test_true, y_test_pred, show_plot=True, save_plot=True,
        plot_title='NN_traintest_{}steps_{}HLN_{}'.format(N_steps, N_hidden_layer_nodes, fst.getDateString()))


    def simpleLinReg(self):

        #This is garbage right now.
        X_tr, X_test, y_tr, y_test = self.trainTestSplit()

        lr = LinearRegression()
        lr.fit(X_tr, y_tr)

        print("\n\nLR train score: {}".format(lr.score(X_tr,y_tr)))
        print("LR test score: {}".format(lr.score(X_test,y_test)))



    def addExtraDataSets(self, dirs):
        #You can either pass this a list or a single dir.
        #This is for adding data in other dirs to self.df. You just pass it the
        #dir like you did in init(). It should already have an aggregate file. It's
        #not going to try and create one here. If it doesn't, run createAggFile()
        #on that dir separately. Also, don't run addTzCol() or any functions that call
        #it until you've imported all the data you want to, otherwise only part of the
        #dataset will have the Tz column, but it won't get added again.

        if type(dirs).__name__=='list':
            dir_list = dirs
        else:
            dir_list = [dirs]

        for dir in dir_list:

            file_list = glob.glob(fst.addTrailingSlashIfNeeded(dir) + 'aggregate' + '*')
            if file_list!=1:
                print('Not the right amount of aggregate files in dir: ' + str(len(file_list)) + '. Creating one now.')
                csv_file = self.createAggFile(dir)
            else:
                csv_file = file_list[0]

            extra_df = pd.read_csv(csv_file, index_col=0)
            self.df = self.df.append(extra_df, ignore_index=True)


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
            ax.plot((df_subreddit_sum/df_subreddit_sum.sum()).values, label=(sub + ' ({})'.format(self.region_tz_dict[sub])))

        ax.legend()
        ax.set_xlim(0, self.N_bins)
        ax.set_xlabel('Hour (24H)')
        ax.set_ylabel('Post frequency')
        plt.savefig('savefigs/{}_posttimes_{}.png'.format('_'.join(unique_subs),fst.getDateString()))
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

    def run_lightgbm(self,x_train,x_val,y_train,y_val):

        #x_train,x_val,y_train,y_val = self.trainTestSplit()
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
        return model































#
