import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
from statistics import mean
import requests
from time import time, sleep
import FileSystemTools as fst
from collections import Counter

class DBTools:


    def __init__(self):

        #The guy requests that people don't poll more than once a second, so we'll limit it like this.
        self.api_request_limit = 1.0
        self.last_request_time = time()
        self.N_bins = 24
        self.max_N_users = -1 # set to -1 to revert to default (in getUserPostTimesForSub).
        self.N_post_limit = -1 # set to -1 to revert to default (in getUserPostTimes).
        self.verbose = 0 
        self.max_req_size = 500


        city_tz_dict = {
        'anchorange' : -9,
        'losangeles' : -8,
        'vancouver' : -8,
        'denver' : -7,
        'dallas' : -6,
        'chicago' : -6,
        'mexicocity' : -6,
        'nyc' : -5,
        'boston' : -5,
        'washingtondc' : -5,
        'puertorico' : -4,
        'buenosaires' : -3,
        'london' : 0,
        'ireland' : 0,
        'paris' : 1,
        'madrid' : 1,
        'rome' : 1,
        'greece' : 2,
        'romania' : 2,
        'saudiarabia' : 3,
        'pakistan' : 5,
        'india' : 5.5,
        'thailand' : 7,
        'vietnam' : 7,
        'indonesia' : 7,
        'beijing' : 8,
        'shanghai' : 8,
        'perth' : 8.75,
        'taiwan' : 8,
        'japan' : 9,
        'korea' : 9,
        'sydney' : 10,
        'melbourne' : 10,
        'newzealand' : 12
        }



    def requestToJson(self,request_str):

        #This will make it not poll too often. It will always go through, but
        #will wait a second if it has been too soon.
        if(self.verbose):print('requesting',request_str)
        while True:
            if time() - self.last_request_time >= self.api_request_limit:
                r = requests.get(request_str)
                self.last_request_time = time()
                break
            else:
                sleep(self.api_request_limit/5)

        return(r)


    def normalizeJson(self,json_obj,field='data',aggs=False):

        if aggs:
            df = pd.io.json.json_normalize(json_obj.json()['aggs'],[field])
        else:
            df = pd.io.json.json_normalize(json_obj.json()[field])
        return(df)



    def prettyPrintDB(self,df):

        print(tabulate(df.head(), headers=df.columns.values, tablefmt='psql'))
        print('\n')
        print('columns: ', df.columns)


    def getUsersInSub(self,subreddit):

        request = 'https://api.pushshift.io/reddit/search/comment/?subreddit={}&aggs=author&size={}'.format(subreddit,self.max_req_size)
        J = self.requestToJson(request)
        df = self.normalizeJson(J,field='author',aggs=True)

        user_list = df['key'].values.tolist()


        if '[deleted]' in user_list:
            user_list.remove('[deleted]')
        if 'AutoModerator' in user_list:
            user_list.remove('AutoModerator')

        if(self.verbose):print('\nFound',len(user_list),'users in subreddit [',subreddit,']:',user_list,'\n')

        return(user_list)



    def getUserStartEndDates(self,user):
        if(self.verbose): print('Collecting start end dates for user',user)

        #For a user, get their beginning and ending dates of posting.
        oldest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&fields=created_utc&size=5&sort=asc'.format(user)
        oldest_df = self.normalizeJson(self.requestToJson(oldest_req))
        oldest_post = oldest_df['created_utc'].values.tolist()[0]

        newest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&fields=created_utc&size=5&sort=desc'.format(user)
        newest_df = self.normalizeJson(self.requestToJson(newest_req))
        newest_post = newest_df['created_utc'].values.tolist()[0]

        return((oldest_post,newest_post))



    def getUserPostTimes(self,user):

        #All times at this point are in units of epoch, i.e., a long integer string.
        start_time, end_time = self.getUserStartEndDates(user)

        #You can only request up to 500 items at once, so we have to loop through until there's nothing left.
        after_time = start_time
        post_times = []
        i = 0
        N_post_limit = 1000 
        if(self.N_post_limit != -1): N_post_limit = self.N_post_limit
        while True:
            range_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&after={}&fields=created_utc&size={}&sort=asc'.format(user,after_time,self.max_req_size)
            range_df = self.normalizeJson(self.requestToJson(range_req))
            if len(range_df) == 0 or len(post_times)>=N_post_limit:
                break
            else:
                range_list = range_df['created_utc'].values.tolist()
                post_times += range_list
                after_time = range_list[-1]
                i += 1

        #print('took {} iterations, {} total posts'.format(i,len(post_times)))
        return(post_times)



    def getUserPostTimesForSub(self,subreddit):

        N_users = 300 #default
        if(self.max_N_users != -1 ): N_users = self.max_N_users 

        users = self.getUsersInSub(subreddit)[:N_users]
        if(self.verbose):print('\ngetUserPostTimesForSub : limiting to',len(users),'users:',users,'\n')

        user_post_times_condensed = []
        user_post_times_all = []
        fig, ax = plt.subplots(1,1,figsize=(8,8))

        for i,user in enumerate(users):
            if(self.verbose):print('\nGetting info for user: {} ({} out of {})'.format(user,i+1,len(users)))            
            post_times = self.getUserPostTimes(user)
            pt_binned = self.binPostTimes(post_times)
            #most_common_bin, _ = Counter(pt_binned).most_common(1)[0]
            max_bin = np.argmax(pt_binned)
            user_post_times_all.append(pt_binned)
            user_post_times_condensed.append(max_bin)

            sns.distplot(pt_binned, bins=self.N_bins, kde=True, rug=False, hist=False)
            ax.vlines(max_bin,0,.05,linestyles='dashed')



        ax.set_xlim(0,24)
        ax.set_title('Post-time maxs dist. for subreddit: ' + subreddit)
        ax.set_xlabel('Hour (24H)')
        ax.set_ylabel('Post frequency')
        plt.savefig('savefigs/{}_posttimes_{}.png'.format(subreddit,fst.getDateString()))


        fname = 'savedat/{}users_{}_subreddit_{}bins_{}.txt'.format(N_users, subreddit, self.N_bins, fst.getDateString())
        np.savetxt(fname, np.array(user_post_times_all), fmt='%d')

        #plt.show()
        return(user_post_times_condensed)



    def postTimesRegions(self,region_list):

        if(self.verbose):print('\nBegin processing postTimesRegion for:',region_list,'\n')

        start_time = fst.getCurTimeObj()
        region_stats = pd.DataFrame({'region':[],'post_time':[]})

        for subreddit in region_list:
            print('\n\nGetting stats for city {} ({} out of {})\n'.format(subreddit,region_list.index(subreddit)+1,len(region_list)))
            '''stats = self.getUserPostTimesForSub(subreddit)
            sub_stats = pd.DataFrame({'region':[subreddit]*len(stats),'post_time':stats})
            region_stats = region_stats.append(sub_stats, ignore_index=True)'''
            try:
                stats = self.getUserPostTimesForSub(subreddit)
                sub_stats = pd.DataFrame({'region':[subreddit]*len(stats),'post_time':stats})
                region_stats = region_stats.append(sub_stats, ignore_index=True)
            except Exception as e:
                print('problem getting stats for city', subreddit)
                print('exception:',e)

            print('\n\ntook this long for {}: {}'.format(subreddit,fst.getTimeDiffStr(start_time)))

        #self.prettyPrintDB(region_stats)
        print('\n\ntook this long to run: ' + fst.getTimeDiffStr(start_time))

        fig, ax = plt.subplots(1,1,figsize=(16,12))
        sns.violinplot(x='region', y='post_time', data=region_stats)
        ax.set_title('Post-time maxs dist. for subreddits: ' + ', '.join(region_list))
        ax.set_xlabel('city')
        ax.set_ylabel('post_time (24H)')
        plt.savefig('savefigs/{}_posttimes_{}.png'.format('_'.join(region_list),fst.getDateString()))


    def binPostTimes(self,pt):
        #This bins times in the epoch format to a list of times of whatever bin you choose. Right now it's by hour.
        pt_hours = np.array([datetime.strftime(datetime.utcfromtimestamp(ts),'%H') for ts in pt]).astype('int')
        bins = np.histogram(pt_hours, bins=list(range(self.N_bins+1)))[0]
        return(bins)


    def plotUserPostTimes(self,user):

        pt = self.getUserPostTimes(user)


        pt_hours = self.binPostTimes(pt)

        dp = sns.distplot(pt_hours, bins=24, kde=True, rug=False, hist=False);
        dp.axes.set_xlim(0,24)
        dp.axes.set_title('Post-time dist. for user ' + user)
        dp.axes.set_xlabel('Hour (24H)')
        dp.axes.set_ylabel('Post frequency')
        dp.axes.vlines(max(pt_hours),0,1,linestyles='dashed')
        plt.savefig('savefigs/{}_posttimes.png'.format(user))
        plt.show()
#
