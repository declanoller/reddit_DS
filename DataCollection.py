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
import os

class DataCollection:


    def __init__(self,**kwargs):

        #The guy requests that people don't poll more than once a second, so we'll limit it like this.
        self.api_request_limit = 1.0
        self.last_request_time = time()
        self.N_bins = 24
        self.N_users = kwargs.get('N_users',300)
        self.N_post_limit = kwargs.get('N_post_limit',500)
        self.verbose = kwargs.get('verbose',0)

        self.save_dat_dir = 'savedat'

        self.base_fname = '{}users_{}bins_{}'.format(self.N_users, self.N_bins, fst.getDateString())
        self.ext = '.csv'

        #number of users in comment following time zone
        city_tz_dict = {
        'anchorange' : -9,
        'losangeles' : -8, #124k
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

        request = 'https://api.pushshift.io/reddit/search/comment/?subreddit={}&aggs=author&size=500'.format(subreddit)
        J = self.requestToJson(request)
        df = self.normalizeJson(J,field='author',aggs=True)

        user_list = df['key'].values.tolist()


        if '[deleted]' in user_list:
            user_list.remove('[deleted]')
        if 'AutoModerator' in user_list:
            user_list.remove('AutoModerator')

        if(self.verbose):print('\nFound',len(user_list),'users in subreddit [',subreddit,']:',user_list,'\n')

        return(user_list)


    def getUserStartEndDates(self, user, subreddit=None):
        if(self.verbose): print('Collecting start end dates for user',user)

        #For a user, get their beginning and ending dates of posting. I've combined this one and the one for
        #subreddit-specific, now you just pass it the subreddit if you want that, and nothing otherwise.
        if subreddit == None:
            oldest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&fields=created_utc&size=5&sort=asc'.format(user)
        else:
            oldest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&subreddit={}&fields=created_utc&size=5&sort=asc'.format(user,subreddit)

        if subreddit == None:
            newest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&fields=created_utc&size=5&sort=desc'.format(user)
        else:
            newest_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&subreddit={}&fields=created_utc&size=5&sort=desc'.format(user,subreddit)

        oldest_df = self.normalizeJson(self.requestToJson(oldest_req))
        oldest_post = oldest_df['created_utc'].values.tolist()[0]

        newest_df = self.normalizeJson(self.requestToJson(newest_req))
        newest_post = newest_df['created_utc'].values.tolist()[0]

        return((oldest_post,newest_post))


    def getUserPostTimes(self,user,subreddit):

        #All times at this point are in units of epoch, i.e., a long integer string.
        #start_time, end_time = self.getUserStartEndDates(user) #maybe depracated.
        start_time, end_time = self.getUserStartEndDates(user, subreddit=subreddit)

        #You can only request up to 500 items at once, so we have to loop through until there's nothing left.
        after_time = start_time
        post_times = []
        i = 0

        while True:
            range_req = 'https://api.pushshift.io/reddit/search/comment/?author={}&after={}&before={}&fields=created_utc&size=500&sort=asc'.format(user,after_time,end_time)
            range_df = self.normalizeJson(self.requestToJson(range_req))
            if len(range_df) == 0 or len(post_times)>=self.N_post_limit:
                break
            else:
                range_list = range_df['created_utc'].values.tolist()
                post_times += range_list
                after_time = range_list[-1]
                i += 1

        return(post_times)



    def getDataFrameDict(self, user=None, bin_vals=None, subreddit=None):

        #Creates a list of the column names for each dataframe, changes with N_bins
        columns = ['user'] + ['bin'+str(i) for i in range(self.N_bins)] + ['subreddit']

        if user == None and bin_vals == None and subreddit == None:
            # +2 for the user and subreddit columns.
            col_vals = [[]]*(self.N_bins + 2)

        else:
            col_vals = [[user]] + [[bv] for bv in bin_vals] + [[subreddit]]

        return(dict(zip(columns, col_vals)))



    def getUserPostTimesForSub(self,subreddit):

        users = self.getUsersInSub(subreddit)[:self.N_users]
        if(self.verbose):print('\ngetUserPostTimesForSub : limiting to',len(users),'users:',users,'\n')

        user_sub_stats = pd.DataFrame(self.getDataFrameDict())

        for i,user in enumerate(users):
            try:
                if(self.verbose):print('\nGetting info for user: {} ({} out of {})'.format(user,i+1,len(users)))
                post_times = self.getUserPostTimes(user,subreddit)
                pt_binned = self.binPostTimes(post_times)

                #Appends the dataframe for this user to the one for the sub.
                user_stats = pd.DataFrame(self.getDataFrameDict(user=user, bin_vals=pt_binned, subreddit=subreddit))
                user_sub_stats = user_sub_stats.append(user_stats, ignore_index=True)

            except Exception as e:
                print('problem at iteration {}: getting info for user: {}'.format(i, user))


        self.saveDataFrame(user_sub_stats, label=subreddit)
        return(user_sub_stats)



    def saveDataFrame(self, df, label=''):
        fname = label + '_' + self.base_fname + self.ext
        df.to_csv(fst.combineDirAndFile(self.run_dir, fname))


    def postTimesRegions(self,region_list):

        if(self.verbose):print('\nBegin processing postTimesRegion for:',region_list,'\n')

        self.run_dir = fst.combineDirAndFile(self.save_dat_dir, '_'.join(region_list) + '_' + self.base_fname)

        if not os.path.isdir(self.run_dir):
            print('Creating run dir: ',self.run_dir)
            os.mkdir(self.run_dir)
        else:
            print('Problem creating run dir: ',self.run_dir)
            exit(0)

        start_time = fst.getCurTimeObj()
        region_stats = pd.DataFrame(self.getDataFrameDict())

        print('\n\nEstimated runtime for {} cities, {} users each, {} posts each: {} seconds'.format(len(region_list), self.N_users, self.N_post_limit,
        len(region_list)*self.N_users*(2 + int(self.N_post_limit/500))))

        for subreddit in region_list:
            print('\n\nGetting stats for city {} ({} out of {})\n'.format(subreddit,region_list.index(subreddit)+1,len(region_list)))

            try:
                sub_stats = self.getUserPostTimesForSub(subreddit)
                region_stats = region_stats.append(sub_stats, ignore_index=True)

            except Exception as e:
                print('problem getting stats for city', subreddit)
                print('exception:',e)

            print('\n\ntook this long for {}: {}'.format(subreddit,fst.getTimeDiffStr(start_time)))

        #self.prettyPrintDB(region_stats)
        print('\n\ntook this long to run: ' + fst.getTimeDiffStr(start_time))
        self.prettyPrintDB(region_stats)

        self.saveDataFrame(region_stats, label='all')



    def postTimesToHours(self,pt):
        pt_hours = np.array([datetime.strftime(datetime.utcfromtimestamp(ts),'%H') for ts in pt]).astype('int')
        return(pt_hours)



    def binPostTimes(self,pt):
        #This bins times in the epoch format to a list of times of whatever bin you choose. Right now it's by hour.
        #the bins parameter defines the "boundaries" of the bins, so if you want to do by half hour, you'll have to
        #give it a list of half integers (and also change postTimesToHours probably).
        bins = np.histogram(self.postTimesToHours(pt), bins=list(range(self.N_bins+1)))[0]
        return(bins)


#
