import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
from statistics import mean

class DBTools:


    def __init__(self,fname):

        self.fname = fname
        self.df = pd.read_json(self.fname,lines=True)



    def getDBInfo(self):

        print('\n\n\n')
        self.prettyPrintDB(self.df)

        print('\n')
        print('columns: ', self.df.columns)
        print('# posts: ', self.df.size)

        print('# unique subreddits: ',self.df['subreddit'].unique().size)
        print('# unique authors: ',self.df['author'].unique().size)

        print('\n\n\n')


    def prettyPrintDB(self,df):

        print(tabulate(df.head(), headers=df.columns.values, tablefmt='psql'))



    def dropUnimportant(self):

        '''
        Default columns:
            ['author', 'author_flair_css_class', 'author_flair_text', 'body',
           'controversiality', 'created_utc', 'distinguished', 'edited', 'gilded',
           'id', 'link_id', 'parent_id', 'retrieved_on', 'score', 'stickied',
           'subreddit', 'subreddit_id', 'ups']
        '''

        drop_labels = ['author_flair_css_class', 'author_flair_text', 'body',
       'controversiality', 'distinguished', 'edited', 'gilded',
        'parent_id', 'retrieved_on', 'score', 'stickied',
       'ups']

       #Probably actually should just keep certain ones... not choose which to drop

        for label in drop_labels:
            if label in self.df.columns:
                self.df.drop(labels=label, inplace=True, axis='columns')



    def getPostsFromSub(self,subreddit):
        return(self.df[self.df['subreddit'] == subreddit])


    def getUsersInSub(self,subreddit):

        sub_df = self.getPostsFromSub(subreddit)

        user_list = sub_df['author'].unique()

        return(user_list)




    def getUserInfo(self,user):

        user_df = self.df[self.df['author'] == user]

        print('\n\n')
        print('user:', user)
        print('# posts:',user_df.size)

        return(user_df)




    def getUserPostTimes(self,user):

        user_df = self.getUserInfo(user)

        timestamps = user_df['created_utc'].values

        timestamps = np.array([datetime.strftime(datetime.utcfromtimestamp(ts),'%H') for ts in timestamps]).astype('int')

        return(timestamps)
        '''dp = sns.distplot(timestamps, bins=24, kde=True, rug=True);
        dp.axes.set_xlim(0,24)
        dp.axes.set_title('Post-time dist. for user ' + user)
        dp.axes.set_xlabel('Hour (24H)')
        dp.axes.set_ylabel('Post frequency')
        plt.show()'''





    def getUserPostTimesForSub(self,subreddit):

        N_users = 20

        sub_df = self.getPostsFromSub(subreddit)
        print('size of sub:',len(sub_df))
        #Get rid of deleted accounts
        sub_df = sub_df[sub_df['author'] != '[deleted]']

        #Rank users of the sub by # posts, take top N
        freq_ranked_users = sub_df.groupby(by='author')[['author']].count().sort_values(by='author',ascending=False)[:N_users]

        freq_ranked_users = freq_ranked_users.index.values
        print(freq_ranked_users)
        fig,ax = plt.subplots(1,1,figsize=(6,6))

        all_post_time_means = []

        for user in freq_ranked_users:

            post_times = self.getUserPostTimes(user)

            all_post_time_means.append(post_times.mean())

            '''ax.clear()
            dp = sns.distplot(post_times, bins=24, kde=True, rug=True);
            dp.axes.set_xlim(0,24)
            dp.axes.set_title('Post-time dist. for user ' + user)
            dp.axes.set_xlabel('Hour (24H)')
            dp.axes.set_ylabel('Post frequency')
            dp.axes.vlines(post_times.mean(),0,1,linestyles='dashed')
            plt.savefig('savefigs/' + user + '.png')'''


        ax.clear()
        dp = sns.distplot(all_post_time_means, bins=24, kde=True, rug=True);
        dp.axes.set_xlim(0,24)
        dp.axes.set_title('Post-time means dist. for subreddit ' + subreddit)
        dp.axes.set_xlabel('Hour (24H)')
        dp.axes.set_ylabel('number of users')
        dp.axes.vlines(mean(all_post_time_means),0,1,linestyles='dashed')
        plt.savefig('savefigs/' + subreddit + '.png')






#
