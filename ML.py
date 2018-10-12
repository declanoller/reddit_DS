




class ML:

    def __init__(self, data_dir):

        pass




    def preprocessin():





    def userPostTimesSub():

            fig, ax = plt.subplots(1,1,figsize=(8,8))

                                sns.distplot(self.postTimesToHours(post_times), bins=self.N_bins, kde=True, rug=False, hist=False)
                                #Should make these lines taller, but the math is annoying...
                                ax.vlines(max_bin,0,.1,linestyles='dashed')

                ax.set_xlim(0,24)
                ax.set_title('Post-time maxs dist. for subreddit: ' + subreddit)
                ax.set_xlabel('Hour (24H)')
                ax.set_ylabel('Post frequency')
                plt.savefig('savefigs/{}_posttimes_{}.png'.format(subreddit,fst.getDateString()))


    def postTimesRegion():

                fig, ax = plt.subplots(1,1,figsize=(16,12))
                sns.violinplot(x='region', y='post_time', data=region_stats)
                ax.set_title('Post-time maxs dist. for subreddits: ' + ', '.join(region_list))
                ax.set_xlabel('city')
                ax.set_ylabel('post_time (24H)')
                plt.savefig('savefigs/{}_posttimes_{}.png'.format('_'.join(region_list),fst.getDateString()))



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
