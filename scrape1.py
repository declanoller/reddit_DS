import pandas as pd

from DataCollection import DataCollection
from ML import ML

'''

data found here: http://files.pushshift.io/reddit/

API docs here: https://github.com/pushshift/api

don't request more than ~1/sec. Each request can return <500 terms.

getting 1000 posts each for 100 users each from 4 subreddits (i.e., 800 requests)
took 48 minutes, so ~4s/request. Hmmm.

'''

ml = ML('/home/declan/Documents/code/reddit_DS/savedat/nyc_losangeles_unitedkingdom_greece_indonesia_japan_newzealand_1000users_24bins_00-16-17')
#ml = ML('/home/declan/Documents/code/reddit_DS/savedat/rizki_dat1/')
ml.addExtraDataSets('/home/declan/Documents/code/reddit_DS/savedat/rizki_dat1/')
#ml.prettyPrintDB(ml.df)
#ml.postAvgTimesByRegion()
#ml.simpleLinReg()
ml.cyclicMetricSGD(alpha=10**-4, timesteps=4000, show_plot=False)
exit(0)

ml.cyclicMetricSGD(alpha=10**-3, timesteps=4000, show_plot=False)
ml.cyclicMetricSGD(alpha=10**-3, timesteps=40000, show_plot=False)
ml.cyclicMetricSGD(alpha=10**-5, timesteps=40000, show_plot=False)


dc = DataCollection(verbose=1, N_users=1000, N_post_limit=500)
#dc.predictRunTime(['nyc','losangeles','unitedkingdom','greece','indonesia','japan','newzealand'])
dc.postTimesRegions(['nyc','losangeles','unitedkingdom','greece','indonesia','japan','newzealand'])
exit(0)







#
