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

# ml = ML('/Users/rizki/Documents/Projects/withDeclan/reddit_DS/savedat/losangeles_boston_5users_24bins_18-46-47')

# ml.postAvgTimesByRegion()
#ml.plotUserPostTimes('405freeway')
#ml.plotUserPostTimesSub('boston')
#ml.plotUserPostTimes(['405freeway', 'cpxh'])

#ml.simpleLinReg()

# exit(0)

dc = DataCollection(verbose=1, N_users=10, N_post_limit=50)
# dc.predictRunTime(['nyc','losangeles','unitedkingdom','france','greece','saudiarabia','pakistan','indonesia','taiwan','japan','melbourne','newzealand'])


#dc.meanPostTimesRegions(['nyc','losangeles'])
#dc.postTimesRegions(['boston','losangeles','london','paris','greece','saudiarabia','pakistan','vietnam','taiwan','korea','melbourne','newzealand'])
dc.postTimesRegions(['losangeles','london','bali'])
#dc.getUserPostTimesForSub('providence')
#dc.plotUserPostTimes('m1327')

