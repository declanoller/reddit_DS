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

# dc = DataCollection(verbose=1, N_users=10, N_post_limit=50)
# dc.postTimesRegions(['losangeles','london','bali'])

# Ran on Oct 14,2018
# dc = DataCollection(verbose=1, N_users=1000, N_post_limit=500)
# dc.postTimesRegions(['sanfrancisco','geneva','sydney'])

# Ran on Oct 16, 2018
# dc = DataCollection(verbose=1, N_users=1000, N_post_limit=1000)
# dc.postTimesRegions(['chicago','moscow','india']) # this got cut at moscow

# Ran on Oct 17, 2018
# dc = DataCollection(verbose=1, N_users=1000, N_post_limit=1000)
# dc.postTimesRegions(['moscow','india'])

# Ran on Oct 19, 2018, # this got cut at brazil at 94/98
# dc = DataCollection(verbose=1, N_users=1000, N_post_limit=1000)
# dc.postTimesRegions(['denver','brazil','turkey'])


# Ran on Oct 19, 2018, this took ~20 mins.
dc = DataCollection(verbose=1, N_users=1000, N_post_limit=1000)
dc.postTimesRegions(['brazil','turkey'])

