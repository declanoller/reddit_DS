import pandas as pd

from DataCollection import DataCollection

'''

data found here: http://files.pushshift.io/reddit/

API docs here: https://github.com/pushshift/api

don't request more than ~1/sec. Each request can return <500 terms.

getting 1000 posts each for 100 users each from 4 subreddits (i.e., 800 requests)
took 48 minutes, so ~4s/request. Hmmm.

'''



dc = DataCollection(verbose=1, N_users=5, N_post_limit=500)

#dc.meanPostTimesRegions(['nyc','losangeles'])
#dc.postTimesRegions(['boston','losangeles','london','paris','greece','saudiarabia','pakistan','vietnam','taiwan','korea','melbourne','newzealand'])
#dc.postTimesRegions(['losangeles','boston','unitedkingdom','india','japan','newzealand'])
dc.postTimesRegions(['losangeles', 'boston'])
#dc.getUserPostTimesForSub('providence')
#dc.plotUserPostTimes('m1327')


#
