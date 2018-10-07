import pandas as pd

from DBTools import DBTools

'''

other ideas:

-look at S.A./etc for massively upvoted/downvoted comments

-data found here: http://files.pushshift.io/reddit/

example API request here:

url = 'https://api.pushshift.io/reddit/search/comment/?subreddit=dallas&size=500'

r = requests.get(url)
df = pd.io.json.json_normalize(r.json()['data'])


uhhhhh yeah so this one month json file (the 2010 one) is 2.5GB, and trying to
load it makes it shit the bed...how can you read it in chunks?

'''



dbt = DBTools('RC_2008-04')


dbt.dropUnimportant()

dbt.getDBInfo()

dbt.getUserPostTimesForSub('science')

exit(0)

test_user = 'cartooncorpse'
dbt.getUserPostTimes(test_user)



#
