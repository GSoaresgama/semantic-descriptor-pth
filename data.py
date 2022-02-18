import pandas as pd
import pylab

my_data_sem = pd.Series([1.270882,
1.541523,
1.553915,
1.255453,
1.456378,
1.763657,
1.167109,
1.467461,
4.556010,
2.924524])


my_data = pd.Series([5.188256,
4.186968,
3.238907,
4.614054,
1.681659,
1.822847,
3.209124,
3.871628,
5.168882,
1.372640])

data = pd.concat([my_data, my_data_sem], axis=1)

# my_data.hist()
# my_data.plot()
data.plot(kind = 'box')

pylab.show()
