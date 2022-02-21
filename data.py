import pandas as pd
import pylab
# import scipy
from scipy import stats, optimize, interpolate

my_data_sem = pd.Series([
    1.270882,
    1.541523,
    1.553915,
    1.255453,
    1.456378,
    1.763657,
    1.167109,
    1.467461,
    4.556010,
    2.924524
])

my_data = pd.Series([
    5.188256,
    4.186968,
    3.238907,
    4.614054,
    1.681659,
    1.822847,
    3.209124,
    3.871628,
    5.168882,
    1.372640
])

# res = stats.normaltest(my_data_sem, axis=0, nan_policy='propagate')
# ORB = stats.skewtest(my_data, axis=0, nan_policy='propagate')
# ORB_sem = stats.skewtest(my_data_sem, axis=0, nan_policy='propagate')
ORB = stats.normaltest(my_data, axis=0, nan_policy='propagate')
ORB_sem = stats.normaltest(my_data_sem, axis=0, nan_policy='propagate')

# res = stats.kstest(my_data_sem, 'norm')
print("ORB: ",ORB)
print("ORB_sem: ",ORB_sem)
testT = stats.ttest_ind(my_data,my_data_sem)
print("testT: ",testT)
# data = pd.concat([my_data, my_data_sem], axis=1)
kruskal = stats.kruskal(my_data_sem, my_data)
print("kruskal: ", kruskal)
# my_data.hist()
# my_data.plot()
# data.plot(kind = 'box')

# pylab.show()
