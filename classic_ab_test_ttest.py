"""
Example A/B test for click through rate.
The test used is the ttest. Three different ttest are performed. The normal
ttest from scipy, the Welche's ttest from scipy and a manual Welche's test.
The data are not normally distributed and therefore the ttest assumpions is not
fulfilled.
"""


import numpy as np
import pandas as pd
from scipy import stats

# get the data
data = pd.read_csv("advertisement_clicks.csv")

a = data[data["advertisement_id"] == "A"]["action"].values
b = data[data["advertisement_id"] == "B"]["action"].values
N_a = a.size
N_b = b.size

# Manual Welch's test
# calculating variance for a and b
var_a = a.var()
var_b = b.var()

# calculate pooled std
s = np.sqrt(var_a / N_a + var_b / N_b)
# t statictics
t = (a.mean() - b.mean()) / s

# degrees of freedom
nu_a = N_a - 1
nu_b = N_b - 1
df = (
    (var_a / N_a + var_b / N_b)**2 /
    ((var_a**2) / (N_a**2 * nu_a) + (var_b**2) / (N_b**2 * nu_b))
)

# calculate p-value
p = (1 - stats.t.cdf(np.abs(t), df=df)) * 2
print(f"Manual Welche's test\nt: {t} p: {p}")

t2, p2 = stats.ttest_ind(a, b)
print(f"Built in scipy ttest\nt2: {t2} p2: {p2}")

# welch's t-test:
t, p = stats.ttest_ind(a, b, equal_var=False)
print(f"Built in scipy Welch's t-test\nt: {t} p: {p}")
