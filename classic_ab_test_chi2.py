"""
Example A/B test for click through rate.
The test used is the chi2. Two different chi2 tests are performed. The normal
chi2 from scipy and a manual chi2 test.

contingency table
       click       no click
-----------------------------
ad A |   a            b
ad B |   c            d

chi^2 = (ad - bc)^2 (a + b + c + d) / [ (a + b)(c + d)(a + c)(b + d)]
degrees of freedom = (#cols - 1) x (#rows - 1) = (2 - 1)(2 - 1) = 1

"""


import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

# get the data
data = pd.read_csv("advertisement_clicks.csv")

# Build the contingency table
T = pd.crosstab(data["advertisement_id"], data["action"]).values

# Manual chi2 test
c2 = (
    ((np.linalg.det(T)**2) * np.sum(T)) /
    (np.sum(T[0]) * np.sum(T[1]) * np.sum(T[:,0]) * np.sum(T[:,1]))
)
p_value = 1 - chi2.cdf(x=c2, df=1)
print(f"MANUAL CHI2 TEST\nchi2: {c2}, p-value: {p_value}")

# scipy built in chi2 test
c2, p_value, dof, expected = chi2_contingency(T, correction=False)
print(f"SCIPY BUILT IN CHI2 TEST\nchi2: {c2}, p-value: {p_value}")
