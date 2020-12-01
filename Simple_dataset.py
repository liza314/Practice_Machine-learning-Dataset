import numpy as np
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant
from matplotlib import pyplot as plt
confirmed = np.array([
        45,
        62,
        121,
        198,
        291,
        440,
        571,
        830,
        1287,
        1975,
        2744,
        4515,
        5974,
        7711,
        9692,
        11791,
        14380,
        17205,
        20438,
        24324,
        28018,
        31161,
        34546,
        37198,
        40171,
        48315,
        55220,
        58761,
        63851,
        66492,
        68500,
        70548,
        72436,
        74185,
        75002,
        75891,
        76228
    ])
    
x = np.arange(len(confirmed))
x = add_constant(x)
x

model = OLS(np.log(confirmed[:14]),x[:14])
result = model.fit()
result.summary()

plt.plot(
    np.exp(result.predict(x[:14])),
    label = "Fitted exp. function"
)
plt.plot(confirmed[:14],".",label="Reported cases,CN")
plt.legend()
plt.show()

world_population = 7763252653
days = 0
infected = confirmed[14]
while infected < world_population:
    days += 1
    infected = np.exp(result.predict([1, 13 + days]))[0]
print(f"Number of days until whole world is infected: {days}")

plt.plot(np.exp(result.predict(x[:16])))
plt.plot(confirmed[:16], ".")
plt.show()

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
logistic_function = lambda x, a, b, c, d: \
    a / (1 + np.exp(-c * (x - d))) + b
    
 confirmed = np.array(confirmed)
x = x[:, 1]

(a_, b_, c_, d_), _ = curve_fit(logistic_function, x, confirmed)

def plot_logistic_fit(confirmed, logistic_params):
    a_, b_, c_, d_ = logistic_params
    x = np.arange(0, len(confirmed))
    plt.plot(x, confirmed, ".", label="Reported cases")
    confirmed_pred = logistic_function(x, a_, b_, c_, d_)
    plt.plot(x, confirmed_pred, label="Fitted logistic function")
    plt.legend()
    plt.show()
    return confirmed_pred
confirmed_pred = plot_logistic_fit(confirmed, (a_, b_, c_, d_))

r2_score(confirmed, confirmed_pred)

def plateau(confirmed, logistic_params, diff=10):
    a_, b_, c_, d_ = logistic_params 
    confirmed_now = confirmed[-1]
    confirmed_then = confirmed[-2]
    days = 0
    now = x[-1]
    while confirmed_now - confirmed_then > diff:
        days += 1
        confirmed_then = confirmed_now
        confirmed_now = logistic_function(
            now + days,
            a_,
            b_,
            c_,
            d_,
        )
    return days, confirmed_now
days, confirmed_now = plateau(confirmed, (a_, b_, c_, d_))
print(f"In {days} days the number of infected people will plateau at {int(confirmed_now)}")

x_ = np.linspace(0, x[-1] + days)
plt.plot(
    x_,
    logistic_function(x_, a_, b_, c_, d_)
)
plt.show()

cases_italy = np.array([
    20,
    79,
    150,
    229,
    322,
    400,
    650,
    888,
    1128,
    1694,
    2036,
    2502,
    3089,
    3858,
    4636,
    5883,
    7375,
    9172,
    10149,
    12462,
    15113,
    17660,
    21157,
    24747,
    27980,
    31506,
    35713,
    41035,
    47021,
    53578,
    59138,
    63928,
])

x = np.arange(0, len(cases_italy))
params, _ = curve_fit(logistic_function, x, cases_italy)
italy_pred = plot_logistic_fit(cases_italy, params)

r2_score(cases_italy, italy_pred)

diff = 100
days, cases = plateau(cases_italy, params, diff=diff)
print(f"{days} days until growth is lower than {diff} per day")
print(f"The total cases will be at {int(cases)}")

x = np.linspace(0, len(cases_italy) + 31)
y = logistic_function(x, *params)
plt.plot([len(cases_italy), len(cases_italy)], [0, 100000])
plt.plot(x, y)
plt.show()


