from data_sources import *
import pandas as pd

query = '''
    WITH c AS (
    SELECT resample_period, COUNT(*) AS total
    FROM   signal_signal
	WHERE signal = 'ANN_Simple'
	AND probability_up is not null
	GROUP BY resample_period
    )
SELECT signal_signal.resample_period, COUNT(*) / c.total::float as num_filtered_signals_percent, COUNT(*) as num_filtered_signals, c.total as total_signals 
FROM signal_signal JOIN c ON signal_signal.resample_period = c.resample_period 
WHERE signal = 'ANN_Simple' AND ABS(probability_up - probability_down) <= {} 
GROUP BY signal_signal.resample_period, c.total order by resample_period
'''

all_results = pd.DataFrame(columns=("percent_difference", "resample_period", "num_filtered_signals_percent", "num_filtered_signals", "total_signals"))
for percent in range(1,50):
    df = pd.read_sql(query.format(float(percent)/100), postgres_db.conn)
    df["percent_difference"] = pd.Series([percent]*3)
    all_results = all_results.append(df)

all_results = all_results.drop(["num_filtered_signals","total_signals"], axis = 1)
import matplotlib.pyplot as plt
#all_results.plot(x='percent_difference', y='num_filtered_signals_percent')
colors = ["r", "g", "b"]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()


plt.xticks(range(0,51,2))
plt.yticks([x/10.0 for x in range(11)])
plt.grid(zorder=0)

r60 = plt.scatter(all_results[all_results.resample_period==60]['percent_difference'], all_results[all_results.resample_period==60]['num_filtered_signals_percent'], color='r', zorder=3)
r240 = plt.scatter(all_results[all_results.resample_period==240]['percent_difference'], all_results[all_results.resample_period==240]['num_filtered_signals_percent'], color='g', zorder=3)
r1440 = plt.scatter(all_results[all_results.resample_period==1440]['percent_difference'], all_results[all_results.resample_period==1440]['num_filtered_signals_percent'], color='b', zorder=3)

plt.xlabel("Percent difference in probability_up and probability_down")
plt.ylabel("Percent filtered signals")
plt.legend((r60, r240, r1440),
           ('resample_period=60', 'resample_period=240', 'resample_period=1440'))

#plt.scatter(all_results['percent_difference'], all_results['num_filtered_signals_percent'], c=colors)
#ax.legend()
writer = pd.ExcelWriter("ann.xlsx")
all_results.to_excel(writer, "Results")
writer.save()
plt.show()
print(all_results)
