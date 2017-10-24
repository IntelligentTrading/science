# Data Science, ML, NLP
### MA_indicators.ipynb
Initial SMA (Simple Moving Average) implementation.

It gets a CSV file exporded from DB (one-day snapshot), imports it into a Pandas DataFrame, then implements SMA low/high, indicators A,B,C, bullish/bearish market as well as signals to buy/sell. 

TA-Lib installation is required. 

TODO: better detection of intersection of the indicators... some false positive signals because plots are too close to each other and difference is smaller then epsilon.