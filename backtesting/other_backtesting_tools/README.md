This folder contains sample code related to other backtesting libraries (and some interfacing with ITF data).

# Notes on other backtesting tools

### PyAlgoTrade
http://gbeced.github.io/pyalgotrade/
- primarily for Python 2, there's a port for Python 3 but not everything is working (needed to dig into library code in one place; documented in my code in the science repo)
- got it to work with ITF data from CSV and directly from the DB
- crypto was an afterthought, supports only integer shares, did a fix to support fractional
- last update in 2016 but seems to work reasonably well
- support for technical indicators, multiprocess backtesting
- event-based (in contrast to our backtesting, trades executed on bar events, simulates the market)
- lots of interesting stats (Sharpe ratio, drawdown analysis, loss stats etc.), could be useful for doge baby evolution
- works with TA-Lib

### Enigma Catalyst
https://enigma.co/catalyst/
- based on Zipline with a focus on crypto
- provides historical pricing data, see https://enigma.co/catalyst/status/
- support for backtesting, various stats, also live trading and paper trading
- paper trading + doge babies = an interesting option?
- examples with TA-Lib
- seems to be well maintained

### Zipline
https://www.zipline.io
- clunky, not well maintained
- their own tutorial for stocks doesn't work
- couldn't get it to work after sinking a few hours into it, not with ITF data nor with their own code (API endpoints are broken; Python 3.5 <-> 3.6 compatibility issues, etc.)
