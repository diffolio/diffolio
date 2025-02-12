# Diffusion Models for Risk-Aware Portfolio Optimization
This is the official code repository for **"Diffusion Models for Risk-Aware Portfolio Optimization"**, submitted to **KDD 2025**.

## Code Information
All code is written in Python 3.8.5 and PyTorch 1.13.0.
This repository contains the implementation of **Diffolio** (<u>**Dif**</u>fusion Models for Risk-Aware Port<u>**folio**</u> Optimization), a novel diffusion-based framework that directly learns a pseudo-optimal portfolio distribution.
Instead of forecasting entire future time series, **Diffolio** directly samples portfolios, 
immediately adapting to user-specified risk levels through a dedicated risk guidance mechanism embedded in the denoising diffusion process.

### Code Structure
The code of Diffolio is in the `src` directory.
  * `main.py`: The main script that initializes the market environment and sequentially trains/tests the model.
  * `environment.py`: Defines the Market environment class.
  * `network.py`: Implements Diffolio's neural network architecture.
  * `normalizer.py`: Contains normalization methods for price signals.
  * `utils.py`: Includes utility functions.
  * `mysql.py`: Handles reading and preprocessing asset price data, including File I/O for CSV files.
  * `experiment.py`: Computes performance and risk metrics.

Additional directories:
  * `src/configs/`:  Configuration files (.yml) with default model arguments for each dataset. 
  * `src/utils/`: Utility functions, such as fixing seeds and saving results.
  * `src/diff_utils/`: Basic diffusion model utilities and neural network implementations.

### Dependencies
All required libraries/packages are listed in `requirements.txt`. To set up the environment:
  * To create the corresponding Conda environment:
    * `conda env create -n <your_env_name> -f requirements.txt`
  * Or if you already have an existing virtual environment:
    * `pip install -r requirements.txt`

## Dataset Information
The experiment utilized six real-world market datasets (U.S., KR, Crypto, CN, JP, U.K.).
Most datasets (except the CN stock market dataset) were collected via APIs (e.g., Yahoo Finance, Binance) or downloaded from educational institution websites (e.g., WRDS) for research purposes.
Unfortunately, adhering to the data redistribution policies of the data sources, we are unable to publicly release these market datasets.
Thus, we recommend the following approach to construct the dataset:

1) Each dataset's metadata is provided in the `{country}_info` file within the `data` directory, containing details such as start date, end date, tickers, and index names.

2) Following the sources and Terms of Use listed below, collect data for the specified tickers, align them with the given start and end dates, and store the prepared dataset in the data folder.   


* **Publicly Available Dataset**  
  * **CN stock market** (*34 stocks*) - Sourced from [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster).  

* **Extended Public Datasets**  
  * To ensure robustness over a longer testing period, we updated two datasets by extending the data period or increasing the number of assets:  
    * **JP stock market** (*118 stocks*) - Sourced from [DTML](https://datalab.snu.ac.kr/dtml/#datasets) and extended from [Yahoo Finance](https://finance.yahoo.com).  
    * **U.K. stock market** (*21 stocks*) - Sourced from [DTML](https://datalab.snu.ac.kr/dtml/#datasets) and extended from [Yahoo Finance](https://finance.yahoo.com).
      * [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html).  

* **Privately Acquired Datasets**  
  * To assess the model’s ability to select valuable assets from large markets, we utilized three additional datasets:  
    * **U.S. stock market** (*224 stocks*) - Sourced from [WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp).  
      * [WRDS Terms of Use](https://wrds-www.wharton.upenn.edu/users/tou).  
    * **KR stock market** (*528 stocks*) - Collected from [Yahoo Finance](https://finance.yahoo.com).  
      * [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html).  
    * **Crypto market** (*37 cryptocurrencies*) - Gathered from [Binance](https://binance.com).  
      * [Binance Terms of Use](https://www.binance.com/en/terms).  

### **Example Dataset in `data` Directory**
  * `stocks_cn.csv` - Publicly available CN stock market dataset (**h**).  
  * `index_cn.csv` - SSE50 index for the CN market (**g**).  



## Execution Guideline
You can run and test the code in command-line interface (CLI) like terminal with the following examples:
   
`python -u src/main.py` 
will run everything in a default settings.

Or, 

`python -u src/main.py --test_type denoising --date_from 2009-01-05 --date_to 2020-12-31 --dim1 192` 
will run with some of the specified keyword arguments.


Note that all the keyword arguments can be modified and list of available arguments are shown in `main.py`'s `add_arguments` methods.
