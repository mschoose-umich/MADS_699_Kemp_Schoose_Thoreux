# MADS_699_Kemp_Schoose_Thoreux
MADs Capstone

Data is pulled using multiple scripts from the CFBD api:
https://collegefootballdata.com/

In order to run the application, you will need an API key. An API key can be obtained from the
above linked site.

Your API key should be stored in a .env file as:
CFBD_API_KEY=[your key]

An example .env file is included in the repo.

If using a Mac, you will need to use homebrew to install libomp in order for lightgbm to function. Run this command before performing pip install:
brew install libomp

Install dependencies with:
pip install -r requirements.txt

RUN:
python app/run_all.py
