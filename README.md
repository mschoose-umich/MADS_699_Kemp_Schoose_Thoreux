# MADS_699_Kemp_Schoose_Thoreux
MADs Capstone

Data is pulled using get_game_data.py, get_recruit_data.py and get_roster_data.py from the CFBD api:
https://collegefootballdata.com/

In order to run the scripts, you will need an API key. An API key can be obtained from the
above linked site.

Your API key should be stored in a .env file as:
CFBD_API_KEY=[your key]

An example .env file is included in the repo.

If using a Mac, you will need to use homebrew to install libomp in order for lightgbm to function. Run this command before performing pip install:
brew install libomp

Install dependencies with:
pip install -r requirements.txt

Once that has been completed you can run scripts individually in order to request data, clean data, build the model, and run the streamlit app. Or, more conveniently, you can do it all at once with the run_pipeline_and_dashboard.py script.

RUN:
python app/run_all.py
