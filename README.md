# MADS_699_Kemp_Schoose_Thoreux
MADs Capstone

Data is pulled using get_game_data.py, get_recruit_data.py and get_roster_data.py from the CFBD api:
https://collegefootballdata.com/

In order to run the scripts, you will need an API key. An API key can be obtained from the
above linked site.

Your API key should be stored in a .env file as:
CFBD_API_KEY=[your key]

An example .env file is included in the repo.

Once that has been completed you can run scripts individually in order to request data, clean data, build the model, and run the streamlit app. Or, more conveniently, you can do it all at once with the run_pipeline_and_dashboard.py script.

RUN AT ONCE:
python app/run_pipeline_and_dashboard.py

RUN IN ORDER:
1. python app/get_game_data.py
2. python app/get_recruit_data.py
3. python app/get_roster_data.py
4. python app/merge_roster_rankings.py
5. python app/team_starters_pipe.py
6. python app/data_clean_and_split.py
7. python app/build_model.py
8. streamlit run app/streamlit_app.py