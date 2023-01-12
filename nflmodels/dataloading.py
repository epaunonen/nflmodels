import os
import pg8000 as pg
import pandas as pd

PRESET_COLS = ['play_id', 'game_id', 'home_team', 'away_team', 'posteam', 'posteam_type', 'defteam', 'yardline_100',
    'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining', 'quarter_end', 'drive', 'sp', 'qtr', 'game_half',
    'down', 'goal_to_go', 'yrdln', 'ydstogo', 'ydsnet', 'play_type', 'yards_gained', 'shotgun', 'no_huddle', 'qb_dropback', 'qb_kneel',
    'qb_spike', 'qb_scramble', 'pass_length', 'pass_location', 'air_yards', 'yards_after_catch', 'run_location', 'run_gap', 'field_goal_result',
    'kick_distance', 'extra_point_result', 'two_point_conv_result', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'timeout', 
    'timeout_team', 'td_team', 'posteam_score', 'defteam_score', 'score_differential', 'posteam_score_post', 'defteam_score_post', 
    'score_differential_post', 'ep', 'epa', 'wp', 'def_wp', 'home_wp', 'away_wp', 'wpa', 'penalty', 'safety', 'interception', 'touchdown', 'pass_touchdown',
    'rush_touchdown', 'return_touchdown', 'extra_point_attempt', 'two_point_attempt', 'field_goal_attempt', 'kickoff_attempt', 'punt_attempt', 
    'complete_pass', 'penalty_team', 'penalty_yards', 'season', 'drive_ended_with_score', 'away_score', 'home_score', 'game_stadium', 'roof', 
    'surface', 'weather', 'wind', 'temp', 'kicker_player_name']

def load_data_postgresql(seasons : list = [], min_season : int = None, max_season : int = None, column_preset : bool = True, verbose=True, *args):
     
    '''
    Loads nflverse pbp data from a PostgreSQL server. 
    Obtains credentials from env.
    
        Parameters:
        seasons (list): Which seasons to include, all or defined min-max if empty; default empty
        min_season (int): First season to include, ignored if "seasons" is not empty; default None
        max_season (int): Last season to include, ignored if "seasons" is not empty; default None
        column_preset (bool): Load preset collection of columns necessary for generating the models; default True
        *args (str): Columns to load, if "column_preset" == True loads these in addition
        verbose (bool): Print progress messages; default True
        
        Returns:
        df (DataFrame): Loaded data
    '''
     
    USER = os.getenv('USER')
    PASSWORD = os.getenv('PASSWORD')
    HOST = os.getenv('HOST')
    PORT = os.getenv('PORT')
    DBNAME = os.getenv('DBNAME')
    TABLENAME = os.getenv('TABLENAME')
    
    columns = []
    conditions = []
    
    if column_preset:
        columns = PRESET_COLS
    
    for a in args:
        columns.append(a)
    
    column_string = ', '.join(columns)
    
    # either all seasons or using defined min-max
    if len(seasons) == 0:
        if min_season is not None and max_season is not None: # defined min-max
            conditions.append(f'season >= {min_season}')
            conditions.append(f'season <= {max_season}')
    else:
        s = ', '.join(list(map(str, seasons)))
        conditions.append(f'season IN ({s})')
        
    condition_string = ' AND '.join(conditions)

    # construct complete query string
    query = f'SELECT {column_string} FROM {TABLENAME}'
    if len(condition_string) != 0: query += f' WHERE {condition_string}'
    query += ';'

    if verbose: print(query)
    
    # complains but works!
    # https://hackersandslackers.com/connecting-pandas-to-a-sql-database-with-sqlalchemy/
    with pg.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, database=DBNAME) as conn:
        df = pd.read_sql_query(sql=query, con=conn)
    
    return df


def load_data_nflverse():
    raise NotImplementedError()