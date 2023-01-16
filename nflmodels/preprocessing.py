import numpy as np
import pandas as pd


def preprocess_field_goal(data: pd.DataFrame, seasons = [], use_extra_points = True, use_wind = True, use_kickers = True, kickers = 'active', kicker_threshold = 100, return_X_y = False, verbose = True):
    '''
    Preprocesses nflverse play-by-play data for use in field goal prediction.
    Requires at least field_goal_result and yardline_100 columns to be present in the data.
    
        Parameters:
            data (DataFrame): Dataset to preprocess
            seasons (List): Which seasons to include, empty for all
            use_extra_points (bool): Include extra point attempts
            use_wind (bool): Include wind data
            use_kickers (bool): Include kicking player data
            kickers (str): "active" for currently active kickers, "all" for all kickers present in the dataset. Ignored if use_kickers == False
            kicker_threshold (int): How many attempts should a kicker have to be included individually. Ignored if use_kickers == False
            return_X_y (bool): Should the function return separate X and y data for features and target, returns complete DataFrame if False 
            verbose (bool): Print progress messages? default True
        
        Returns:
            data (DataFrame): Preprocessed dataset, if return_X_y = False
            (X (DataFrame), y (DataFrame)), if return_X_y = True
    '''
    
    query_string = ''
    features = ['kick_result', 'yardline_100',]
    categoricals = []
    
    # ===== Query data =====
    
    # add and separated with spaces if the given string is not empty
    def add_to_query(query_string: str, add: str):
        if len(query_string) != 0: return query_string + ' and ' + add
        else: return query_string + add
    
    # query only specified seasons, else use all seasons present in data
    if len(seasons) != 0:
        query_string = add_to_query(query_string, 'season in @seasons')
        
    # add play type to query
    if use_extra_points: query_string = add_to_query(query_string, '(play_type == "extra_point" or play_type == "field_goal")')
    else: query_string = add_to_query(query_string, 'play_type == "field_goal"')

    # returns all features
    data = data.query(query_string).copy()


    # ===== Process feature by feature =====
    
    # kick result
    data['field_goal_result'] = data['field_goal_result'].map({'blocked': 0, 'missed': 0, 'made': 1, None: 0})
    if use_extra_points:
        data['extra_point_result'] = data['extra_point_result'].map({'failed': 0, 'blocked': 0, 'aborted': 0, 'good': 1, None: 0})
        data['kick_result'] = np.maximum(data['field_goal_result'], data['extra_point_result'])
    else: data['kick_result'] = data['field_goal_result']
    
    # wind
    if use_wind:
        data['closed'] = data['roof'].map({'outdoors': 0, 'open': 0, 'closed': 1, 'dome': 1, None: 0})
        data['wind'] = np.where(data['closed'] == 1, 0, data['wind']) # Set wind to zero when the roof is closed, was NaN
        data['wind'] = np.where(data['wind'].isna(), 
                                data['weather'].str.split(' ').str[-2].replace(regex=r'[^0-9]', value='').replace('', None), 
                                data['wind']) # Fetch missing wind data from weather string
        data.dropna(axis=0, subset=['wind'], inplace=True) # Drop remaining NaN values
        data['wind'] = data['wind'].astype(int) 
        
        features.append('closed')
        features.append('wind')
    
    # kicker player
    if use_kickers:
        
        if kickers == 'active':
            #http://www.espn.com/nfl/players/_/position/k
            # TODO: automate fetch active kickers
            active_kickers = [
                'M.Badgley', 'T.Bass', 'C.Boswell', 'R.Bullock', 'H.Butker', 'D.Carlson','M.Crosby','C.Dicker',
                'J.Elliott', 'K.Fairbairn','N.Folk', 'G.Gano', 'M.Gay', 'R.Gould', 'G.Joseph', 'Y.Koo', 'W.Lutz',
                'B.Maher', 'C.McLaughlin', 'B.McManus', 'E.McPherson', 'J.Myers', 'R.Patterson', 'E.Pineiro', 'M.Prater',
                'J.Sanders', 'C.Santos', 'J.Slye', 'R.Succop', 'J.Tucker', 'T.Vizcaino', 'C.York', 'G.Zuerlein'
            ]

            # all but active kickers to "Other"
            data['kicker_player_name'] = np.where(np.isin(data['kicker_player_name'], active_kickers), data['kicker_player_name'], 'Other')
        
        elif kickers == 'all':
            pass
        
        else:
            raise Exception('Unknown value for parameter kickers, use "all" or "active"')
        
        # change kickers with fewer than threshold attempts to "Other"
        kickers_few_atts = data.groupby(by='kicker_player_name')['kicker_player_name'].agg(['count']).query(f'count <= {kicker_threshold}').reset_index()['kicker_player_name'].to_numpy()
        data['kicker_player_name'] = np.where(data['kicker_player_name'].isin(kickers_few_atts), 'Other', data['kicker_player_name'])
        
        features.append('kicker_player_name')
        categoricals.append('kicker_player_name')
    
    
    # ===== Select features =====
    data = data[features]
    
    
    # ===== Encode categoricals =====
    data = pd.get_dummies(data=data, columns=categoricals)
    
    
    if return_X_y:
        return (data.drop(['kick_result'], axis=1), data['kick_result'])
    return data
    
    
def preprocess_next_play(data : pd.DataFrame, seasons = [], 
            include_pass = True, include_run = True, include_fg = True, include_punt = True, include_qbkneel = False, include_qbspike = False, 
            return_X_y = False, map_target_to_int = False, verbose=True):
    '''
    Preprocesses nflverse play-by-play data for use in next play prediction.
    
        Parameters:
            data (DataFrame): Dataset to preprocess
            seasons (List): Which seasons to include, empty for all
            include_pass (bool): Use pass play type, default True
            include_run (bool): Use run play type, default True
            include_fg (bool): Use field_goal play type, default True
            include_punt (bool): Use punt play type, default True
            include_qbkneel (bool): Use qb_kneel play type, default False
            include_qbspike (bool): Use qb_spike play type, default False
            return_X_y (bool): Should the function return separate X and y data for features and target, returns complete DataFrame if False 
            map_target_to_int (bool): Encode playtype as integer, default False
            verbose (bool): Print progress messages? default True
        
        Returns:
    '''
    
    if verbose: print('Preprocessing {} rows...'.format(len(data)))
    
    query_string = ''
    target = ['play_type']
    continuous = ['score_differential', 'game_seconds_remaining', 'half_seconds_remaining', 
                'yardline_100', 'ydstogo', 'goal_to_go', 'posteam_type', 'shotgun']
    categoricals = ['qtr', 'down', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining']
    
    # add and separated with spaces if the given string is not empty
    def add_to_query(query_string: str, add: str):
        if len(query_string) != 0: return query_string + ' and ' + add
        else: return query_string + add
    
    play_types = []
    if include_pass: play_types.append('pass')
    if include_run: play_types.append('run')
    if include_fg: play_types.append('field_goal')
    if include_punt: play_types.append('punt')
    if include_qbkneel: play_types.append('qb_kneel')
    if include_qbspike: play_types.append('qb_spike')
    
    if verbose: print('Preprocessing for play types: {}'.format(play_types))
    
    query_string = add_to_query(query_string, 'play_type in @play_types')
    
    if len(seasons) != 0:
        query_string = add_to_query(query_string, 'season in @seasons')

    # query dataset to a copy
    data = data.query(query_string).copy()
    
    # drop NaN values
    if verbose: len_na = len(data)
    data.dropna(axis=0, inplace=True, subset=target+continuous+categoricals)
    if verbose: 
        print('Dropped {} rows containing NaN values.'.format(len_na - len(data)))
        print('Resulting dataset size is {} rows.'.format(len(data)))
    
    # ===== Preprocess =====
    data['posteam_type'] = data['posteam_type'].map({'home': 1, 'away': 0}).astype('int')
    #data['posteam_score'] = data['posteam_score'].astype('int')
    #data['defteam_score'] = data['defteam_score'].astype('int')
    data['score_differential'] = data['score_differential'].astype('int')
    data['down'] = data['down'].astype('int')
    data['yardline_100'] = data['yardline_100'].astype('int')
    data['posteam_timeouts_remaining'] = data['posteam_timeouts_remaining'].astype('int')
    data['defteam_timeouts_remaining'] = data['defteam_timeouts_remaining'].astype('int')
    
    
    # ===== Select features =====
    data = data[target+continuous+categoricals]
    
    
    # ===== Encode categoricals =====
    data = pd.get_dummies(data=data, columns=categoricals)
    
    # Transforms play type to integers, required for some models e.g. xgboost
    if map_target_to_int:
        if verbose: print('Encoding play types as integers...')
        map_dict = {}
        for i in range(len(play_types)):
            map_dict[play_types[i]] = i
        data[target[0]] = data[target[0]].map(map_dict)
        
        if return_X_y:
            if verbose: print('Returned preprocessed X, y, play_types')
            return (data.drop(target, axis=1), data[target], play_types)
        if verbose: print('Returned preprocessed DataFrame, play_types')
        return (data, play_types)
    else: 
        if return_X_y:
            if verbose: print('Returned preprocessed X, y')
            return (data.drop(target, axis=1), data[target])
        if verbose: print('Returned preprocessed DataFrame')
        return data
    
    
def preprocess_ep(data : pd.DataFrame, seasons = [], verbose : bool = True):
    '''
    
    '''
    
    if verbose: print('Preprocessing {} rows...'.format(len(data)))
    
    query_string = ''
    target = ['next_score']
    continuous = ['yardline_100', 'ydstogo', 'goal_to_go', 'half_seconds_remaining', 'game_seconds_remaining']
    categoricals = ['qtr', 'down', 'posteam_type', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining']
    
    # add and separated with spaces if the given string is not empty
    def add_to_query(query_string: str, add: str):
        if len(query_string) != 0: return query_string + ' and ' + add
        else: return query_string + add
    
    if len(seasons) != 0:
        query_string = add_to_query(query_string, 'season in @seasons')

    # query dataset and sort to a copy
    # game_id, play_id is not enough as there are some games where the id is broken --> use game id, game clock and then play id (if equal clock) to mitigate 
    if len(query_string) != 0:
        data = data.query(query_string).sort_values(by=['game_id' ,'game_seconds_remaining', 'play_id'], ascending=[True, False, True]).copy()
    else:
        data = data.sort_values(by=['game_id', 'game_seconds_remaining', 'play_id'], ascending=[True, False, True]).copy()
    
    # drop invalid rows, these include timeouts and start/end of quarter marks etc. 
    # not enough information is present to make these useful for analysis
    data = data.dropna(axis=0, subset=['posteam', 'yardline_100'])
    if verbose: print(f'After dropping some rows, dataset size is {len(data)}')
    
    
    # CANNOT drop kickoffs because may end up as a td --> TODO: must be considered separately? or encode down as some other value?
    
    
    # ========================= Calculate next scoring events for each play =========================

    
    # 1: Get scoring play score value
    # 2: Map scoring play values to scoring types (e.g. 6 = "td")
    # 3: Fill rest of the plays based on mapped values
    
    # score gained for play
    data['play_score_gained'] = np.where(data['sp'] == 1, data['score_differential_post'] - data['score_differential'], None)
    
    # extra point attempts, set to zero if no score made
    data['play_score_gained'] = np.where(((data['extra_point_attempt'] == 1) | (data['two_point_attempt'] == 1)) 
                                         & (data['play_score_gained'].isna()), 0, data['play_score_gained'])
    
    
    # map values to strings; SCORING PLAYS ONLY
    data['next_score'] = data['play_score_gained'].map({0: 'no_score', 
                                                        6: 'td', 3: 'fg', 1: 'pat',
                                                        -6: 'opp_td', -3: 'opp_fg',
                                                        None: None})

    # fix two point attempts and opponent returns on pat
    data['next_score'] = np.where((data['play_score_gained'] == 2) & (data['safety'] == 1), 'safety', data['next_score'])
    data['next_score'] = np.where((data['play_score_gained'] == 2) & (data['safety'] == 0), '2pat', data['next_score'])
    
    data['next_score'] = np.where((data['play_score_gained'] == -2) & (data['safety'] == 0), 'opp_patreturn', data['next_score'])
    data['next_score'] = np.where((data['play_score_gained'] == -2) & (data['safety'] == 1), 'opp_safety', data['next_score'])

    
    # Fill values for other (non scoring) plays
    # First, reset index to make iteration work, iterating is not "pandas-like" but here it is the simplest way to accomplish this task
    # Additionally, the execution speed is still very good
    data.reset_index(inplace=True)
    
    prev_half = ''
    prev_game_id = ''
    prev_score_gained = 'no_score'
    prev_posteam = ''
    
    datalength = len(data)
    
    # create temp lists for pandas columns, this massively improves speeds when iterating
    l_half = data['game_half'].values.tolist()
    l_game_id = data['game_id'].values.tolist()
    l_score_gained = data['play_score_gained'].values.tolist()
    l_posteam = data['posteam'].values.tolist()
    l_next_score = data['next_score'].values.tolist()
    
    # Iterate on ordered data, starting from the last play
    for i in range(datalength - 1, -1, -1):
        
        if verbose: #print(f'Processing row {datalength - i} out of {datalength}', end='\r')
            if (datalength - i) % 2500 == 0:
                print(f'Processing row {datalength - i} out of {datalength}', end='\r')
        
        # Get current row game id, half, score made and posteam
        row_half = l_half[i] 
        row_game_id = l_game_id[i] 
        row_score_gained = l_next_score[i] 
        row_posteam = l_posteam[i] 
        
        # Game or half changed from previous iteration -> reset
        if row_game_id != prev_game_id or row_half != prev_half: 
            prev_half = row_half
            prev_game_id = row_game_id
            prev_score_gained = 'no_score'
        
        # If row does not have score information = row is not a scoring play
        # Add "currentyl active" score to the row
        if row_score_gained is None:
            if row_posteam != prev_posteam:
                if prev_score_gained == 'no_score':
                    l_next_score[i] = prev_score_gained
                else:
                    if prev_score_gained.startswith('opp_'): l_next_score[i] = prev_score_gained[4:] 
                    else: l_next_score[i] = 'opp_' + prev_score_gained 
            else:
                l_next_score[i] = prev_score_gained 
        
        # Row is a scoring play
        # Save current row scoring information    
        else:
            prev_score_gained = row_score_gained
            prev_posteam = row_posteam        
        
    print(f'All {datalength} out of {datalength} rows processed!')
    
    data['next_score'] = l_next_score
    
    # delete invalid vals
    data = data[data['next_score'] != 'opp_pat'] # -1 or opponent pat should not be possible and is a result of faulty data
    
    # map next score type to next score value
    # td quarantees an extra point attempt, therefore should the value of a td be higher...?
    data['next_score_value'] = data['next_score'].map({'td': 6, 'fg': 3, 'safety': 2, '2pat': 2, 'pat': 1,
                                                       'opp_td': -6, 'opp_fg': -3, 'opp_safety': -2, 'opp_patreturn': -2,
                                                       'no_score': 0})
    
    if verbose: print(f'Final dataset size is {len(data)} rows')
    
    # ========================= Separate field goal and extra (& 2) point attempts =========================
    
    #data_fg = data.query()
    # should cover all possible extra-point-related plays
    data_pat = data.query('(extra_point_attempt == 1 or two_point_attempt == 1) or (next_score == "pat" or next_score == "2pat" or next_score == "opp_patreturn")')
    
    # remove fg and pat rows from main set
    #data_normal = pd.concat([data, data_fg]).drop_duplicates(keep=False)
    #data_normal = pd.concat([data_normal, data_pat]).drop_duplicates(keep=False)
    data_normal = pd.concat([data, data_pat]).drop_duplicates(keep=False)
    
    # ========================= Feature selection and engineering =========================
    
    data_normal_y = data_normal[target]
    data_normal_X = data_normal[continuous+categoricals]
    data_normal_X = pd.get_dummies(data=data_normal_X, columns=categoricals)
    
    data_pat_y = data_pat[target]
    data_pat_X = data_pat[continuous+categoricals]
    data_pat_X = pd.get_dummies(data=data_pat_X, columns=categoricals)
    
    #return data_normal, data_fg, data_pat
    return data_normal_X, data_normal_y, data_pat_X, data_pat_y