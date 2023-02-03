import pandas as pd
from nflmodels.dataloader.presets import preset_cols
from urllib.request import urlopen
import datetime

class nflverseLoader():
    
    def __init__(self):
        self.loc = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_'
        self.s_columns = []
        self.seasons = []
        
    def select(self, select_preset = False, columns = []):
        '''
        Set which columns will be queried from the database
        
        Parameters:
            select_preset (bool): Include preset columns? This includes all columns required to contruct the models in this package, default False
            columns (list): Individual columns to include
            
        Returns:
            self: nflverseloader object
        '''
        # add preset columns
        if select_preset: self.s_columns += preset_cols()
        # don't add if duplicate
        for c in columns:
            if c not in self.s_columns:
                self.s_columns.append(c)
        return self
    
    def where(self, seasons = []):
        '''
        
        '''
        self.seasons = seasons
        return self
    
    def load(self):
        
        if len(self.seasons) == 0:
            self.seasons = [*range(1999, datetime.datetime.now().year + 1)]
            
        return None