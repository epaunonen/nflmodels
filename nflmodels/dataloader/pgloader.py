import pg8000 as pg
import pandas as pd
from nflmodels.dataloader.presets import preset_cols

class PgLoader():
    
    def __init__(self, credentials = {}):
        self.c_user = credentials['user']
        self.c_password = credentials['password']
        self.c_host = credentials['host']
        self.c_port = credentials['port']
        self.c_dbname = credentials['dbname']
        self.c_tablename = credentials['tablename']
        
        self.s_columns = []
        self.where_string = ''
        
    def select(self, select_preset = False, columns = []):
        '''
        Set which columns will be queried from the database
        
        Parameters:
            select_preset (bool): Include preset columns? This includes all columns required to contruct the models in this package, default False
            columns (list): Individual columns to include
            
        Returns:
            self: PgLoader object
        '''
        
        # add preset columns
        if select_preset: self.s_columns += preset_cols()
        # don't add if duplicate
        for c in columns:
            if c not in self.s_columns:
                self.s_columns.append(c)
        return self

    def where(self, where_string : str = ''):
        '''
        Set conditions to query, expects a valid PostgreSQL expression.
        '''
        self.where_string = where_string
        return self
    
    def load(self, verbose=True):
        '''
        Connect to database and load data
        
        Parameters:
            verbose (bool): Print status messages, default True
            
        Returns:
            pandas DataFrame
        '''
        
        if len(self.s_columns) == 0: raise Exception('Unknown select statement, call .select() first to specify which columns to query.')
        
        select_string = ', '.join(self.s_columns)
        
        # construct query string from select and where strings
        query = f'SELECT {select_string} FROM {self.c_tablename}'
        if len(self.where_string) != 0: query += f' WHERE {self.where_string}'
        query += ';'
        
        if verbose: print(query)
    
        # complains but works!
        # https://hackersandslackers.com/connecting-pandas-to-a-sql-database-with-sqlalchemy/
        with pg.connect(user=self.c_user, password=self.c_password, host=self.c_host, port=self.c_port, database=self.c_dbname) as conn:
            df = pd.read_sql_query(sql=query, con=conn)
    
        return df
        