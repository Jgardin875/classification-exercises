import env
import pandas as pd

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



# TITANIC

def new_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))


import os

def get_titanic_data():
    filename = "titanic.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_titanic = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_titanic.to_csv(filename)

        # Return the dataframe to the calling code
        return df_titanic 


# IRIS

def new_iris_data():
    return pd.read_sql('SELECT * FROM measurements JOIN species using (species_id)', get_connection('iris_db'))


import os

def get_iris_data():
    filename = "iris.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_iris = new_iris_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_iris.to_csv(filename)

        # Return the dataframe to the calling code
        return df_iris


# TELCO

def new_telco_data():
    return pd.read_sql('''SELECT 
            *
            FROM
            customers
                JOIN
            contract_types USING (contract_type_id)
                JOIN
            internet_service_types USING (internet_service_type_id)
                JOIN
            payment_types USING (payment_type_id)''', get_connection('telco_churn'))


import os

def get_telco_data():
    filename = "telco.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_telco = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_telco.to_csv(filename)

        # Return the dataframe to the calling code
        return df_telco
