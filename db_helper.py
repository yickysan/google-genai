from dotenv import load_dotenv
import os
from pydantic import BaseModel
import oracledb
import pandas as pd
from typing import Callable, TypedDict

load_dotenv("C:/Users/aeniatorudabo/Desktop/FSDH-Internal-Audit-Continuous Monitoring/.env")


USER = os.environ["DB_USER"]
PASSWORD = os.environ["DB_PASSWORD"]
CONFIG_DIR = os.environ["CONFIG_DIR"]
DSN = os.environ["DSN"]

def query(query: str) -> pd.DataFrame:
    """
    This function connects to an oracle database and executes the query string
    and then closes the database connection.

    The result of the query is a `pandas.DataFrame` object
    """

    try:
        with oracledb.connect(user=USER, password=PASSWORD, dsn=DSN, config_dir=CONFIG_DIR) as conn:
            print("Connection OK!")
            with conn.cursor() as cursor:
                result = cursor.execute(query).fetchall()
                cols = [description[0] for description in cursor.description]
                df = pd.DataFrame(result, columns = cols)
                return df
        
    except Exception as e:
        print("Connection Failed")
        raise(e)


class DBCustomer(BaseModel):
    customer_id: str
    first_name: str
    last_name: str
    name: str
    email: str
    customer_bvn: str
    phone_number: str


class DBSchema(BaseModel):
    customer: str =  "fcjlive.sttm_customer"
    account: str = "fcjlive.sttm_cust_account"
    personal: str = "fcjlive.sttm_cust_personal"
    corporate: str = "fcjlive.sttm_cust_corporate"
    bvn: str = "fcjlive.vw_custac_bvns"
    bvn_2: str = "fcjlive.vw_cust_bvn"
    transaction: str = "fcjlive.acvw_all_ac_entries"


def get_table_column_names(table_name: str) -> list[str]:
    """
    This function gets the column names of a table in the database
    """
    query_str = f"""
    SELECT * from {table_name}
    fetch first 1 row only
    """
    df = query(query_str).rename(columns=str.lower)
    return df.columns.tolist()



def get_customer_by_id(customer_id: str) -> DBCustomer:
    """
    This function gets a customer by id from the database
    """
    query_str = f"""
    SELECT c.customer_no customer_id, p.first_name, p.last_name,
           c.customer_name1 name, nvl(p.e_mail, '') email, nvl(p.telephone, '') phone_number, 
           b.customer_bvn 
      FROM fcjlive.sttm_customer c
      JOIN fcjlive.sttm_cust_personal p on p.customer_no = c.customer_no
      JOIN fcjlive.vw_custac_bvns b on b.cust_no = c.customer_no
     WHERE c.customer_no = '{customer_id}'"""
    
    df = query(query_str).rename(columns=str.lower)
    if df.empty:
        raise ValueError(f"No customer found with id {customer_id}")
    customer_dict = df.iloc[0].to_dict()
    return DBCustomer(**customer_dict)


def get_customers_by_name(name: str) -> list[DBCustomer]:
    """
    This function gets customers by name from the database
    """
    query_str = f"""
    SELECT c.customer_no customer_id, p.first_name, p.last_name,
           c.customer_name1 name, nvl(p.e_mail, '') email, nvl(p.telephone, '') phone_number, 
           b.customer_bvn 
      FROM fcjlive.sttm_customer c
      JOIN fcjlive.sttm_cust_personal p on p.customer_no = c.customer_no
      JOIN fcjlive.vw_custac_bvns b on b.cust_no = c.customer_no
     WHERE LOWER(c.customer_name1) LIKE LOWER('%{name}%')"""
    
    df = query(query_str).rename(columns=str.lower)
    if df.empty:
        raise ValueError(f"No customer found with name {name}")
    customers = []
    for _, row in df.iterrows():
        customer_dict = row.to_dict()
        customers.append(DBCustomer(**customer_dict))
    return customers


def query_database(query_string: str) -> str:
    """
    This function returns a function that can be used to query the database
    """
    return query(query_string).to_string()

