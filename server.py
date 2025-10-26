from mcp.server.fastmcp import FastMCP
from db_helper import get_customer_by_id, get_customers_by_name, DBSchema, get_table_column_names, query_database

mcp = FastMCP("db assistant")

@mcp.tool()
async def get_customer_info(customer_id: str) -> dict:
    """
    Get customer information by customer ID

    args:
        customer_id: str: The ID of the customer to retrieve information for
    
    returns:
        dict: A dictionary containing customer information or an error message
    """
    try:
        customer = get_customer_by_id(customer_id)
        return customer.model_dump()
    except ValueError as e:
        return {"error": str(e)}

@mcp.tool()
def get_customers_info_by_name(name: str) -> list[dict]:
    """
    Get customers by name

    args:
        name: str: The name of the customer to search for
    
    returns:
        list[dict]: A list of dictionaries containing customer information
    """
    try:
        customers = get_customers_by_name(name)
        return [customer.model_dump() for customer in customers]
    except ValueError as e:
        return [{"error": str(e)}]
    

@mcp.tool()
def get_columns_of_table(table_name: str) -> list[str]:     
    """
    Get the column names of a table in the database

    args:
        table_name: str: The name of the table to retrieve column names for
    
    returns:
        list[str]: A list of column names
    """
    try:
        columns = get_table_column_names(table_name)
        return columns
    except Exception as e:
        return [f"Error: {str(e)}"]
    
@mcp.tool()
def get_db_schema() -> dict:
    """
    Get the database schema

    returns:
        dict: A dictionary containing the database schema
    """
    schema = DBSchema()
    return schema.model_dump()


@mcp.tool()
async def query_database_tool(query) -> str:
    """
    Use this tool to query an oracle database and give the result of the query to the user.
    args:
        query: str: The SQL query to execute
    returns:
        str: A string representation of the query result
    
    Example:
        query = "SELECT * FROM fcjlive.sttm_customer FETCH FIRST 5 ROWS ONLY"
        result = query_database_tool(query)
    """
    return query_database(query.strip(";"))

@mcp.prompt()
async def system_prompt() -> str:
    return """You are a helpful assistant that helps users query a bank's customer database.
    You have access to the following tools:
    1. get_customer_info: Get customer information by customer ID
    2. get_customers_info_by_name: Get customers by name
    3. get_columns_of_table: Get the column names of a table in the database
    4. get_db_schema: Get the database schema and the names of all tables in the database
    5. query_database_tool: Query the database with SQL

    When using the query_database_tool, ensure that your SQL queries are safe and do not contain any harmful operations.
    Always use SELECT statements to retrieve data.
    Show the generated SQL query before executing it.
    You must understand the database schema before querying the database.
    You must understand the column names and retrieve relevant columns only.

    Here are some examples of how to use the tools:
    - To get customer information by ID, use: get_customer_info(customer_id="12345")
    - To get customers by name, use: get_customers_info_by_name(name="John Doe")
    - To get the column names of a table, use: get_columns_of_table(table_name="fcjlive.sttm_customer")
    - To get the database schema, use: get_db_schema()

    The database schema is as follows:
    {get_db_schema()}
    """



if __name__ == "__main__":
    mcp.run(transport="stdio")