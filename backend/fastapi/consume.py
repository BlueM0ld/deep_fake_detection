import pandas as pd
from sqlalchemy import create_engine
import os


def setup_db_connection():
    try:
        # Create a SQLAlchemy engine
        engine = create_engine("postgresql+psycopg2://postgres:docker@db/db")
        return engine
    except Exception as e:
        print("Error setting up DB connection: ", e)
        return None


def consume_data():
    print("WEEE!")
    # This will call data from the postgres db and convert to a DataFrame
    try:
        engine = setup_db_connection()

        if engine is not None:
            get_doc_query = """
                SELECT * FROM user_logs;
            """

            # Use pandas to read SQL query into a DataFrame
            df = pd.read_sql_query(get_doc_query, engine)

            # print(df.head())  # Print the first few rows of the DataFrame

            # cleaned_df = df[['search_query', 'related_docs']].copy()
            # # cleaned_df = cleaned_df[{
            # #     'search_query': 'query', 'related_docs': 'doc_relevant'}]

            # cleaned_df.rename(
            #     columns={'search_query': 'query', 'related_docs': 'doc_relevant'}, inplace=True)
            # Print the first few rows of the DataFrame
            # print(cleaned_df.head())

            df.to_parquet('df.parquet')

    except Exception as e:
        print("Error in getting data from the DB: ", e)


if __name__ == "__main__":
    consume_data()
