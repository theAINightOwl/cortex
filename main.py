import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import Complete
import os

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "account": st.secrets["snowflake"]["account"],
    "warehouse": "TED_TALKS_WH",
    "database": "TED_TALKS_DB",
    "schema": "TED_TALKS_SCHEMA"
}

# Constants
MAX_RESULTS_PER_PAGE = 50
DEFAULT_LIMIT = 100

def get_snowflake_session():
    """Get a Snowflake session."""
    return Session.builder.configs(SNOWFLAKE_CONFIG).create()

def initialize_snowflake():
    """Initialize Snowflake resources."""
    session = get_snowflake_session()
    try:
        # Create warehouse
        session.sql("""
            CREATE WAREHOUSE IF NOT EXISTS TED_TALKS_WH
            WITH WAREHOUSE_SIZE = 'XSMALL'
            AUTO_SUSPEND = 60
            AUTO_RESUME = TRUE
        """).collect()
        
        # Create database
        session.sql("CREATE DATABASE IF NOT EXISTS TED_TALKS_DB").collect()
        
        # Create schema
        session.sql("CREATE SCHEMA IF NOT EXISTS TED_TALKS_DB.TED_TALKS_SCHEMA").collect()
        
        # Set context
        session.sql("USE DATABASE TED_TALKS_DB").collect()
        session.sql("USE SCHEMA TED_TALKS_SCHEMA").collect()
        session.sql("USE WAREHOUSE TED_TALKS_WH").collect()
        
        # Create table for TED talks data with actual CSV structure
        session.sql("""
            CREATE TABLE IF NOT EXISTS ted_talks (
                VIDEO_TITLE VARCHAR(1000),
                THUMBNAIL VARCHAR(1000),
                VIDEO_DESCRIPTION VARCHAR(4000),
                VIDEO_YEAR INTEGER
            )
        """).collect()
        
        return True
    except Exception as e:
        st.error(f"Error initializing Snowflake: {e}")
        return False

def get_top_results_summary(results_df):
    """Generate a coherent summary for top 3 results using Cortex LLM Complete."""
    try:
        session = get_snowflake_session()
        
        # Create a single prompt with all three videos
        videos = []
        for idx in range(min(3, len(results_df))):
            row = results_df.iloc[idx]
            videos.append(f"""Talk {idx + 1}:
Title: {row['VIDEO_TITLE']}
Description: {row['VIDEO_DESCRIPTION']}""")
        
        prompt = f"""Here are the top 3 TED talks from a search. Please provide a coherent summary that connects these talks and their main themes in 3-4 sentences:

{chr(10).join(videos)}"""
        
        response = Complete(
            model="mistral-large2",
            prompt=prompt,
            session=session
        )
        
        return [{
            'title': 'Top 3 Talks Summary',
            'summary': response.strip()
        }]
    except Exception as e:
        st.error(f"Error generating summaries: {e}")
        return None

def upload_csv_to_snowflake(file_path):
    """Upload CSV data to Snowflake."""
    session = get_snowflake_session()
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Rename columns to match Snowflake table
        df = df.rename(columns={
            'Title': 'VIDEO_TITLE',
            'Thumbnail URL': 'THUMBNAIL',
            'Description': 'VIDEO_DESCRIPTION',
            'Year': 'VIDEO_YEAR'
        })
        
        # Convert DataFrame to Snowpark DataFrame
        snowpark_df = session.create_dataframe(df)
        
        # Write to Snowflake table
        snowpark_df.write.mode("overwrite").save_as_table("ted_talks")
        return True
    except Exception as e:
        st.error(f"Error uploading data: {e}")
        return False

def semantic_search(query: str, page: int = 1):
    """Perform semantic search using Cortex Search Service."""
    try:
        session = get_snowflake_session()
        root = Root(session)
        
        # Get the search service
        svc = root.databases["TED_TALKS_DB"].schemas["TED_TALKS_SCHEMA"].cortex_search_services["TED_SEARCH_SERVICE_CS"]
        
        # Calculate offset based on page number
        offset = (page - 1) * MAX_RESULTS_PER_PAGE
        
        # Perform the search with pagination
        response = svc.search(
            query,
            ["VIDEO_TITLE", "VIDEO_DESCRIPTION", "THUMBNAIL", "VIDEO_YEAR"],
            limit=MAX_RESULTS_PER_PAGE,
            offset=offset
        )
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(response.results)
        
        # Get total count for pagination
        total_count = session.sql("SELECT COUNT(*) FROM ted_talks").collect()[0][0]
        
        return results_df, total_count
    except Exception as e:
        st.error(f"Error performing semantic search: {e}")
        return None, 0

def main():
    st.title("Video Search")
    
    # Create tabs for different functionalities
    upload_tab, search_tab = st.tabs(["Upload Data", "Search Talks"])
    
    with upload_tab:
        st.write("Initialize Snowflake resources and upload TED talks data.")
        
        # Initialize Snowflake resources
        if 'initialized' not in st.session_state:
            with st.spinner("Initializing Snowflake resources..."):
                success = initialize_snowflake()
                if success:
                    st.success("✅ Snowflake resources initialized successfully!")
                    st.session_state.initialized = True
                else:
                    st.error("❌ Failed to initialize Snowflake resources")
                    return
        
        # Upload data
        if st.button("Upload TED Talks Data"):
            with st.spinner("Uploading data to Snowflake..."):
                csv_path = "youtube_data_TED.csv"
                if os.path.exists(csv_path):
                    if upload_csv_to_snowflake(csv_path):
                        st.success("✅ Data uploaded successfully!")
                    else:
                        st.error("❌ Failed to upload data")
                else:
                    st.error("❌ CSV file not found")
        
        # Display table preview if data exists
        if st.button("Preview Data"):
            session = get_snowflake_session()
            try:
                result = session.sql("SELECT * FROM ted_talks LIMIT 5").collect()
                if result:
                    df = pd.DataFrame(result)
                    st.dataframe(df)
                else:
                    st.info("No data available in the table yet.")
            except Exception as e:
                st.error(f"Error previewing data: {e}")
    
    with search_tab:
        st.write("Search YouTube videos using natural language.")
        
        # Search input
        query = st.text_input("Enter your search query:", 
                             placeholder="E.g., talks about artificial intelligence and its impact on society")
        
        if query:
            # Initialize or get current page from session state
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            
            with st.spinner("Searching..."):
                results, total_count = semantic_search(query, st.session_state.current_page)
                
                if results is not None and not results.empty:
                    # Generate summaries for top 3 results if on first page
                    if st.session_state.current_page == 1:
                        st.markdown("### 🌟 Top 3 Talks Summary")
                        with st.spinner("Generating summaries..."):
                            summaries = get_top_results_summary(results)
                            if summaries:
                                for summary in summaries:
                                    with st.container(border=True):
                                        st.markdown(f"_{summary['summary']}_")
                        st.markdown("---")
                    
                    total_pages = (total_count + MAX_RESULTS_PER_PAGE - 1) // MAX_RESULTS_PER_PAGE
                    
                    # Display pagination controls
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.write(f"Page {st.session_state.current_page} of {total_pages}")
                        
                        # Pagination buttons
                        prev, next = st.columns(2)
                        with prev:
                            if st.session_state.current_page > 1:
                                if st.button("← Previous"):
                                    st.session_state.current_page -= 1
                                    st.rerun()
                        with next:
                            if st.session_state.current_page < total_pages:
                                if st.button("Next →"):
                                    st.session_state.current_page += 1
                                    st.rerun()
                    
                    st.success(f"Showing results {((st.session_state.current_page-1)*MAX_RESULTS_PER_PAGE)+1} to {min(st.session_state.current_page*MAX_RESULTS_PER_PAGE, total_count)} of {total_count} talks")
                    
                    # Display results in a nice format
                    for _, row in results.iterrows():
                        with st.container(border=True):
                            st.markdown(f"### {row['VIDEO_TITLE']}")
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write("**Description:**")
                                st.write(row['VIDEO_DESCRIPTION'][:300] + "..." if len(row['VIDEO_DESCRIPTION']) > 300 else row['VIDEO_DESCRIPTION'])
                            with col2:
                                st.write(f"**Year:** {row['VIDEO_YEAR']}")
                                if row['THUMBNAIL']:
                                    st.image(row['THUMBNAIL'], use_container_width=True)
                else:
                    st.warning("No results found.")

if __name__ == "__main__":
    main() 