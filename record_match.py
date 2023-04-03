import warnings
import numpy as np
import pandas as pd
import recordlinkage 
from recordlinkage.index import Block
from recordlinkage.preprocessing import phonetic
import io
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode

warnings.filterwarnings('ignore')

st. set_page_config(layout="wide")

def perform_record_linkage(input_path, source_path):
    # read the data
    input = pd.read_csv(input_path)
    # input1 = pd.read_csv(input_path)

    source = pd.read_csv(source_path)
    source = source.set_index('rec_id')

    # convert date of birth as string in input table
    input['date_of_birth'] = pd.to_datetime(input['date_of_birth'],format='%Y%m%d', errors='coerce')
    input['YearB'] = input['date_of_birth'].dt.year.astype('Int64') 
    input['MonthB'] = input['date_of_birth'].dt.month.astype('Int64') 
    input['DayB'] = input['date_of_birth'].dt.day.astype('Int64') 

    input['metaphone_given_name'] = phonetic(input['given_name'], method='metaphone')
    input['metaphone_surname'] = phonetic(input['surname'], method='metaphone')

    # convert date of birth as string in source table
    source['date_of_birth'] = pd.to_datetime(source['date_of_birth'],format='%Y%m%d', errors='coerce')
    source['YearB'] = source['date_of_birth'].dt.year.astype('Int64') 
    source['MonthB'] = source['date_of_birth'].dt.month.astype('Int64') 
    source['DayB'] = source['date_of_birth'].dt.day.astype('Int64') 
    source['metaphone_given_name'] = phonetic(source['given_name'], method='metaphone')
    source['metaphone_surname'] = phonetic(source['surname'], method='metaphone')

    indexer = recordlinkage.Index()
    indexer.block(left_on=['metaphone_given_name','metaphone_surname','date_of_birth'], 
                  right_on=['metaphone_given_name','metaphone_surname','date_of_birth'])
    candidate_record_pairs = indexer.index( input, source)

    print("Number of record pairs :",len(candidate_record_pairs))
    candidate_record_pairs.to_frame(index=False)

    compare_cl = recordlinkage.Compare()
    compare_cl.string('given_name', 'given_name', method='jarowinkler', threshold=0.85, label='given_name')
    compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
    compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
    compare_cl.exact('soc_sec_id', 'soc_sec_id', label='soc_sec_id')
    compare_cl.string('address_1', 'address_1', method='levenshtein', threshold=0.85, label='address_1')
    compare_cl.string('address_2', 'address_2', method='levenshtein', threshold=0.85, label='address_2')
    compare_cl.string('suburb', 'suburb', method='levenshtein', threshold=0.85, label='suburb')
    compare_cl.exact('postcode', 'postcode', label='postcode')
    compare_cl.exact('state', 'state', label='state')

    features = compare_cl.compute(candidate_record_pairs, input, source)
    # print(features.head(50))

    #Classifier
    ecm = recordlinkage.ECMClassifier()
    matches = ecm.fit(features)
    p = ecm.prob(features)
    # p = ecm.predict
    # print(p.tail(50))

    match_table= pd.DataFrame(p)
    match_table.reset_index(inplace=True)
    match_table = match_table.rename(columns={'level_0': 'input_index', 0: 'prob'})
    match_table['Status'] = 'Unsure'

    # set conditions for 'Match' column based on 'Value' column
    match_table.loc[match_table['prob'] > 0.8, 'Status'] = 'Duplicate'
    # df1.to_csv('master_table.csv')
    #match_table.loc[match_table['prob'] < 0.3, 'Status'] = 'Unsure'


    input_ = input.reset_index()
    source_ = source.reset_index()

    # print(source_.columns)
    # print(input_.columns)


    merged_df = pd.merge(match_table, input_, left_on='input_index', right_on='index', how='inner')

    new_indexes = input_[~input_['index'].isin(merged_df['index'].unique())]['index']

    # Get the rows in input_ with the new indexes
    new_rows = input_[input_['index'].isin(new_indexes)]

    # Append the new rows to merged_df
    merged_df = merged_df.append(new_rows, ignore_index=True)
    merged_df['Status'].fillna('Unique', inplace=True)
    merged_df['prob'].fillna(0, inplace=True)
    merged_df.drop(columns=['input_index'], inplace=True)
    # merged_df['date_of_birth'] = pd.to_datetime(merged_df[['YearB', 'MonthB', 'DayB']])
    #merged_df['date_of_birth'] = merged_df.apply(lambda row: str(row['YearB']) + str(row['MonthB']).zfill(2) + str(row['DayB']).zfill(2), axis=1)
    #merged_df.drop(columns=['YearB','MonthB', 'DayB', 'metaphone_given_name', 'metaphone_surname'], inplace=True)
    merged_df = merged_df.rename(columns={'index': 'Input_Index'})
    merged_df = merged_df[['Input_Index', 'given_name', 'surname', 'Status', 'prob', 'rec_id']]
    print(merged_df)

    match_df = source_.loc[source_['rec_id'].isin(match_table['rec_id'])].reset_index(drop=True)
    match_df['date_of_birth'] = match_df.apply(lambda row: str(row['YearB']) + str(row['MonthB']).zfill(2) + str(row['DayB']).zfill(2), axis=1)


    return match_table, input_, merged_df, match_df




st.title("Record Matching")

# Upload source file
source_file = st.file_uploader(
    label="Upload Source File (CSV)",
    type="csv",
    accept_multiple_files=False,
    key="source_file_" + str(hash("source_file")),
)

# Upload input file
input_file = st.file_uploader(
    label="Upload Input File (CSV)",
    type="csv",
    accept_multiple_files=False,
    key="input_file_" + str(hash("input_file")),
)

# Add custom styles to file uploader
input_file_style = """
<style>
#input_file_{} {{
    padding: 8px;
    background-color: #f5f5f5;
    color: #444;
    border-radius: 4px;
    border: none;
    box-shadow: none;
    display: inline-block;
    font-size: 14px;
    font-weight: 400;
    line-height: 20px;
    margin: 0px 4px 8px 0px;
    vertical-align: middle;
}}
</style>
""".format(hash("input_file"))

st.markdown(input_file_style, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .stButton button {
        background-color: green;
        color: white;
    }
    # .centered {
    #     display: flex;
    #     justify-content: center;
    #     align-items: center;
    # }
    </style>
    """,
    unsafe_allow_html=True,
)

if input_file and source_file:


    df = pd.read_csv(input_file, encoding='utf-8')
    st.write("# Edit Input File")
    # Create grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gridOptions = gb.build()
    
    # Display the AgGrid
    selected_rows = AgGrid(df, gridOptions=gridOptions, height=200, width=200,
                            update_mode=GridUpdateMode.VALUE_CHANGED,
                            on_grid_ready=None)
    

    # Process button
    if st.button("Process"):
        updated_data = selected_rows['data']
        updated_df = pd.DataFrame(updated_data)

        # Convert updated table data to a CSV string
        updated_csv = updated_df.to_csv(index=False)

        # Use updated table as input file
        input_file = io.StringIO(updated_csv)
        # Perform record linkage
        match_table,input_data, merged_df, match_df = perform_record_linkage(input_file, source_file)
        

        # Filter data for the three values of interest
        values_of_interest = ['Duplicate', 'Unique', 'Unsure']
        filtered_data = merged_df[merged_df['Status'].isin(values_of_interest)]

        # Calculate counts for each value of interest
        value_counts = filtered_data['Status'].value_counts()

        # Create a pandas DataFrame from the counts
        count_table = pd.DataFrame({'Status': value_counts.index, 'Count': value_counts.values})

        # Display count table in Streamlit
        st.write("# Summary : ", count_table)

        # Show match table
        # st.write("# Processed Input Data")
        # st.dataframe(input_data)
        st.write("# Output Data")
        st.dataframe(merged_df)
        st.write("# Matched Data from Master Table")
        st.dataframe(match_df)

        #Download match table
        csv = merged_df.to_csv(index=False)
        st.download_button(
            label="Download Match Table",
            data=csv,
            file_name="match_table.csv",
            mime="text/csv",
        )


