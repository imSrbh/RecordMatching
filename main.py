import warnings
import numpy as np
import pandas as pd
import recordlinkage 
from recordlinkage.index import Block
from recordlinkage.preprocessing import phonetic
warnings.filterwarnings('ignore')

def perform_record_linkage(input_path, source_path):
    # read the data
    input = pd.read_csv(input_path)
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
    print(features.head(50))

   


    ecm = recordlinkage.ECMClassifier()
    matches = ecm.fit(features)
    p = ecm.prob(features)
    # p = ecm.predict
    print(p.tail(50))

    match_table= pd.DataFrame(p)
    match_table.reset_index(inplace=True)
    # df2 = df1.rename(columns={'level_0': 'input_index'}, {'rec_id': 'match_index'}, {0: 'prob'})

    match_table = match_table.rename(columns={'level_0': 'input_index', 0: 'prob'})

    print(match_table)

    return match_table


import streamlit as st


st.title("Record Linkage App")


# Upload input file
st.sidebar.title("Upload Input File")
input_file = st.sidebar.file_uploader(
    label="Choose a CSV file",
    type="csv",
    key="input_file_" + str(hash("input_file")),
)

# Upload source file
st.sidebar.title("Upload Source File")
source_file = st.sidebar.file_uploader(
    label="Choose a CSV file",
    type="csv",
    key="source_file_" + str(hash("source_file")),
)

if input_file and source_file:
    # Process button
    if st.button("Process"):
        # Perform record linkage
        match_table = perform_record_linkage(input_file, source_file)

        # Download match table
        csv = match_table.to_csv(index=False)
        st.download_button(
            label="Download Match Table",
            data=csv,
            file_name="match_table.csv",
            mime="text/csv",
        )

        # Show match table
        st.dataframe(match_table)