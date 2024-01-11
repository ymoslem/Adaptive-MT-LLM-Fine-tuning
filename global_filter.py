#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filtering/Cleaning parallel datasets for Machine Translation
# Command: python3 filter.py <source_file_path> <target_file_path> <source_lang> <target_lang>


import pandas as pd
import numpy as np
import re
import sys
import csv
from time import sleep


# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
# from IPython.display import display



def prepare(source_file, target_file, source_lang, target_lang, lower=False):

    df_source = pd.read_csv(source_file,
                            names=['Source'],
                            sep="\0",
                            quoting=csv.QUOTE_NONE,
                            skip_blank_lines=False,
                            on_bad_lines="skip")
    df_target = pd.read_csv(target_file,
                            names=['Target'],
                            sep="\0",
                            quoting=csv.QUOTE_NONE,
                            skip_blank_lines=False,
                            on_bad_lines="skip")
    df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
    print("Dataframe shape (rows, columns):", df.shape)


    # Delete nan
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Drop duplicates
    df = df.drop_duplicates()
    #df = df.drop_duplicates(subset=['Target'])

    print("--- Duplicates Deleted\t\t\t--> Rows:", df.shape[0])

    
    # Drop too-long rows (source or target)
    df["Too-Long"] = (df['Source'].str.len() > df['Target'].str.len() * 1.5) |  \
                     (df['Target'].str.len() > df['Source'].str.len() * 1.5) |  \
                     (df['Source'].str.count(' ')+1 > 70) |  \
                     (df['Target'].str.count(' ')+1 > 70)

    #display(df.loc[df['Too long'] == True]) # display only too long rows
    df = df.set_index(['Too-Long'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Long'], axis = 1)

    print("--- Too-Long Source/Target Deleted\t--> Rows:", df.shape[0])


    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("--- Rows Shuffled\t\t\t--> Rows:", df.shape[0])


    # Write the dataframe to two Source and Target files
    source_file = source_file+'-filtered.'+source_lang
    target_file = target_file+'-filtered.'+target_lang


    df_source = df["Source"]
    df_target = df["Target"]

    df_source.to_csv(source_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    print("--- Source Saved:", source_file)
    sleep(1)
    df_target.to_csv(target_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    print("--- Target Saved:", target_file)


if __name__ == "__main__":
    # Corpora details
    source_file = sys.argv[1]    # path to the source file
    target_file = sys.argv[2]    # path to the target file
    source_lang = sys.argv[3]    # source language
    target_lang = sys.argv[4]    # target language
    prepare(source_file, target_file, source_lang, target_lang, lower=False)
