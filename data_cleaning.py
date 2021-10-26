"""
Where we got our data:
    - https://www.kaggle.com/pashupatigupta/emotion-detection-from-text
    - https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp


Our goals:
    - Remove the tweet_id column
    - Remove all the usernames (starts with "@") from df1's content column
    - Replace all the numbers with "@" sign
    - Remove the links
    - Order the target column names
    
Final form ot fhe target column will contain:
    - sadness
    - neutral
    - worry and fear (fear)
    - love
    - anger and hate (anger)
    - surprise, fun, enthusiasm and happiness (joy)
"""

import pandas as pd

# Create a function to remove all the usernames and links from the texts and replace numbers with "@"
def remove_usernames_and_links_and_replace_numbers(content):
    new_list = list(content.str.split())
    for i in range(len(new_list)):
        new_list[i] = [x for x in new_list[i] if not x.startswith("@") and not x.startswith("http")]
        for j in range(len(new_list[i])):
            if new_list[i][j].isdigit():
                new_list[i][j] = "@"
        new_list[i] = " ".join(new_list[i])
    return pd.Series(new_list)

# Create a function to read the lines of a document
def get_lines(filename):
  with open(filename, "r") as f:
    return f.readlines()

def delete_newLine(lines):    
    # Delete the "\n" from each line
    for i in range(len(lines)):
        lines[i] = lines[i][0:-1]
    return lines

# Create a list of dictionaries to turn it into a DataFrame
def create_dict_list(lines):
    
    abstract_lines = []
    for i in range(len(lines)):
        line_dict={}
        line_list = lines[i].split(";")
        line_dict["text"] = line_list[0]
        line_dict["target"] = line_list[1]
        abstract_lines.append(line_dict)
    return abstract_lines

# Create a function to order the target column names
def change_target_names(df, column):
    for i in range(len(df)):
        if df[column].iloc[i] == "hate":
            df[column].iloc[i] = "anger"
        elif df[column].iloc[i] == "worry":
            df[column].iloc[i] = "fear"
        elif df[column].iloc[i] == "surprise" or df[column].iloc[i] == "fun" or df[column].iloc[i] == "happiness" or df[column].iloc[i] == "enthusiasm":
            df[column].iloc[i] = "joy"
    return df

# Get df1
df1 = pd.read_csv("data/Raw Data/data 1/tweet_emotions.csv")

# Remove the tweet_id column from df1
df1.drop("tweet_id", axis=1, inplace=True)

# Change the column names and reorder columns
df1 = df1.rename(columns={"content": "text", "sentiment": "target"})
df1 = df1[["text", "target"]]

# Get lines of d2
lines_d2 = get_lines("data/Raw Data/data 2/emotion_data.txt")

# Delete new lines at the end of lines_d2
lines_d2 = delete_newLine(lines_d2)

# Create a list of dictionaries for lines_d2
dict_list_2 = create_dict_list(lines_d2)

# Create df2
df2 = pd.DataFrame(dict_list_2)

# Concatenate DataFrames
DF = pd.concat([df1, df2])

# Drop some rows to decrease the number of classes
DF.drop(DF[DF["target"] == "jo"].index, inplace=True)
DF.drop(DF[DF["target"] == "boredom"].index, inplace=True)
DF.drop(DF[DF["target"] == "empty"].index, inplace=True)
DF.drop(DF[DF["target"] == "relief"].index, inplace=True)

# Remove usernames and links and replace numbers with "@"
DF["text"] = remove_usernames_and_links_and_replace_numbers(DF["text"])

# Order the target column names
DF = change_target_names(DF, "target")

# Remove rows with Null/NaN values.
DF.dropna(inplace=True)

# Save the DataFrame as a csv
DF.to_csv("data/Cleaned Data/emotion_data_cleaned.csv", index=False)

load_df = pd.read_csv("data/Cleaned Data/emotion_data_cleaned.csv")

