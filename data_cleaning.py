"""
Where we got our data:
    - https://www.kaggle.com/pashupatigupta/emotion-detection-from-text
    - https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp
    - https://www.kaggle.com/ishantjuyal/emotions-in-text
"""

import pandas as pd
df1 = pd.read_csv("data/Raw Data/data 1/Emotion_final.csv")
df2 = pd.read_csv("data/Raw Data/data 2/tweet_emotions.csv")

"""
df1 has 2 columns: Text, Emotion.
df2 has 3 columns: tweet_id, sentiment, content

Our goals:
    - Remove the tweet_id column from df2
    - Remove all the usernames (starts with "@") from df2's content column
    - Replace all the numbers in df1 and df2 with "@" sign
    - Concatenate the dataframes
"""

# Remove the tweet_id column from df2
df2.drop("tweet_id", axis=1, inplace=True)

# Create a function to remove all the usernames from the texts and replace numbers with "@"
def remove_usernames_and_replace_numbers(content):
    new_list = list(content.str.split())
    for i in range(len(new_list)):
        new_list[i] = [x for x in new_list[i] if not x.startswith("@")]
        for j in range(len(new_list[i])):
            if new_list[i][j].isdigit():
                new_list[i][j] = "@"
        new_list[i] = " ".join(new_list[i])
    return pd.Series(new_list)

# Change the column names and reorder columns
df2 = df2.rename(columns={"content": "text", "sentiment": "target"})
df2 = df2[["text", "target"]]
df1 = df1.rename(columns={"Emotion": "target", "Text": "text"})

# Preprocess "data3"
## Create a function to read the lines of a document
def get_lines(filename):
  with open(filename, "r") as f:
    return f.readlines()

lines = get_lines("data/Raw Data/data 3/emotion_data.txt")

## Delete the "\n" from each line
for i in range(len(lines)):
    lines[i] = lines[i][0:-1]

## Creating a dictionary to turn it into a DataFrame
abstract_lines = []
for i in range(len(lines)):
    line_dict={}
    line_list = lines[i].split(";")
    line_dict["text"] = line_list[0]
    line_dict["target"] = line_list[1]
    abstract_lines.append(line_dict)
    
df3 = pd.DataFrame(abstract_lines)

# Concatenate DataFrames
DF = pd.concat([df1, df2, df3])
# Remove all of the usernames from the texts and replace all the numbers with "@"
DF["text"] = remove_usernames_and_replace_numbers(DF["text"])

DF.to_csv("data/Cleaned Data/emotion_data_cleaned.csv", index=False)

