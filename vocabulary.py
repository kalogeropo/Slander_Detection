import pandas as pd
from os.path import exists
from parsing_utils import parse_file
import re


sample_file = './sample_col.csv'
path = "./sample_collection"


if __name__ == "__main__":

    if not exists(sample_file):
        # create the sample_col.csv
        df = pd.DataFrame.from_dict(parse_file(path))
        df.to_csv(sample_file, index=False)

    #load all texts into a string to parse
    sample_df = pd.read_csv(sample_file, usecols= [2], skiprows= [1])

    text_list = sample_df["excerpt"].to_list()
    master_string = ' '.join(text_list)
    cleaned = re.sub(r"[,]", "", master_string)
    print(cleaned)

    store_path = "./lemmas/"
    with open("./lemmas/master_string.txt", "w") as f:
        f.write(cleaned)