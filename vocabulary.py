import pandas as pd
from os.path import exists
from parsing_utils import parse_file
import re


sample_file = './sample_col.csv'
path = "./sample_collection"
lemmas_path = "./lemmas/master_string.txt.with_Frog.wlt.txt"

def main():

    test()
    
    if not exists(sample_file):
        # create the sample_col.csv
        df = pd.DataFrame.from_dict(parse_file(path))
        df.to_csv(sample_file, index=False)

    if not exists(lemmas_path):
        print("There's no lemma file!\nPlease go to /lemmas and run\n[ python3 ../glem/glem/glem.py -f master_string.txt -v -s with_Frog ]")
        return
    #load all texts into a string to parse
    sample_df = pd.read_csv(sample_file, usecols= [2], skiprows= [1])

    text_list = sample_df["excerpt"].to_list()
    master_string = ' '.join(text_list)
    cleaned = re.sub(r"\,|\.+|\·|\”|\'|\“|\(|\)", "", master_string)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"ττ", "σσ", cleaned)


    store_path = "./lemmas/"
    with open("./lemmas/master_string.txt", "w") as f:
        f.write(cleaned)

    create_vocabulary(lemmas_path)

def test():
    print([ord(c) for c in "ώτατόνο"])

def create_vocabulary(source):
    # after creating the lemmas, create a list of unique lemmas to save
    lemmas = pd.read_csv(source, header= None, skipfooter=13, sep= "\t", names= ["Word", "Lemma", "POS-tag"], engine= "python")

    vocab = lemmas["Lemma"].drop_duplicates(ignore_index= True)

    print(lemmas.sort_values(by= "Lemma", axis= 0))
    print(vocab)


if __name__ == "__main__":
    main()