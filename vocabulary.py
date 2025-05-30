import pandas as pd
from os.path import exists
from parsing_utils import parse_file
import re, pickle

# python3 ../glem/glem/glem.py -f master_string.txt -v -s with_Frog

sample_file = './sample_col.csv'
path = "./sample_collection"
lemmas_path = "./lemmas/master_string.txt.with_Frog.wlt.txt"

def main():
    
    #test()

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
    # remove punctuations and same value letters
    cleaned = re.sub(r"\,|\.+|\·|\”|\'|\“|\(|\)", "", master_string)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"ττ", "σσ", cleaned)

    # some transformation for hanlding unexpected cases of Dotiki klisi
    # used unicode characters for the substitution
    cleaned = cleaned.split()    
    for i, w in enumerate(cleaned):
        match = re.search("ηι$|ῆι$|ᾶι$|ῶι$|ωι$", w)
        if match:
            if w[-2] == "ῆ":
                cleaned[i] = re.sub("ῆι$", "\u1fc7", w)  # ῆι - ῇ
            elif w[-2] == "ᾶ":
                cleaned[i] = re.sub("ᾶι$", "\u1fb7", w)  # ᾶι - ᾷ
            elif w[-2] == "ῶ":
                cleaned[i] = re.sub("ῶι$", "\u1ff7", w)  # ῶι - ῷ
            elif w[-2] == "ω":
                cleaned[i] = re.sub("ωι$", "\u1ff3", w)  # ωι - ῳ
            elif w[-2] == "η":
                cleaned[i] = re.sub("ηι$", "\u1fc3", w)  # ηι - ῃ

    cleaned = " ".join(cleaned)


    with open("./lemmas/master_string.txt", "w") as f:
        f.write(cleaned)

    lemmas, vocab = create_vocabulary(lemmas_path)
    create_iif(lemmas, sample_df)



def test():
    word = "διαβολίηι"
    new_word = re.sub("ηι$", "\u1fc3", word)
    print(word, "\n", new_word)

def create_vocabulary(source):
    # after creating the lemmas, create a list of unique lemmas to save
    lemmas = pd.read_csv(source, header= None, skipfooter=13, sep= "\t", names= ["Word", "Lemma", "POS-tag"], engine= "python")

    vocab = lemmas["Lemma"].drop_duplicates(ignore_index= True)

    vocab.to_csv("./lemmas/Lemmas_Vocabulary.csv", header= None, index= None)
    
    return lemmas, vocab

# inverted index file for later analysis
def create_iif(lemmas: pd.DataFrame, excerpts: pd.DataFrame):

    lemmas.drop_duplicates(subset= ["Word", "POS-tag"], inplace= True) # keep unique entries of words-pos_tag
    iif_df = lemmas.set_index(["Lemma", "Word"]).sort_index() # sort by lemmas and words
    iif_df["Excerpt"] = [[] for i in range(iif_df.size)] # create a column of lists of excerpt occurances
    
    print(type(excerpts))
    for text in excerpts:
        print(text[0])

    iif_df.to_csv("./test.csv")


if __name__ == "__main__":
    main()