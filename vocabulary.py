import pandas as pd
from os.path import exists
from parsing_utils import parse_file
import re, pickle
import spacy

# python3 ../glem/glem/glem.py -f master_string.txt -v -s with_Frog

sample_file = './sample_col.csv'
path = "./sample_collection"
lemmas_path_glem = "./lemmas/master_string.txt.with_Frog.wlt.txt"
lemmas_path_grecy = "./lemmas/grecy_lemmas.csv"



def main():

    if not exists(sample_file):
        # create the sample_col.csv
        df = pd.DataFrame.from_dict(parse_file(path))
        df.to_csv(sample_file, index=False)

    if not exists(lemmas_path_glem):
        print("There's no lemma file!\nPlease go to /lemmas and run\n[ python3 ../glem/glem/glem.py -f master_string.txt -v -s with_Frog ]")
        return
    if not exists(lemmas_path_grecy):
        print("There's no lemma file!\nPlease run grecy_proiel_trf()\nand produce a lemmas csv.\nIf you don't have the grecy module, please download it.")

    #load all texts into a string to parse
    sample_df = pd.read_csv(sample_file, usecols= [2], skiprows= [1])

    sample_df["excerpt"] = sample_df["excerpt"].apply(text_clean_up)

    text_list = sample_df["excerpt"].to_list()
    master_string = ' '.join(text_list)

    with open("./lemmas/master_string.txt", "w") as f:
        f.write(master_string)

    lemmas, vocab = create_vocabulary(lemmas_path_glem)
    print(lemmas)
    #create_iif(lemmas, sample_df)


def create_vocabulary(source):
    # after creating the lemmas, create a list of unique lemmas to save
    if source == lemmas_path_glem:
        lemmas = pd.read_csv(source, header= None, skipfooter=13, sep= "\t", names= ["Word", "Lemma", "POS-tag"], engine= "python")

        vocab = lemmas["Lemma"].drop_duplicates(ignore_index= True)

        vocab.to_csv("./lemmas/Lemmas_Vocabulary.csv", header= None, index= None)
    
    elif source == lemmas_path_grecy:
        lemmas = pd.read_csv(source, header= 0, names= ["Word", "Lemma", "POS-tag"], engine= "python")

        vocab = lemmas["Lemma"].drop_duplicates(ignore_index= True)

        vocab.to_csv("./lemmas/Lemmas_Vocabulary_grecy.csv", header= None, index= None)
    
    return lemmas, vocab


# inverted index file for later analysis
def create_iif(lemmas: pd.DataFrame, excerpts: pd.DataFrame):

    lemmas.sort_values(by= "Lemma", inplace= True)
    lemmas.drop_duplicates(subset= ["Word", "POS-tag"], inplace= True) # keep unique entries of words-pos_tag

    iif_df = lemmas.set_index(["Lemma", "Word"], drop= True) # sort by lemmas and words
    iif_df.insert(1, "Excerpt", [[] for i in range(len(iif_df.index))]) 
    print(iif_df.loc["αἴτιος", ["POS-tag", "Excerpt"]])

    iif_df.to_csv("./iif.csv")


def grecy_proiel_trf(text: str):
    nlp = spacy.load("grc_proiel_trf")
    doc = nlp(text)
    lem_df = pd.DataFrame(columns= ["word", "lemma", "pos-tag"])


    for token in doc:
        lem_df = pd.concat([pd.DataFrame(
                [[token.text, token.lemma_, token.pos_]],
                columns=lem_df.columns),
                lem_df], ignore_index=True)

    lem_df.to_csv("./lemmas/grecy_lemmas.csv")
    # print(f'{token.text}\t lemma: {token.lemma_}\t pos:{token.pos_}')
    

# Function to process texts to brong them to an acceptable form
def text_clean_up(text: str):
    # remove punctuations and same value letters
    cleaned = re.sub(r"\,|\.+|\·|\”|\'|\“|\(|\)", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\,|\.+|\·|\”|\'|\“|\(|\)", "σσ", cleaned)

    # for grecy: Handle different accents of apostrophes
    cleaned = re.sub(r"[ʼ’‘′‵ʹʾʿ`´`´ʹ͵ͻͼ’＇՚]", "\u02BC", cleaned)  # gpt prompted, all common variations that could be an apostrophe


    # some transformation for handling unexpected cases of Dotiki klisi
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
    return cleaned


if __name__ == "__main__":
    main()