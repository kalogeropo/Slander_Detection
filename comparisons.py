import pandas as pd
from os.path import exists
import vocabulary
from rank_bm25 import BM25Plus # based on the paper of reference, outperforms simple bm25 in every corpus

def main():
    bm25_1()

def bm25_1():
    documentDf = pd.read_csv(vocabulary.sample_file, usecols= ["excerpt"])
    documentDf.dropna(inplace= True)
    # apply text cleanup to each doc and convert to list od cleaned docs
    docList = documentDf["excerpt"].apply(vocabulary.text_clean_up).to_list()

    print(docList)


if __name__ == "__main__":
    main()