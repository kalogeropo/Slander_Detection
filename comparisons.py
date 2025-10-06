import pandas as pd
from os.path import exists
import vocabulary
from rank_bm25 import BM25Plus, BM25L, BM25Okapi # based on the paper of reference, outperforms simple bm25 in every corpus

def main():
    bm25_1()

def bm25_1():
    documentDf = pd.read_csv(vocabulary.sample_file, usecols= ["excerpt"])
    documentDf.dropna(inplace= True)
    # apply text cleanup to each doc and convert to list od cleaned docs
    docList = documentDf["excerpt"].apply(vocabulary.text_clean_up).to_list()

    tokenized_corpus = [doc.split(" ") for doc in docList]
    bm25l = BM25L(tokenized_corpus)
    scoreDF = pd.DataFrame(index= range(1,27) ,columns= range(1, 27))

    print(docList[1])
    print(bm25l.get_scores(docList[0]))
    
    index = 0
    for doc in docList:

        scoreDF[index] = bm25l.get_scores(doc.split())
        scoreDF[index] = scoreDF[index] / scoreDF[index].max(0) # normalize
        index = index + 1
    scoreDF = scoreDF.map(lambda x: round(x, 3))
    scoreDF.to_csv("results/exact_match_bm25.csv")

if __name__ == "__main__":
    main()