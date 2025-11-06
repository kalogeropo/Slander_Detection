import pandas as pd
import pickle, spacy, time, os
from os.path import exists
import vocabulary
from rank_bm25 import BM25Plus, BM25L, BM25Okapi # based on the paper of reference, outperforms simple bm25 in every corpus

def main():
    
    get_top_n_results("average_similarity.csv", 3, 5)
    #bm25_1("lemmas")
    #weighted_value()



def bm25_1(mode: str):
# we can pickle the splitted documents and append to it each time we want to add
# a new document in the corpus but the training has to be redone each time with 
# this bm25 library, we can't use existing vocabulary.


    documentDf = pd.read_csv(vocabulary.sample_file, usecols= ["excerpt"])
    documentDf.dropna(inplace= True)
    # apply text cleanup to each doc and convert to list od cleaned docs
    docList = documentDf["excerpt"].apply(vocabulary.text_clean_up).to_list()

    if mode == "lemmas":
        pickle_path = "pickles/lemmatized_list.pkl"
        os.makedirs("pickles", exist_ok= True)

        # if not created, create the lemma pickle
        if not os.path.exists(pickle_path):

            for index, doc in enumerate(docList):
                print(doc)
                st = time.monotonic()
                docList[index] = grecy_proiel_trf_lemmatize(doc)
                print(docList[index])
                print("doc # ", index, ": ", time.monotonic() - st)
            
            with open(pickle_path, "wb") as f:
                pickle.dump(docList, f)

        else:
            with open(pickle_path, "rb") as f:
                docList = pickle.load(f)


    tokenized_corpus = [doc.split(" ") for doc in docList]
    bm25plus = BM25Plus(tokenized_corpus)
    scoreDF = pd.DataFrame(index= range(1,27) ,columns= range(1, 27))

    print(bm25plus.get_scores(docList[3].split()))
    
    index = 1
    for doc in docList:

        scoreDF[index] = bm25plus.get_scores(doc.split())
        # normalize
        min = scoreDF[index].min(0)
        scoreDF[index] = (scoreDF[index] - min)
        max = scoreDF[index].max(0)
        scoreDF[index] = scoreDF[index] / max 
        # increase index 
        index = index + 1

    scoreDF = scoreDF.map(lambda x: round(x, 3))
    
    if mode == "exact":
        scoreDF.to_csv("results/exact_match_bm25.csv")
    elif mode == "lemmas":
        scoreDF.to_csv("results/lemmas_match_bm25.csv")

def get_top_n_results(table: str, text_id: int, n: int):
    table = "results/" + table
    if exists(table):
        doc_sims = pd.read_csv(table, header= 0, index_col= 0, usecols= [0, text_id])
        doc_sims.sort_values(by= str(text_id), ascending= False, inplace= True)
        print(doc_sims.head(n))
        return

    else:
        print("Not correct input table name!!")
        return


# create a better visualization for the similarity of each text
def weighted_value():
    lemmatized_similarity_df = pd.read_csv("results/lemmas_match_bm25.csv", header= 0, index_col= 0)
    exact_similarity_df = pd.read_csv("results/exact_match_bm25.csv", header= 0, index_col= 0)

    average_similarity_df = (lemmatized_similarity_df + exact_similarity_df) / 2
    average_similarity_df = average_similarity_df.map(lambda x: round(x, 3))
    average_similarity_df.to_csv("results/average_similarity.csv")

def grecy_proiel_trf_lemmatize(text: str):
    nlp = spacy.load("grc_proiel_trf")
    doc = nlp(text)
    lemmatized_text = ""
    for token in doc:
        lemmatized_text = lemmatized_text + token.lemma_ + " "
    return lemmatized_text

if __name__ == "__main__":
    main()