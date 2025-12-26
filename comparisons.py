import pandas as pd
import pickle, spacy, time, os
from os.path import exists
import vocabulary
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Plus, BM25L, BM25Okapi # based on the paper of reference, outperforms simple bm25 in every corpus
#from flair.models import SequenceTagger

def main():
    average = "average_similarity.csv"
    lemmas = "lemmas_match_bm25.csv"
    exact = "exact_match_bm25.csv"

    documentDf = pd.read_csv(vocabulary.sample_file, usecols= ["excerpt"])
    documentDf.dropna(inplace= True)
    # apply text cleanup to each doc and convert to list of cleaned docs
    docList = documentDf["excerpt"].apply(vocabulary.text_clean_up).to_list()
    AG_BERT(docList)
    #bm25_1("lemmas")
    #weighted_value()

# Function to create embeddings with Ancient Greek Bert model of HuggingFace
# Input: list of texts
# Output: list of vectors of the [CLS] tokens of all texts
# The function also creates and saves the dataframes with all embeddings 
# of each text in a pickle format for future usage without the need of running the model again.
def AG_BERT(docList: list):

    os.makedirs("./pickles/text_embeddings", exist_ok= True)
    os.makedirs("./pickles/embeddings_csvs", exist_ok= True)
    path_to_save_dfs_as_pickles = "./pickles/text_embeddings/"
    path_to_save_dfs_as_csvs = "./pickles/embeddings_csvs/"

    # README instructions to load the model
    model_name = "pranaydeeps/ancient-greek-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## tagger = SequenceTagger.load('libr/final-model.pt')
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode

    # model is set, time to tokenize texts
    for id, text in enumerate(docList):
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        #print((inputs.input_ids[0]))

        with torch.no_grad():
            # get the embeddings from bert
            output = model(**inputs)
            #create the dataframe to store the values
            emb_df = pd.DataFrame(columns= ["token_id",
                                            "position",
                                            "token",
                                            "embedding_vec",
                                            "sub-word"])
            
            for i, token in enumerate(inputs.input_ids[0]):
                sub_word = False # if current token has ## in front, it's a subtoken. Mark for later.
                if tokens[i][0] == "#": sub_word = True
#                                 tensor value id -> int    |||   tensor -> list of floats
                emb_df.loc[i] = [int(token), i+1, tokens[i], output.last_hidden_state[0, i].tolist(), sub_word]
            print(emb_df)
            emb_df.to_pickle(path_to_save_dfs_as_pickles + f"text_{id+1}_emb_dataFrame.pkl")
            emb_df.to_csv(path_to_save_dfs_as_csvs + f"text_{id+1}_embeddings.csv", header= True)
            #print(id, "\n", len(output.last_hidden_state[0]))

# inputs.input_ids[0] == list of token ids
# output.last_hidden_state == output tensor for the model, only interested for the first entry
# output.last_hidden_state[0] == list of embeddings for each token
# output.last_hidden_state[0, 0] == embedding for [CLS], sum of meaning in a vector



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
        print("from " + table + ":\n", doc_sims.head(n))
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