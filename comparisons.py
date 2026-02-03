import pandas as pd
import numpy as np
import pickle, spacy, time, os, math, torch, vocabulary, matplotlib
from os.path import exists
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Plus, BM25L, BM25Okapi # based on the paper of reference, outperforms simple bm25 in every corpus
from flair.models import SequenceTagger
from flair.data import Sentence
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    average = "average_similarity.csv"
    lemmas = "lemmas_match_bm25.csv"
    exact = "exact_match_bm25.csv"

    documentDf = pd.read_csv(vocabulary.sample_file, usecols= ["excerpt"])
    documentDf.dropna(inplace= True)
    # apply text cleanup to each doc and convert to list of cleaned docs
    docList = documentDf["excerpt"].apply(vocabulary.text_clean_up).to_list()

    #lol = AG_BERT(docList)
    #compare_synonyms()
    plot_similarities(pd.read_csv("./results/lemmas_match_bm25.csv",  index_col=0), "lemmas_match_bm25")
    #print(morph_tagger(docList= docList)[1])

    # print(AG_BERT(docList)[0])
    #bm25_1("lemmas")
    #weighted_value() 



def morph_tagger(docList: list):
    # Function to load the morphological tagger and parse all
    # texts with it to generate the tags. Then use them to create 
    # ngrams, probably trigrams, and create frequency vectors for each text.
    # Compare, plot, store and return the results 

    ###### UTILITY VARIABLES ###########
    morphological_mode = "pos_only" # one place to change the name of files based on what morph pos we get

    # hardcoded the path to the model to avoid any issues at last minute :((
    os.chdir("/home/rabanis/Desktop/ceid/diploma/model/Ancient-Greek-BERT/SuperPeitho-FLAIR-v2")
    tagger = SequenceTagger.load("best-model.pt")   # Load the tagger
    print("\n\n\n") # clearer print output

    ids = []
    morph_triplets = []

    ## Get tag triplets for all texts and append them to the list
    for id, text in enumerate(docList): #get each text
        s = Sentence(text)
        tagger.predict(s) #generate tags
        s_tags = [simplify(token.get_label( "pos" ).value ) for token in s] # list to hold the simplified tags in order
        tags = morph_profile(s_tags)    # create the ngrams

        ids.append(id+1) # keep the id starting from 1 to be inline with the rest of the results
        morph_triplets.append(tags)

    # Vectorize all text triplet counts with Scikit learn for speed
    vec = DictVectorizer()
    morph_matrix = vec.fit_transform(morph_triplets)

    #change cwd due to the change earlier for the model loading
    os.chdir("/home/rabanis/Desktop/ceid/diploma/Slander_Detection")
    # Compare with cosine similarity
    Sim = cosine_similarity(morph_matrix)
    sim_df = pd.DataFrame(Sim, index= ids, columns= ids).map(lambda x: round(x, 4))
    sim_df.to_csv("./results/morph_similarities_" + morphological_mode + ".csv")

    # plot and return results
    plot_similarities(sim_df, "morph_" + morphological_mode)
    return sim_df, morph_triplets
            

###### HELPFUL FUNCTIONS ####################
def ngrams(seq, n=3):
    # return the trigram for n=3 of the input sequence
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def morph_profile(tags):
    # get the simplified tags and return a counted set of their ngrams -> vector representation
    grams = ngrams(tags, 3)
    return Counter(grams)

def simplify(tag):
    # simplify the pos tags using only up to 4 of the 9 parts.
    pos = tag[0]
    number = tag[1]
    gender = tag[5]
    case = tag[6]
    return f"{pos}"#{number}{gender}{case}"

def plot_similarities(sim_df, name):

    pth = "./figures/"+name

    plt.figure(figsize=(10,8))

    sns.heatmap(
        sim_df,
        xticklabels=sim_df.index,
        yticklabels=sim_df.columns,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        linecolor="black",
        cbar_kws={"label": "Σημασιολογική Ομοιότητα"}
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(pth + ".png")

######## END OF HELPFUL FUNCTIONS #########

def cls_token_dataframe():
    cls_df = pd.DataFrame()

    for i in range(1, 27):
        cls_df[i] = compare_cls(i)["score"]
    
    cls_df.to_csv("./results/CLS_token_scores.csv")

# get the embedding of the CLS token of the text and compare it to all other CLS tokens
# returns: dataframe with id of text and cosine similarity between cls token embeddings.
def compare_cls(no_of_txt: int):
    no_of_files = 26    # hard coded iterations for easy of use, change here for more text files
    results = pd.DataFrame(columns= ["id", "score"])
    file_path = f"./pickles/text_embeddings/text_{no_of_txt}_emb_dataFrame.pkl"

    text_df = pd.read_pickle(file_path) # load the pickled dataframe with embeddings
    my_embdng = np.array(text_df.at[0, "embedding_vec"]) # make lists to np arrays for better perfomance

    for i in range(1, no_of_files + 1):
        if no_of_txt == i:  # if condition, then no need to load and 0compute, absolute similarity
            results.loc[i] = [i, 1.0] 
            continue

        file_path = f"./pickles/text_embeddings/text_{i}_emb_dataFrame.pkl"
        current_text_df = pd.read_pickle(file_path)
        curr_embdng = np.array(current_text_df.at[0, "embedding_vec"])

        cos_sim = np.dot(curr_embdng, my_embdng) / (np.linalg.norm(my_embdng) * np.linalg.norm(curr_embdng))
        results.loc[i] = [i, float(cos_sim)]
        

    return results


def AG_BERT(docList: list):
# Function to create embeddings with Ancient Greek Bert model of HuggingFace
# Input: list of texts
# Output: list of vectors of the [CLS] tokens of all texts
# The function also creates and saves the dataframes with all embeddings 
# of each text in a pickle format for future usage without the need of running the model again.
    os.makedirs("./pickles/text_embeddings", exist_ok= True)
    os.makedirs("./pickles/embeddings_csvs", exist_ok= True)
    path_to_save_dfs_as_pickles = "./pickles/text_embeddings/"
    path_to_save_dfs_as_csvs = "./pickles/embeddings_csvs/"

    cls_emb_list = []
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

                # for normalization and cosine similarity purposes later on
                embdng = np.array(output.last_hidden_state[0, i].tolist())
                embdng = embdng / np.linalg.norm(embdng) # normalize
                print(tokens[i], i)

                emb_df.loc[i] = [int(token), i+1, tokens[i], embdng.tolist(), sub_word]
                # if cls token
                if i == 0:
                    cls_emb_list.append(embdng.tolist())

            emb_df.to_pickle(path_to_save_dfs_as_pickles + f"text_{id+1}_emb_dataFrame.pkl")
            emb_df.to_csv(path_to_save_dfs_as_csvs + f"text_{id+1}_embeddings.csv", header= True)

    return(cls_emb_list)

    # inputs.input_ids[0] == list of token ids
    # output.last_hidden_state == output tensor for the model, only interested for the first entry
    # output.last_hidden_state[0] == list of embeddings for each token
    # output.last_hidden_state[0, 0] == embedding for [CLS], sum of meaning in a vector


def concat_embeddings(no_of_txt):
# function to get a series of tokens and concatenate them into one if part of subword
# Also keeps track of the position of the original token. Returns dataframe with original tokens and average embeddings
    file_path = f"./pickles/text_embeddings/text_{no_of_txt}_emb_dataFrame.pkl"

    text_df = pd.read_pickle(file_path)
    text_df["embedding_vec"] = text_df["embedding_vec"].map(lambda x : np.array(x)) # nd.arrays for better perfomance

    mstr_tkn = "" # string to hold the whole word
    curr_tkn_pos_index = 0 # track the position of the word in text
    no_of_subwords = 1

    for i in range(len(text_df.index)):
        if text_df.at[i, "sub-word"] == False:
            mstr_tkn = text_df.at[i, "token"] # track the word if splitted
            text_df.at[i, "position"] = curr_tkn_pos_index # save position in text
            curr_tkn_pos_index = curr_tkn_pos_index + 1 # append position in text for next word
            no_of_subwords = 1 # reset tracker of subwords

        else:
            ind = i - no_of_subwords # index of the master token
            no_of_subwords = no_of_subwords + 1
            mstr_tkn = mstr_tkn + text_df.at[i, "token"][2:]
            text_df.at[ind, "token"] = mstr_tkn
            text_df.at[ind, "embedding_vec"] = text_df.at[ind, "embedding_vec"] + text_df.at[i, "embedding_vec"]

            # if next entry is a master token, get the average of the embeddings of all subwords and 
            # save it in the master tokens embedding.
            if text_df.at[i+1, "sub-word"] == False:
                text_df.at[ind, "embedding_vec"] = text_df.at[ind, "embedding_vec"] / no_of_subwords
            
    final_df = text_df[text_df["sub-word"] == False]
    return(final_df)

def synonyms(no_of_text: int, threshold = 0.7):
#   compare the embeddings of all words in a text with all words from other texts
#   and keep only the words that matched over a certain percentile.

    # store the concatenated embeddings
    my_text_df = concat_embeddings(no_of_txt=no_of_text)
    results = []

    # list of stopword tokens to skip while comparing. They get
    #extremely similar scores but serve no purpose in the comparisons
    stopwords = ["[SEP]", "[CLS]", "[UNK]", "και", "τις",
                "γαρ", "το", "τα", "κατα", "του", "εκ",
                "αν", "δε", "τον", "την", "τοις",
                "δη", "μεν", "τω", "τους", "ο", "η"]

    for i in range(1, 27):
        if i == no_of_text: #skip same text, no reason for that
            continue

        testing_df = concat_embeddings(i)

        for j in my_text_df.index:
            word = my_text_df.at[j, "token"]
            pos = my_text_df.at[j, "position"]

            if word in stopwords:
                continue

            # current text, embdng of the target word j
            emb1 = my_text_df.at[j, "embedding_vec"]

            for k in testing_df.index:
                #check the testing texts' words one by one
                word_k = testing_df.at[k, "token"]
                pos_k = testing_df.at[k, "position"]
                emb2 = testing_df.at[k, "embedding_vec"]

                # cosine similarity (for normalized embeddings, dot is enough)
                sim = np.dot(emb1, emb2)

                if sim >= threshold:
                    results.append({
                        "source_word": word,
                        "source_w_pos": pos,
                        "target_word": word_k,
                        "target_w_pos": pos_k,
                        "target_text": i,
                        "similarity": float(sim)
                    })

    if len(results) != 0:
        synonym_df = pd.DataFrame(results) 
        return synonym_df.sort_values(axis= 0, by= "similarity", ascending= False)
    else:
        return 0

def compare_synonyms():
    sim_threshold = 0.65
    for i in range(1, 27):
        i_synonyms = synonyms(no_of_text= i, threshold= sim_threshold)
        print(i, "\t", i_synonyms)

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