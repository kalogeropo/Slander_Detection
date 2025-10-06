from os.path import exists

from pandas import DataFrame

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from parsing_utils import parse_file

from matplotlib import pyplot as plt

from text_matching import document_comparisons

path = "./sample_collection"

if __name__ == '__main__':
    # Parse the uploaded file with relevant ids
    parsed_data_with_relevant_ids = parse_file(path)
    # print(parsed_data_with_relevant_ids)
    df = DataFrame.from_dict(parsed_data_with_relevant_ids)
    sample_file = './sample_col.csv'
    if not exists(sample_file):
        df.to_csv(sample_file, index=False)
    # print(df.head(10))
    # print(df['excerpt'])
    # print(df['author'])

    vectorizer = TfidfVectorizer()  # min_df = mia emfanisi
    # vectorizer = TfidfVectorizer(min_df=0.1) #stopwords

    X = vectorizer.fit_transform(df["excerpt"])
    document_df = DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # print(vectorizer.get_feature_names_out())
    print(document_df.head(15))
    exit()

    pca = PCA(n_components=2)
    df2d = DataFrame(pca.fit_transform(X.toarray()), columns=list('xy'))

    # Plot Data Visualization (Matplotlib)
    df2d.plot(kind='scatter', x='x', y='y')
    # plt.show()

    for i, doc in enumerate(X):
        cosineSimilarities = cosine_similarity(doc, X).flatten()
        # REMEMBER doc indices starts at 1, 0 is a null doc!!!
        sort_index = [i for i, x in sorted(enumerate(cosineSimilarities), reverse=True, key=lambda x: x[1])]
        sorted_doc_sim_list = list(zip(sort_index, sorted(cosineSimilarities, reverse=True)))
        document_comparisons(i, sorted_doc_sim_list, df)
        # with open("similarities.csv","a") as fd:
        #     fd.write(f"doc id: {i}, Sim: {' '.join(map(str,cosineSimilarities))}\n")