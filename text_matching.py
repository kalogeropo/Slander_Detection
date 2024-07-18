from os.path import exists


def exact_matching(doc, candidate_text):
    counter = 0
    for word in candidate_text.split():
        if word in doc.split():
            counter += 1
    return counter / len(doc.split())

def document_comparisons(doc_index, zippled_list, df):
    #print(zippled_list)
    doc = df["excerpt"][doc_index]

    for i, candidate in enumerate(zippled_list):
        if candidate[0] == 0 or candidate[1] == 0:
            continue
        candidate_text = df["excerpt"][candidate[0]]
        exact_matching_score = exact_matching(doc, candidate_text)
        print(f"text with id {doc_index} has exact matching score {exact_matching_score} with doc {candidate[0]}\n\n")
        # with open("test.txt","a") as fd:
        #     fd.write(f"id: {doc_index}, can id: {candidate[0]}, score: {exact_matching_score}\n")
