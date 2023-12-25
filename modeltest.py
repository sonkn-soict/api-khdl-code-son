import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models, util

model = SentenceTransformer('kaggle/working/duong-model-v3')

import ast


# Custom converter function
def convert_to_list(cell):
    return ast.literal_eval(cell)


import pandas as pd

with open(
        "indexed_book_model_v3_first_100k.csv",
        'r', encoding='utf-8') as f:
    df = pd.read_csv(f, converters={'embedding': convert_to_list})

with open("output.csv", 'r', encoding='utf-8') as f:
    book = pd.read_csv(f)

corpus_embeddings = torch.Tensor(df['embedding'])
import json


def search(query, top_k):
    query_embedding = model.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Output of top-5 hits from bi-encoder
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    result_data = []
    for hit in hits[0:top_k]:
        index = hit['corpus_id']
        index_book = df['id_book'][index]
        title_and_author = book['title'][index_book].split('-')
        result_data.append({'title': title_and_author[0],
                            'chapter': df['name_chapter'][hit['corpus_id']],
                            'author': title_and_author[1] if len(title_and_author) > 1 else "None",
                            'url': book['link_book'][index_book],
                            'related_content': df['passages'][hit['corpus_id']].replace("\n", " ").replace("\"", ""),
                            })
    result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
    return result_data