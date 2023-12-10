from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads

import sys
import argparse 

import torch

import pandas as pd

class Embeddings():
    def __init__(self, model, tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForPreTraining.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=False)
    
    def generateEmbeddings(self, sample, padding=False): #semple is a dictinary(key: author_id, value: list of tweets)
        embeddings = []
        author_ids = []

        for author_id, tweet in sample.items():
            author_ids.append(author_id)
            segment_embeddings = []

            for input in tweet:
                input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.device)

                with torch.no_grad():
                    outs = self.model(input_ids)
                    encoded = outs[0][0,1:-1]

                segment_embeddings.append(encoded.mean(dim=0))

            if(segment_embeddings):
                segment_embeddings = torch.stack(segment_embeddings)
                embeddings.append(torch.mean(segment_embeddings, dim=0))
            

        return torch.stack(author_ids), torch.stack(embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Geração de embeddings usando BERT")
    parser.add_argument("--f", type=str, help="Caminho para o arquivo CSV")
    parser.add_argument("--e", type=str, help="Caminho para salvar o arquivo e.pt")
    parser.add_argument("--a", type=str, help="Caminho para salvar o arquivo a.pt")

    args = parser.parse_args()

    if not all([args.f, args.e, args.a]):
        print("Usage: python3 embeddings.py --f <file.csv> --e <e.pt> --a <a.pt> ")
        sys.exit(1)

    file_csv = args.f
    embeddings_pt = args.e
    authors_pt = args.a

    # Função para criar dict
    def criar_dict(df):

        tweets_dict = {}
        # Itera sobre as linhas do DataFrame
        for index, row in df.iterrows():
            author_id = row['author_id']
            tweet = row['tweet']

            # Verifica se o author_id já está no dicionário
            if author_id in tweets_dict:
                # Adiciona o tweet ao vetor de tweets do autor
                tweets_dict[author_id].append(tweet)
            else:
                # Cria um novo vetor de tweets para o autor
                tweets_dict[author_id] = [tweet]

        return tweets_dict
    
    # Função para remover duplicados e selecionar amostra
    def remover_duplicados(tweets_por_autor, max_tweets = None):
        # Itera sobre os tweets agrupados por autor
        for author_id, tweets in tweets_por_autor.items():
            # Remove tweets duplicados
            tweets_por_autor[author_id] = list(set(tweets))

            # Limita o vetor
            if max_tweets:
                tweets_por_autor[author_id] = tweets_por_autor[author_id][-max_tweets:]


        return tweets_por_autor

    model = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = 'neuralmind/bert-base-portuguese-cased'

    model = Embeddings(model, tokenizer)

    dtype = {
        "id": str,
        "author_id": str,
        "tweet": str,
        "created_at": str,
        "retweet_count": int
    }

    df_tweets = pd.read_csv(file_csv, delimiter=';', dtype=dtype)

    # Gerar dicionário
    tweets_dict = remover_duplicados(criar_dict(df_tweets))

    authors_ids, embeddings = model.generateEmbeddings(tweets_dict, True)

    # Salvar objetos
    torch.save(torch.stack(authors_ids), embeddings_pt)
    torch.save(torch.stack(embeddings), authors_pt)



