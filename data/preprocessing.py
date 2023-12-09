# pip install pandas
# pip install scikit-learn
# pip install emoji

# pip install torch

# pip install setuptools wheel 'spacy[cuda12x]'

# !python -m spacy download pt_core_news_sm

# Baixar o recurso necessário para a tokenização (pode precisar ser executado apenas uma vez)
# nltk.download('punkt')

# Baixar o recurso necessário para o pré-processamento
# nltk.download('stopwords')

import os
import sys
import argparse

import pandas as pd

import re
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Função para remover tweets vazios
def clean_empty(df):
  empty = df['tweet'].apply(len).apply(lambda x: x > 0)
  print(f"Quantidade de tweets vazios encontrados: {empty.value_counts().get(False, 0)}")

  # Removendo linhas com quantidade de caracteres igual a 0
  return df[empty]

# Função Data Cleaning
def data_cleaning(tweet):
    # Remoção de emoji
    tweet = emoji.demojize(tweet, delimiters=(' :', ': '), language='pt')
    tweet = re.sub(r'(?<=\s:)(.*?)(?=:|\s|$)', '', tweet)

    tweet = re.sub(r'\r|\n', ' ', tweet.lower())  # Replace newline and carriage return with space, and convert to lowercase
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)  # Remove links and mentions
    tweet = re.sub(r'rt', '', tweet)
    tweet = re.sub(r"\s\s+", " ", tweet)

    tweet = re.sub(r':', '', tweet) # Remove ":"
    tweet = re.sub(r';', '', tweet) # Remove ";"

    words = tweet.split()
    return tweet if len(words) > 3 else ""

# Remoção de stop-words
def remove_stopwords(text):
  tokens = word_tokenize(text.lower())
  stop_words = set(stopwords.words('portuguese'))  # Escolha o idioma apropriado
  tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
  return ' '.join(tokens)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Processamento de dados")
  parser.add_argument("--l", type=str, help="Caminho para o arquivo less (CSV)")
  parser.add_argument("--m", type=str, help="Caminho para o arquivo more (CSV)")
  parser.add_argument("--save_l", type=str, help="Caminho de saída para less (CSV)")
  parser.add_argument("--save_m", type=str, help="Caminho de saída para more (CSV)")

  args = parser.parse_args()

  if not all([args.l, args.m, args.save_l, args.save_m]):
      print("Usage: python3 preprocessing.py --l <caminho_less.csv> --m <caminho_more.csv> --save_l <caminho_saida_less.csv> --save_m <caminho_saida_more.csv>")
      sys.exit(1)

  input_file_less = args.l
  input_file_more = args.m
  output_file_less = args.save_l
  output_file_more = args.save_m

  folder = './'
  os.chdir(folder)

  dtype = {
     "id": str,
     "author_id": str,
     "tweet": str,
     "created_at": str,
     "retweet_count": int
  }

  less_df = pd.read_csv(input_file_less, delimiter=';', skiprows=0, dtype=dtype)

  more_df = pd.read_csv(input_file_more, delimiter=';', skiprows=0, dtype=dtype)

  print("LESS")
  print(f"Tamanho máximo de tweet no DataFrame: {less_df['tweet'].str.len().max()}")
  print(f"Quantidade de usuários diferentes no DataFrame: {less_df['author_id'].nunique()}")

  print("MORE")
  print(f"Tamanho máximo de tweet no DataFrame: {more_df['tweet'].str.len().max()}")
  print(f"Quantidade de usuários diferentes no DataFrame: {more_df['author_id'].nunique()}")

  # Aplicando limpeza no less_df
  # less_df['original_tweet'] = less_df['tweet']
  less_df['tweet'] = [data_cleaning(tweet) for tweet in less_df['tweet']]
  less_df = clean_empty(less_df)

  # Aplicando limpeza no more_df
  # more_df['original_tweet'] = more_df['tweet']
  more_df['tweet'] = [data_cleaning(tweet) for tweet in more_df['tweet']]
  more_df = clean_empty(more_df)

  # Aplicando remoção de stop-words nos dados
  more_df['tweet'].apply(remove_stopwords)
  less_df['tweet'].apply(remove_stopwords)

  # Removendo tweets vazios após a remoção de stop-words
  more_df = clean_empty(more_df)
  less_df = clean_empty(less_df)

  # Less Toxics
  print('Less')
  print(less_df.head())

  # More Toxics
  print('More')
  print(more_df.head())

  less_df.to_csv(output_file_less, sep=';')
  more_df.to_csv(output_file_more, sep=';')