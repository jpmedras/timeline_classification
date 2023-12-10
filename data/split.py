import os
import sys
import argparse

import pandas as pd

def split_csv_by_column(input_file, output_folder, column_name, index=0):
  df = pd.read_csv(input_file, sep=";", encoding='utf-8', dtype={"id": str, "author_id": str, "tweet": str, "retweeet_count": int})

  os.makedirs(output_folder, exist_ok=True)

  masked_id = {}

  for value in df[column_name].unique():
    subset_df = df[df[column_name] == value]

    masked_id[value] = len(masked_id)

    output_file = os.path.join(output_folder, f"{column_name}_{masked_id[value] + index}.csv")

    subset_df.to_csv(output_file, index=False, sep=';', encoding='utf-8')

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Processamento de dados")
  parser.add_argument("--l", type=str, help="Caminho para o arquivo less (CSV)")
  parser.add_argument("--m", type=str, help="Caminho para o arquivo more (CSV)")

  args = parser.parse_args()

  if not all([args.l, args.m]):
      print("Usage: python3 preprocessing.py --l <caminho_less.csv> --m <caminho_more.csv>")
      sys.exit(1)

  input_file_less = args.l
  input_file_more = args.m
  
  split_csv_by_column(input_file_less, 'less', 'author_id', index=0)
  index = len(os.listdir('less'))
  split_csv_by_column(input_file_more, 'more', 'author_id', index=index)