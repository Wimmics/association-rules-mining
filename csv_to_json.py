
import argparse, sys
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--datafile', help='Path to csv file')

args = parser.parse_args()

rules_df = pd.read_csv(args.datafile, sep=';')
print(rules_df)
rules_df['antecedents'] = [list(eval(x)) for x in rules_df['antecedents']]
rules_df['consequents'] = [list(eval(x)) for x in rules_df['consequents']]
rules_df['source'] = [list(eval(x)) for x in rules_df['source']]
rules_df['target'] = [list(eval(x)) for x in rules_df['target']]
print(rules_df)

filename = args.datafile.replace('.csv', '') + '.json'
rules_df.to_json(path_or_buf=filename, orient='records')