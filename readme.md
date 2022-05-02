
## Installation & Dependencies 
This script requires Python 3.7, which is a strict requirement for stellargraph (used in the script).
To install all dependencies `pip install -r requirements.txt`

## Using the algorithm

usage: rules_generator.py [-h] [--endpoint ENDPOINT] [--input INPUT]
                          [--graph GRAPH] [--lang LANG] [--filename FILENAME]
                          [--conf CONF] [--int INT] [--occurrence OCCURRENCE]

optional arguments:

  -h, --help            show this help message and exit
  
  --endpoint ENDPOINT   The endpoint from where retrieve the data (identified
                        through codes: issa, covid).
  
  --input INPUT         If available, path to the file containing the input
                        data
  
  --graph GRAPH         In case there is a graph where to get the data from in
                        the endpoint, provide (valid for issa: agrovoc,
                        geonames, wikidata, dbpedia)
  
  --lang LANG           The language of the labels
  
  --filename FILENAME   The output file name. If not provided, it will be
                        automatically generated based on the input
                        information.
  
  --conf CONF           Minimum confidence of rules. Default is .7, rules with
                        less than x confidence are filtered out.
  
  --int INT             Minimum interestingness (serendipity, rarity) of
                        rules. Default is .3, rules with less than x
                        interestingess are filtered out.
  
  --occurrence OCCURRENCE
                        Keep only terms co-occurring more than x times.
                        Default is 5

After the first execution, the algorithm saves the input data into a csv file. To run the algorithm again using this file as input data instead of querying the endpoint, give the path as --filename.

The arguments --endpoint, --graph, and --lang are used to retrieve and custom the query from the queries.json file to retrieve the input data. To include a new query or a new SPARQL endpoint, modify the queries.json file accordingly.

# Use cases
Mining association rules from the agrovoc graph of the ISSA dataset:

./rules_generator.py --endpoint 'issa' --graph 'agrovoc' --lang 'en' 

Mining association rules from the CovidOnTheWeb dataset:

./rules_generator.py --endpoint 'covid' --graph 'agrovoc' --lang 'en' 

## Cite this work

When using this algorithm in a publication, please cite this paper:

Lucie Cadorel, Andrea G. B. Tettamanzi. Mining RDF Data of COVID-19 Scientific Literature for Interesting Association Rules. WI-IAT'20 - IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology, Dec 2020, Melbourne, Australia. [hal-03084029](https://hal.inria.fr/hal-03084029)
