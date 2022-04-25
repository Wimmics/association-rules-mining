
## Installation & Dependencies 
This script requires Python 3.7, which is a strict requirement for stellargraph (used in the script).
To install all dependencies `pip install -r requirements.txt`

## Running on Agrovoc graph from the ISSA Project
To extract rules from the agrovoc graph of the issa dataset :

./rules_generator.py --endpoint 'issa' --graph 'agrovoc' --lang 'en' 
