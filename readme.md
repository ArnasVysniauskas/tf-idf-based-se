# Light tf-idf based fuzzy-matching search engine

Light tf-idf based fuzzy-matching search engine with no external dependencies for python 3.10

1. Start python environment:
    `pyenv install 3.10.6`
    `pyenv use 3.10.6`
2. Install dependencies:
    `poetry install`
3. Run the tool with your data:
    `python fuzzy_tests/fuzzy_v1.py --source link_items.csv --gram 1 --depth -1`

    [source] - .csv file containing your data in main project directory
    [gram]   - (default = 1) how finelly slice (tokenize) the word
        gram = 1: "amazon" -> ["amazon"] (does nothing)
        gram = 2: "amazon" -> ["am", "ma", "az", "zo", "on"]
        gram = 3: "amazon" -> ["ama", "maz", "azo", "zon"]
        ...
    [depth]  - (default = -1) How deeply searches for match. Looks for most important tokens first
        depth = -1: infinite depth, searches for all tokens, most precise result
        depth =  1: takes one token, and checks for the matches, 
                    ignores similarity increase from other less important tokens
        depth =  2: takes two tokens, and checks for the matches,
                    ignores similarity increase from other less important tokens
        ...

# Hard-coded config:

1. String into tokens separators: ["_", " ", "."]
2. Lowercase filter