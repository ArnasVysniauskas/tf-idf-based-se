
from collections import defaultdict
from dataclasses import dataclass, field
from copy import deepcopy
from functools import wraps
import math
import time

Term = str
Token = str
Frequency = int
Importance = float
Similarity = float

TIMEIT = -1

def timeit_wrapper(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        if TIMEIT == -1:
            return f(*args)

        start = time.perf_counter()
        for _ in range(TIMEIT):
            result = f(*args)
        end = time.perf_counter()
        print(f"It took {(end-start)/TIMEIT} seconds on average to perform one search in {TIMEIT} samples")

        return result
    
    return wrapper

class Filter:
    def __init__(self, gram: int) -> None:
        self.gram: int = gram    

    def apply(term: list[Term]) -> list[Term]:
        raise NotImplementedError

class LowerCaseFilter(Filter):
    def apply(terms: list[Term]) -> list[Term]:
        return [term.lower() for term in terms]

class Tokenizer:

    def __init__(self, separators: list[str], filters: list[Filter], gram: int) -> None:
        self.separators: list[str] = separators
        self.filters: list[Filter] = filters
        self.gram: int = gram
    
    def _separate(self, tokens: list[Token]) -> list[Token]:
        for separator in self.separators:
            new_tokens: list[Token] = []
            for token in tokens:
                new_tokens += token.split(separator)
            tokens = deepcopy(new_tokens)
        return tokens

    def _filter(self, tokens: list[Token]) -> list[Token]:
        for filter in self.filters:
            tokens = filter.apply(tokens)
        return tokens

    def _gramiffy(self, tokens: list[Token]) -> list[Token]:
        if self.gram == 1:
            return tokens

        new_tokens: list[Token] = []
        for token in tokens:
            if len(token) <= self.gram:
                new_tokens.append(token)
            else:
                new_tokens += self._spliter(token)
        return new_tokens

    def _spliter(self, token: Token) -> list[Token]:
        if len(token) == self.gram:
            return [token]
        return [token[:self.gram]] + self._spliter(token[1:])
    
    def apply(self, term: Term) -> list[Token]:
        tokens: list[Token] = [term]

        tokens = self._separate(tokens)
        tokens = self._filter(tokens)
        tokens = self._gramiffy(tokens)

        return tokens


@dataclass(init=True)
class Entry:
    reference: dict[Term, Importance] = field(default_factory = lambda: {})
    count: Frequency = 0

@dataclass(init=True, frozen=True)
class Index:
    tokenizer: Tokenizer
    data: list[Term]
    index: dict[Token, Entry]

    @classmethod
    def index_data(cls, data: list[Term], tokenizer: Tokenizer) -> "Index":

        data_index: dict[Term, dict[Token, Frequency]] = {}
        tokens_index: dict[Token, Entry] = defaultdict(Entry)

        # Count all the tokens
        for term in data:
            data_index[term] = defaultdict(Frequency)
            for token in tokenizer.apply(term):
                data_index[term][token] += 1
                tokens_index[token].count += 1

        for term in data:
            term_weigth = cls._term_weight(tokens_index, data_index[term])
            # Calculate token Importance in term, and save reference to the term in token Entry
            for token, frequency in data_index[term].items():
                token_importance: Importance = math.sqrt(frequency) / (term_weigth * tokens_index[token].count)
                tokens_index[token].reference[term] = token_importance

        return Index(
            tokenizer=tokenizer,
            data=data,
            index=tokens_index,
        )
    
    @classmethod
    def _term_weight(cls, tokens_index: dict[Token, Entry], term_tokens_count: dict[Token, Frequency]) -> float:
        # Root of sum of squared inverse frequencies of tokens
        return math.sqrt(
            sum(
                [tokens_index[token].count**(-2) * frequency for token, frequency in term_tokens_count.items()]
            )
        )
    
    def html(self) -> str:
        return (
            f"<p>indexed_data_length: {len(self.data)}<br>"
            f"separators: {self.tokenizer.separators}<br>"
            f"gram: {self.tokenizer.gram}<br></p>"
        )
    
    def outside_term_weight(self, term_tokens_count: dict[Token, Frequency]) -> float:
        # same as _term_weight, but considers that token might not exist in the index
        # Used for search_terms after the index was constructed
        sum: int = 0

        for token, frequency in term_tokens_count.items():
            try:
                sum += self.index[token].count**(-2) * frequency
            except:
                sum += 1

        return math.sqrt(sum)

    def get_token_frequency(self, token: Token) -> Frequency:
        try:
            return self.index[token].count
        except KeyError:
            return 0
    
    @timeit_wrapper
    def find_matches_for(self, term: Term, depth: int = -1) -> dict[Term, Similarity]:
        tokens = self.tokenizer.apply(term)

        if depth > len(tokens) or depth == -1:
            depth = len(tokens)

        # Count tokens and get term_weight
        tokens_index: dict[Token, Frequency] = defaultdict(Frequency)
        for token in tokens:
            tokens_index[token] += 1
        term_weight = self.outside_term_weight(tokens_index)

        # Sort tokens in descending importance
        tokens.sort(key = lambda key: self.get_token_frequency(key))
        matches: dict[Term, Similarity] = defaultdict(Similarity)

        for token in tokens[:depth]:
            for term_match, indexed_weight in self.index[token].reference.items():
                matches[term_match] += indexed_weight / (term_weight * self.index[token].count)

        return matches
    
    def find_best_match_for(self, term: Term, count: int = 1, depth: int=-1) -> list[tuple[Term, Similarity]]:
        matches = self.find_matches_for(term, depth)

        # Sort match score descending
        ordered_term_matches = sorted(matches.keys(), key=lambda key: matches[key], reverse=True)

        return [(match, matches[match]) for match in ordered_term_matches[:count]]

def get_data_from_file(file_name: str) -> list[Term]:
    data = []
    with open(file_name, "r") as f:
        csv_headers = f.readline()
        for line in f:
            entry = line[:-1]
            if entry not in data:
                data.append(line[:-1])
    return data

def main(source: str, gram: int, depth: int, timeit: int):
    
    global TIMEIT
    TIMEIT = timeit

    data: list[Term] = get_data_from_file(source)

    tokenizer = Tokenizer(["_", " ", "."], [LowerCaseFilter], gram)
    index = Index.index_data(data, tokenizer)

    term = input("Term to search for [ENTER -> close]: ")
    while term != "":

        for match in index.find_best_match_for(term, count=5, depth=depth):
            print(match)
        term = input("Term to search for [ENTER -> close]: ")

if __name__ == "__main__":
    main(source = "sample_list.csv", gram = 1, depth = 5, timeit = -1)