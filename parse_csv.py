import csv
from typing import List, Dict, Iterable

from models import RawMessage


def read_csv(filename: str) -> Iterable[RawMessage]:
    with open(filename, "r", encoding='utf8') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            yield RawMessage(**row)



