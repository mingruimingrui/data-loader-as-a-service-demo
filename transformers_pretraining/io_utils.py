import io
from typing import List, Generator


def read_line(file: io.TextIOBase):
    """Read a line from a file. If reached the end, go back to start."""
    line = file.readline()
    if line == '':
        file.seek(0)
        line = file.readline()
    if line.endswith('\n'):
        line = line[:-1]
    return line


def load_texts_endless(filepaths: List[str]) -> Generator[str, None, None]:
    """Load texts continuously from a list of files"""
    files = [
        open(filepath, 'r', encoding='utf-8', newline='\n')
        for filepath in filepaths
    ]
    while True:
        for file in files:
            yield read_line(file)
