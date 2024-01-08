from typing import List, Iterator

import pandas as pd


class QAData:

    def __init__(self, dataSource: str | pd.DataFrame | List[dict]):
        if isinstance(dataSource, str):
            self.data = pd.read_csv(dataSource, sep='\t')
        elif isinstance(dataSource, pd.DataFrame):
            self.data = dataSource
        elif isinstance(dataSource, list):
            self.data = pd.DataFrame.from_records(dataSource)
        else:
            raise NotImplementedError(f"Data source type {type(dataSource)} not supported.")

        self.data.rename(columns={"wikipedia_answer": "answer"})

        if "_id" not in self.data:
            self.data["_id"] = [str(i) for i in range(len(self.data))]

    def to_dict(self):
        self.data.to_dict('records')

    def sample(self, n: int = 1) -> "QAData":
        return QAData(self.data.sample(n=n))

    def records(self) -> Iterator[dict]:
        for _, row in self.data.iterrows():
            yield row

    def size(self) -> int:
        return len(self.data)