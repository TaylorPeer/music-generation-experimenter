from abc import ABC, abstractmethod


class Filter(ABC):

    @abstractmethod
    def filter(self, data):
        pass


class FilterException(BaseException):
    pass


class MetadataFilter(Filter):

    def __init__(self, settings):
        self._filter_if_missing_measure = settings["filter_if_missing_measure"]

    def filter(self, data):
        filtered_data = []
        for (path, df) in data:
            if self._filter_if_missing_measure:
                if "measure" not in df:
                    print("No measure information available in {}\n".format(path))
                    continue
                else:
                    filtered_data.append((path, df))
        return filtered_data


class InstrumentFilter(Filter):

    def __init__(self, settings):
        self._must_contain = settings["must_contain"]
        self._must_not_contain = settings["must_not_contain"]

    def filter(self, data):
        filtered_data = []
        for (path, df) in data:
            unique_notes = set(" ".join(" ".join(df["notes"].tolist()).split(",")).split(" "))
            keep = False if len(self._must_contain) > 0 else True
            for must_contain in self._must_contain:
                for note in unique_notes:
                    if must_contain in note:
                        keep = True
            for must_not_contain in self._must_not_contain:
                for note in unique_notes:
                    if must_not_contain in note:
                        keep = False
            if keep is True:
                filtered_data.append((path, df))
        return filtered_data
