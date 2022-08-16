from abc import ABC, abstractmethod

import os
from tqdm import tqdm
from midi_to_dataframe import NoteMapper, MidiReader


class BaseLoader(ABC):

    @abstractmethod
    def load(self, directory):
        pass


class DataLoadingException(BaseException):
    pass


class StandardMIDILoader(BaseLoader):

    def __init__(self, settings):
        self._note_mapping_config_path = settings["path_to_note_mapping_config"]
        self._note_mapper = NoteMapper(self._note_mapping_config_path)
        self._reader = MidiReader(self._note_mapper)

    def load(self, path_to_training_data):
        data = []
        files = list(self._walkdir(path_to_training_data))
        filescount = len(files)
        print("Loading {} files for processing.".format(filescount))
        for full_path_to_midi in tqdm(files, total=filescount):
            if full_path_to_midi.split(".")[-1].endswith("mid"):
                try:
                    df = self._reader.convert_to_dataframe(full_path_to_midi)
                except:
                    print("Unable to convert {} to Dataframe\n".format(full_path_to_midi))
                    continue
                if len(df) > 0:
                    df = df.reset_index()
                    data.append((full_path_to_midi, df))
        return data

    @staticmethod
    def _walkdir(folder):
        for dirpath, dirs, files in os.walk(folder):
            for filename in files:
                yield os.path.abspath(os.path.join(dirpath, filename))
