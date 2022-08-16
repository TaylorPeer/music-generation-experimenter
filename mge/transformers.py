from abc import ABC, abstractmethod
import json


class Transformer(ABC):

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass


class TransformerException(BaseException):
    pass


class NoteRenamer(Transformer):

    def __init__(self, settings):
        self._path_to_note_renamings = settings["path_to_note_renamings"]
        self._separator = settings["separator"]
        with open(self._path_to_note_renamings, 'r') as f:
            self._note_renamings = json.load(f)

    def transform(self, data):
        transformed_data = []
        for (path, df) in data:
            df["notes"] = df["notes"].apply(lambda notes: self._rename_notes(notes))
            transformed_data.append((path, df))
        return transformed_data

    def inverse_transform(self, data):
        # TODO - not sure yet if this is needed
        pass

    def _rename_notes(self, notes):
        notes = notes.split(",")
        renamed_notes = []
        for note in notes:
            if note == "rest":
                renamed_notes.append("rest")
            else:
                (instrument, note, duration) = note.split("_")
                if instrument != "percussion":
                    octave = ''.join(c for c in note if c.isnumeric())
                    octave = self._note_renamings["octaves"][octave]
                    note = ''.join(c for c in note if not c.isnumeric())
                    note = self._note_renamings["notes"][note]
                    note = "{}{}{}".format(note, self._separator, octave)
                else:
                    note = self._note_renamings["drums"][note]
                instrument = self._note_renamings["instruments"][instrument]
                duration = self._note_renamings["durations"][duration]
                renamed = "{}{}{}{}{}".format(instrument, self._separator, note, self._separator, duration)
                renamed_notes.append(renamed)

        return " ".join(renamed_notes)
