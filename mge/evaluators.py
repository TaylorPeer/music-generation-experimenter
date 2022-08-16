from abc import ABC, abstractmethod

import muspy
import tempfile
from midi_to_dataframe import NoteMapper, MidiReader, MidiWriter


class BaseEvaluator(ABC):

    def __init__(self, settings):
        self._note_mapping_config_path = None if "path_to_note_mapping_config" not in settings else settings[
            "path_to_note_mapping_config"]
        self._note_mapper = None if self._note_mapping_config_path is None else NoteMapper(
            self._note_mapping_config_path)
        self._reader = None if self._note_mapper is None else MidiReader(self._note_mapper)
        self._writer = None if self._note_mapper is None else MidiWriter(self._note_mapper)

    @abstractmethod
    def evaluate(self, midi_path):
        pass

    def _evaluate_with_tempfile(self, function, original_path):
        df = self._reader.convert_to_dataframe(original_path)
        with tempfile.NamedTemporaryFile() as temp_file:
            self._writer.convert_to_midi(df, temp_file.name)
            return function(temp_file.name)


class EvaluationException(BaseException):
    pass


class MeasureCount(BaseEvaluator):

    def __init__(self, settings):
        super().__init__(settings)

    def evaluate(self, midi_path):
        df = self._reader.convert_to_dataframe(midi_path)
        return df["measure"].max().item()


class EmptyBeatRate(BaseEvaluator):

    def __init__(self, settings):
        super().__init__(settings)

    def evaluate(self, midi_path):
        music = muspy.read_midi(midi_path)
        return muspy.empty_beat_rate(music)


class DrumPatternConsistency(BaseEvaluator):

    def __init__(self, settings):
        super().__init__(settings)

    @staticmethod
    def _compute_drum_pattern_consistency(midi_path):
        music = muspy.read_midi(midi_path)
        return muspy.drum_pattern_consistency(music)

    def evaluate(self, midi_path):
        return self._evaluate_with_tempfile(self._compute_drum_pattern_consistency, midi_path)


class GrooveConsistency(BaseEvaluator):

    def __init__(self, settings):
        super().__init__(settings)
        self._measure_resolution = settings["measure_resolution"]

    def evaluate(self, midi_path):
        music = muspy.read_midi(midi_path)
        return muspy.groove_consistency(music, self._measure_resolution)
