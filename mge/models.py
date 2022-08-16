from abc import ABC, abstractmethod

import os
import math
import time
import numpy as np
import pandas as pd

# For Markov:
import markovify
from midi_to_dataframe import NoteMapper, MidiWriter

# For AitextgenGPT2:
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import ByteLevelBPETokenizer
from aitextgen import aitextgen
from aitextgen.utils import GPT2Config
from aitextgen.TokenDataset import TokenDataset

OUTPUT_DIR = "/output/"


class BaseModel(ABC):

    def __init__(self):
        self._output_dir = None

    def get_output_dir(self):
        return self._output_dir

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    def create_model_directory(directory):
        if os.path.isdir(directory):
            raise TrainingException("Model directory '{}' already exists".format(directory))
        os.makedirs(directory)

    @staticmethod
    def get_random_output_filename():
        current_milli_time = round(time.time() * 1000)
        return "generated_sequence_{}.mid".format(current_milli_time)

    @staticmethod
    def notes_to_dataframe(notes, bpm=120):
        df = pd.DataFrame(notes)
        df.columns = ["notes"]

        df["timestamp"] = 0
        df["timestamp"] = df.index * 60

        df["bpm"] = bpm
        df["time_signature"] = "4/4"

        beats = np.arange(1.0, 5.0, 0.25).tolist()
        repeat = math.ceil(len(notes) / len(beats))
        beats = beats * repeat
        beats = beats[:len(notes)]
        df["beat"] = beats

        measures = list(range(1, 4 + 1))
        measures = np.repeat(measures, 64)
        df["measure"] = measures[:len(notes)]

        df = df[["timestamp", "bpm", "time_signature", "measure", "beat", "notes"]]
        return df


class TrainingException(BaseException):
    pass


class GenerationException(BaseException):
    pass


class Markov(BaseModel):

    def __init__(self, settings):
        self._model_directory = settings["model_directory"]
        self._state_size = settings["state_size"]
        self._model = None
        self._note_mapping_config_path = settings["path_to_note_mapping_config"]
        self._note_mapper = NoteMapper(self._note_mapping_config_path)
        self._midi_writer = MidiWriter(self._note_mapper)
        self._output_dir = self._model_directory + OUTPUT_DIR

    def train(self, data):
        self.create_model_directory(self._model_directory)

        # Convert MIDI dataframes to newline-separated text representations
        dfs = [df for (_, df) in data]
        lines = [" ".join(df["notes"].tolist()) for df in dfs]

        # Build the model
        self._model = markovify.NewlineText(lines, state_size=self._state_size)
        # TODO store model?

    def generate(self):
        output_dir = self._model_directory + OUTPUT_DIR
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        try:
            sentence = self._model.make_sentence()
            if sentence is not None:
                notes = sentence.split(" ")
                df = self.notes_to_dataframe(notes, bpm=120)  # TODO BPM
                midi_path = output_dir + self.get_random_output_filename()
                self._midi_writer.convert_to_midi(df, midi_path)
            else:
                raise GenerationException("Generated sequence was empty.")
        except Exception as e:
            raise GenerationException(e)


class AitextgenGPT2(BaseModel):

    def __init__(self, settings):
        self._model_directory = settings["model_directory"]
        self._n_positions = settings["n_positions"]
        self._n_layer = settings["n_layer "]
        self._n_head = settings["n_head"]
        self._num_steps = settings["num_steps"]
        self._batch_size = settings["batch_size"]

    @staticmethod
    def _create_term(notes, vectorizer):
        """
        Creates a "compound term" representation from a set of individual notes.
        """
        feature_vector = vectorizer.transform([notes]).toarray()[0]
        scored_terms = [(s, t) for (s, t) in zip(feature_vector, vectorizer.get_feature_names_out()) if s > 0]
        scored_terms.sort(key=lambda tup: tup[0], reverse=True)
        terms = []
        for (score, term) in scored_terms:
            terms.append(term)
        term = "".join(terms)
        return term

    def train(self, data):

        self.create_model_directory(self._model_directory)

        # Compute frequencies of all notes (for external analysis)
        print("Computing note frequencies")
        dfs = [df for (_, df) in data]
        training_data_df = pd.concat(dfs)
        grouped_df = training_data_df.groupby(['notes'])['notes']
        df = grouped_df.count().sort_values(ascending=False).to_frame().rename(columns={'notes': 'count'}).reset_index()
        df.to_csv(self._model_directory + "/note_frequencies.csv", index=None)
        print("...note frequency computation completed")

        print("Creating note words")
        vectorizer = TfidfVectorizer(lowercase=False)
        vectorizer = vectorizer.fit(df["notes"])
        df["word"] = df.parallel_apply(lambda row: self._create_term(row["notes"], vectorizer), axis=1)
        df[["notes", "word"]].to_csv(self._model_directory + "/note_words.csv", index=None)
        print("...note word creation completed")

        unique_notes = len(df[~df["notes"].str.contains(" ")])
        unique_compound_notes = len(df["word"].unique())
        print("Model contains {} unique notes and {} compound notes\n".format(unique_notes, unique_compound_notes))

        print("Matching song information with known note words to create tokenizer corpus")
        texts = []
        for _df in tqdm(dfs):
            _df = pd.merge(_df, df, how='left', on="notes")
            text = " ".join(_df["word"])
            texts.append({"text": text})
        print("Serializing song information as tokenizer corpus file")
        corpus_df = pd.DataFrame(texts)
        corpus_file = self._model_directory + "/tokenizer_corpus.csv"
        corpus_df.to_csv(corpus_file, index=False, header=None)
        print("...tokenizer corpus creation completed")

        print("Training tokenizer")
        tokenizer = ByteLevelBPETokenizer(dropout=None, trim_offsets=True)
        tokenizer.train(
            files=self._model_directory + "/tokenizer_corpus.csv",
            vocab_size=512,  # TODO parameterize
            min_frequency=2,  # TODO parameterize
            special_tokens=["<|endoftext|>"],
            show_progress=True
        )
        model_name = "model_name"  # TODO parameterize
        tokenizer.save(self._model_directory + "/tokenizer.json")
        tokenizer.save_model(self._model_directory, model_name)

        config = GPT2Config()
        config.n_positions = self._n_positions
        config.n_layer = self._n_layer
        config.n_head = self._n_head

        corpus_file = self._model_directory + "/tokenizer_corpus.csv"
        vocab_file = self._model_directory + "/" + model_name + "-vocab.json"
        merges_file = self._model_directory + "/" + model_name + "-merges.txt"
        model = aitextgen(config=config, vocab_file=vocab_file, merges_file=merges_file)
        data = TokenDataset(corpus_file, line_by_line=True, vocab_file=vocab_file, merges_file=merges_file,
                            block_size=64)
        model.train(data, batch_size=self._batch_size, num_steps=self._num_steps, output_dir=self._model_directory)
        model.save(self._model_directory + "/model")

    def generate(self):
        pass  # TODO
