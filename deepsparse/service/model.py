import os
os.environ["WAND_OPT_FLAGS"] = "default,~pyramids"

import numpy as np
from typing import Optional, List, Dict

from deepsparse import Context
from deepsparse.engine import LIB
from deepsparse.pipeline import DEEPSPARSE_ENGINE, create_engine
from deepsparse.utils.onnx import overwrite_onnx_model_inputs_for_kv_cache_models
from deepsparse.transformers.utils.helpers import create_causal_mask

PAST_KEY_VALUES_NAME = "past_key_values"

def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DeepSparsePastKeyValues:
    def __init__(self):
        prev_num_tokens = 0
        num_frozen_tokens = 1
        self.internal_past_key_values = LIB.kv_cache(prev_num_tokens, num_frozen_tokens)

class DeepSparseDecoderEngine:
    def __init__ (
        self,
        onnx_file_path: str,
        sequence_length: int = 1024,
        input_ids_length: int = 1,
        batch_size: int = 1,
        num_threads: int = None,
        engine_context: Optional[Context] = None,
    ):

        # setup ONNX graph(s)
        onnx_file_path, cached_outputs, data_type = overwrite_onnx_model_inputs_for_kv_cache_models(
            onnx_file_path=onnx_file_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )

        self.engine_type = DEEPSPARSE_ENGINE

        if self.engine_type == DEEPSPARSE_ENGINE:
            engine_args = {
                "cached_outputs": cached_outputs,
                "batch_size": batch_size,
                "num_cores": num_threads
            }
        else:
            engine_args = {"batch_size": batch_size}

        # compile engine
        print(f"compiling for batch size/sequence length/input ids length: {batch_size}/{sequence_length}/{input_ids_length}")
        self.engine = create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=self.engine_type,
            engine_args=engine_args,
            context=engine_context,
        )
        print(self.engine)

        # save utilties
        self.past_key_value_dtype = data_type
        self.onnx_inputs = self.engine.input_names
        self.empty_past_key_values = self.make_empty_past_key_values()

    # forward function
    def __call__(
        self,
        engine_inputs: Dict[str, np.ndarray],
        past_key_values: DeepSparsePastKeyValues,  # TODO this can be a list
        val_inputs: bool = True
    ):
        # format input into lists (we pass empty past key values)
        inputs = [
            self.empty_past_key_values[name] if name.startswith(PAST_KEY_VALUES_NAME)
            else engine_inputs[name] for name in self.engine.input_names
        ]

        # validate inputs formatted correctly
        if val_inputs:
             self.engine._validate_inputs(inputs)

        if type(past_key_values) is list:
            caches = [pkv.internal_past_key_values for pkv in past_key_values]
        else:
            caches = past_key_values.internal_past_key_values

        # run inference, updates past_key_values internally
        if self.engine_type == DEEPSPARSE_ENGINE:
            output = self.engine._eng_net.execute_list_out(
                inputs,
                caches
            )
        else:
            output = self.engine.run(inputs)
        logits = output[0]
        return logits, past_key_values

    # empty past kvs (dummy values to be passed around)
    def make_empty_past_key_values(self):
        past_key_values = {}
        for idx, name in enumerate(self.onnx_inputs):
            if name.startswith(PAST_KEY_VALUES_NAME):
                past_key_values[name] = np.zeros(
                    self.engine.input_shapes[idx],
                    dtype=self.past_key_value_dtype
                )

        return past_key_values

class DeepSparseDecoderModel:
    def __init__(
        self,
        onnx_file_path: str,
        sequence_length: int = 1024,
        multitoken_length: int = 16,
        batch_size: int = 1,
        num_threads: int = None,
        engine_context: Optional[Context] = None,
    ):
        self.sequence_length = sequence_length
        self.multitoken_length = multitoken_length
        self.batch_size = batch_size

        # compile prefill engine
        self.multitoken_engine = DeepSparseDecoderEngine(
            onnx_file_path=onnx_file_path,
            engine_context=engine_context,
            sequence_length=sequence_length,
            input_ids_length=self.multitoken_length,
            num_threads=num_threads
        )

        # compile decode engines
        self.singletoken_engine = DeepSparseDecoderEngine(
            onnx_file_path=onnx_file_path,
            engine_context=engine_context,
            sequence_length=sequence_length,
            input_ids_length=1,
            batch_size=1,
            num_threads=num_threads
        )

        if batch_size > 1:
            self.batched_singletoken_engine = DeepSparseDecoderEngine(
                onnx_file_path=onnx_file_path,
                engine_context=engine_context,
                sequence_length=sequence_length,
                input_ids_length=1,
                batch_size=batch_size,
                num_threads=num_threads
            )
        else:
            self.batched_singletoken_engine = None

        assert "input_ids" in self.singletoken_engine.onnx_inputs
        assert "attention_mask" in self.singletoken_engine.onnx_inputs
        assert "causal_mask" in self.singletoken_engine.onnx_inputs
        assert "positions" in self.singletoken_engine.onnx_inputs

        # for debugging
        self.decode_count = 0

    def engine_inputs_for_prefill(
        self,
        input_ids: np.ndarray,
    ):
        # split batch into N token_batches
        num_batches = input_ids.shape[1] // self.multitoken_length
        token_batches = [
            input_ids[:, i*self.multitoken_length : (i+1)*self.multitoken_length]
            for i in range(0, num_batches)
        ]

        # format inputs for each of the N token_batches
        for idx, token_batch in enumerate(token_batches):
            num_processed_tokens = self.multitoken_length * idx

            engine_inputs = {}
            engine_inputs["input_ids"] = token_batch

            # make attention mask from the right
            engine_inputs["attention_mask"] = np.zeros((1, self.sequence_length), dtype=np.int64)
            engine_inputs["attention_mask"][:, -(self.multitoken_length + num_processed_tokens):] = 1

            # make positions (building from the right)
            # TODO: handle case when multitoken engine is 1
            assert self.multitoken_length > 1
            engine_inputs["positions"] = np.arange(
                num_processed_tokens, num_processed_tokens + self.multitoken_length
            ).reshape(1, -1).astype(np.int64)

            # make causal mask (building from the right)
            engine_inputs["causal_mask"] = create_causal_mask(
                input_ids=engine_inputs["input_ids"],
                attention_mask=engine_inputs["attention_mask"]
            )
            yield engine_inputs

    def engine_inputs_for_decode(
        self,
        input_ids: List[np.ndarray],
    ):
        # TODO: assert input_ids all have same shape
        assert type(input_ids) is list
        assert type(input_ids[0]) is np.ndarray
        assert len(input_ids) > 0
        assert len(input_ids[0].shape) == 2
        assert input_ids[0].shape[1] < self.sequence_length

        batch_size = len(input_ids)

        engine_inputs = {}

        last_input_ids = [x[:,-1:] for x in input_ids]

        engine_inputs["input_ids"] = np.concatenate(last_input_ids, axis=0)

        engine_inputs["attention_mask"] = np.zeros((batch_size, self.sequence_length), dtype=np.int64)
        for b in range(batch_size):
            engine_inputs["attention_mask"][b, -input_ids[b].shape[1]:] = 1

        engine_inputs["causal_mask"] = create_causal_mask(
            engine_inputs["input_ids"],
            engine_inputs["attention_mask"]
        )

        poses = [pos.shape[1] - 1 for pos in input_ids]
        engine_inputs["positions"] = np.array(poses, dtype=np.int64)[:,None]

        return engine_inputs

    def decode_common(
        self,
        batched_input_ids: List[np.ndarray],
        batched_past_key_values: List[DeepSparsePastKeyValues]
    ) -> (np.ndarray, List[DeepSparsePastKeyValues]):

        assert len(batched_input_ids) == len(batched_past_key_values)

        batched_logits = []
        batched_new_key_values = []

        chunks = zip(
            chunkify(batched_input_ids, self.batch_size),
            chunkify(batched_past_key_values, self.batch_size)
        )

        for input_ids, past_key_values in chunks:
            # assert input is of shape [1,seq_len] w/ seq_len < self.sequence_len
            assert len(input_ids[0].shape) == 2
            assert input_ids[0].shape[1] < self.sequence_length

            engine_inputs = self.engine_inputs_for_decode(input_ids)

            if len(input_ids) == self.batch_size and self.batch_size != 1:
                logits, new_key_values = self.batched_singletoken_engine(
                    engine_inputs,
                    past_key_values
                )
                batched_logits.append(logits)
                # TODO: this is bogus because the caches are updated in place
                batched_new_key_values.append(new_key_values)
            else:
                for i in range(len(input_ids)):
                    engine_inputs_batch = {}
                    engine_inputs_batch["input_ids"] = engine_inputs["input_ids"][i:i+1,:]
                    engine_inputs_batch["attention_mask"] = engine_inputs["attention_mask"][i:i+1,:]
                    engine_inputs_batch["causal_mask"] =engine_inputs["causal_mask"][i:i+1,:]
                    engine_inputs_batch["positions"] = engine_inputs["positions"][i:i+1,:]

                    logits, new_key_values = self.singletoken_engine(
                        engine_inputs_batch,
                        past_key_values[i])
                    batched_logits.append(logits)
                    batched_new_key_values.append(new_key_values)

        return np.concatenate(batched_logits, axis=0), batched_past_key_values

    def decode(
        self,
        batched_input_ids: List[np.ndarray],
        batched_past_key_values: List[DeepSparsePastKeyValues]
    ) -> (np.ndarray, List[DeepSparsePastKeyValues]):
        self.decode_count = self.decode_count + 1
        logits, pkv = self.decode_common(batched_input_ids, batched_past_key_values)
        return logits, pkv


    def prefill(
        self,
        input_ids: np.ndarray,
    ) -> (np.ndarray, DeepSparsePastKeyValues):

        # assert input is of shape [1,seq_len] w/ seq_len < self.sequence_len
        assert len(input_ids.shape) == 2
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] < self.sequence_length

        tokens_processed = 0

        # setup empty past key values
        past_key_values = DeepSparsePastKeyValues()

        # loop through chunks, run inference w/ multitoken engine
        for engine_inputs in self.engine_inputs_for_prefill(input_ids):
            logits, fake_past_key_values = self.multitoken_engine(
                engine_inputs,
                past_key_values
            )
            tokens_processed += self.multitoken_length

        # if anything left over, run inference w/ singletoken engine
        while tokens_processed < input_ids.shape[1]:
            assert len(input_ids.shape) == 2
            logits, fake_past_key_values = self.decode_common(
                [input_ids[:,:tokens_processed+1]],
                [past_key_values]
            )
            tokens_processed += 1

        return logits, [past_key_values]

    def forward(
        self,
        input_ids: List[np.ndarray],
        past_key_values: List[Optional[DeepSparsePastKeyValues]],
    ):
        assert len(past_key_values) > 0
        if past_key_values[0] is None:
            assert len(input_ids) == 1
            return self.prefill(input_ids[0])
        else:
            return self.decode(input_ids, past_key_values)

    def __call__(
        self,
        input_ids: List[np.ndarray],
        past_key_values: List[Optional[DeepSparsePastKeyValues]] = [],
    ):
        return self.forward(input_ids, past_key_values)
