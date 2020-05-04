"""Mobile Bert for Question Anwsering
"""

import numpy as np
import tensorflow as tf

from nlp.bert import tokenization


class MobileBert:
    """Mobile Bert Class"""

    def __init__(self, vocab_path="./model/vocab.txt",
                 model_path="./model/mobilebert_float_20191023.tflite"):
        self.tokenizer = self._create_tokenizer(vocab_path, do_lower_case=True)

        # Get input and output tensors.
        self.interpreter = self._load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.max_seq_len = self.input_details[0]['shape'][1]

    def _load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    @staticmethod
    def _create_tokenizer(vocab_file, do_lower_case=False):
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def _convert(self, query, context, max_seq_len):
        tokens = ['[CLS]']

        # For query
        tokens.extend(self.tokenizer.tokenize(query))
        tokens.append('[SEP]')
        segment_ids = [0] * len(tokens)

        # For context
        tokens.extend(self.tokenizer.tokenize(context))
        tokens.append('[SEP]')
        segment_ids = [1] * len(tokens)

        if len(tokens) > max_seq_len:
            raise ValueError("Input string too long")

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero Mask till seq_length
        zero_mask = [0] * (max_seq_len - len(tokens))
        input_ids.extend(zero_mask)
        input_mask.extend(zero_mask)
        segment_ids.extend(zero_mask)
        return [input_ids], [input_mask], [segment_ids]

    def infer(self, query: str, context: str) -> str:
        """Infer answer based on query and context

        Args:
            query (str): query string
            context (str): context string

        Returns:
            str: answer
        """

        input_ids, input_mask, segment_ids = \
            self._convert(query, context, max_seq_len=self.max_seq_len)
        input_ids_data = np.array(input_ids, dtype=np.int32)
        input_mask_data = np.array(input_mask, dtype=np.int32)
        segment_ids_data = np.array(segment_ids, dtype=np.int32)

        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_ids_data
        )
        self.interpreter.set_tensor(
            self.input_details[1]['index'], input_mask_data
        )
        self.interpreter.set_tensor(
            self.input_details[2]['index'], segment_ids_data
        )

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        end = self.interpreter.get_tensor(self.output_details[0]['index'])
        start = self.interpreter.get_tensor(self.output_details[1]['index'])

        s_pred = np.argmax(start[0])
        e_pred = np.argmax(end[0])

        return " ".join(
            self.tokenizer.convert_ids_to_tokens(
                input_ids[0][s_pred: e_pred + 1]
            )
        )


def main():
    mb = MobileBert()
    query = input("Question: ")
    context = input("Context: ")
    print("Answer:", mb.infer(query, context))


if __name__ == "__main__":
    main()
