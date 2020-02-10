import logging
import os
import numpy as np
import random
import math
from fgenerator import BinaryClassificationProcessor
from transformers.data.processors.utils import InputFeatures
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
from settings import TqdmLoggingHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

def convert_example_to_feature(
    example_row,
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    sep_token_extra=False
):
    SEQ_LEN_DISPATCH = {
        'ht': _head_tail,
        'rs': _random_span_no_overlap,
        'rsd': _random_span_delete,
        'head': _head,
        'rso': _random_spans_overlap
    }

    example = example_row[0]
    label_map = example_row[1]
    max_seq_length = example_row[2]
    seq_func_params = example_row[3]
    tokenizer = example_row[4]
    output_mode = example_row[5]
    cls_token_at_end = example_row[6]
    cls_token = example_row[7]
    sep_token = example_row[8]
    cls_token_segment_id = example_row[9]
    pad_on_left = example_row[10]
    pad_token_segment_id = example_row[11]
    sep_token_extra = example_row[12]
    seq_len_func = example_row[13]
        
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2

    tokens_a = tokenizer.tokenize(example.text_a)
    if (len(tokens_a) > max_seq_length - special_tokens_count):
        seq_func_params = seq_func_params[0] if len(seq_func_params) < 2 else seq_func_params
        
        tokens_a = SEQ_LEN_DISPATCH[seq_len_func](
            tokens_a,
            max_seq_length,
            seq_func_params,
            special_tokens_count
        )
        
        
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(
        input_ids
    )

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = (
            [pad_token] * padding_length
        ) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1]
            * padding_length
        ) + input_mask
        token_type_ids = (
            [pad_token_segment_id] * padding_length
        ) + segment_ids
    else:
        input_ids = input_ids + (
            [pad_token] * padding_length
        )
        attention_mask = input_mask + (
            [0 if mask_padding_with_zero else 1]
            * padding_length
        )
        token_type_ids = segment_ids + (
            [pad_token_segment_id] * padding_length
        )

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label_id)

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    seq_func_params,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    sep_token_extra=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    seq_len_func='ht'
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {
        label: i for i, label in enumerate(label_list)
    }

    examples = [
        (
            example,
            label_map,
            max_seq_length,
            seq_func_params,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            seq_len_func,
        )
        for example in examples
    ]

    process_count = cpu_count() - 2

    with Pool(process_count) as p:
        features = list(
            tqdm(
                p.imap(
                    convert_example_to_feature,
                    examples,
                    chunksize=500
                ),
                total=len(examples)
            )
        )

    return features


def _truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()


def _head_tail(
    tokens, max_seq_length, special_tokens_count
):
    """
    Based on the following paper: https://arxiv.org/pdf/1905.05583.pdf

    head length default = 128
    tail length default = 382
    """
    head_len, tail_len = max_seq_length
    # Account for [CLS] and [SEP]
    assert head_len + tail_len <= 512
    dv = special_tokens_count / 2
    stc_head = (
        math.ceil(dv)
        if head_len > tail_len
        else math.floor(dv)
    )
    stc_tail = special_tokens_count - stc_head

    return (
        tokens[0 : head_len - stc_head]
        + tokens[-tail_len + stc_tail :]
    )

def _head(tokens, max_seq_length, special_tokens_count):
    return tokens[
            : (max_seq_length - special_tokens_count)
        ]
        
     
     
def _random_span_no_overlap(
    tokens, max_seq_length, seq_func_params, special_tokens_count
):
    """
    Splits the tokens into X spans by scrolling through the document in order
    Randomly selects Y spans from here without repeating spans
    """

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # TODO: figure out how to scale this by the length of the tokens
    random_span_len = seq_func_params
    num_docs = int(max_seq_length / int(random_span_len)) or 1
    
    chunked_tokens = np.array(
        list(chunks(tokens, int(random_span_len)))
    )
    
    random_doc_indices = list(
        np.random.choice(
            np.arange(len(chunked_tokens)),
            num_docs,
            replace=False,
        )
    )
    random_doc_indices.sort()
    return np.concatenate(
        chunked_tokens[random_doc_indices]
    ).tolist()[: max_seq_length - special_tokens_count]

def _random_span_delete(
    tokens, max_seq_length, seq_func_params, special_tokens_count
):
    """
    Select spans randomly of X length from the input tokens
    After selecting a span, it is removed from the tokens before 
    selecting the new one, in this way there is no overlap
    """
    # TODO: figure out how to scale this by the length of the tokens
    random_span_len = seq_func_params
    num_docs = int(max_seq_length / random_span_len) or 1
    spans = np.array([])
    for i in range(0, num_docs):

        rs_ = random.randint(
            0, len(tokens)
        )  # random start index
        re_ = (
            rs_ + random_span_len
            if rs_ + random_span_len < len(tokens)
            else len(tokens)
        )
        spans = np.concatenate((spans, tokens[rs_:re_]))
        tokens = np.delete(tokens, range(rs_, re_))

    return spans.tolist()[: max_seq_length - special_tokens_count]

def _random_spans_overlap(
    tokens, max_seq_length, seq_func_params, special_tokens_count
):
    """
    Tokens arent remove here and spans arent created by scroll through the input document
    instead they are selected randomly from the tokens, with overlap
    """
    # TODO: figure out how to scale this by the length of the tokens
    random_span_len = seq_func_params
    num_docs = int(max_seq_length / random_span_len) or 1
    spans = np.array([])
    for i in range(0, num_docs):

        rs_ = random.randint(
            0, len(tokens)
        )  # random start index
        re_ = (
            rs_ + random_span_len
            if rs_ + random_span_len < len(tokens)
            else len(tokens)
        )
        spans = np.concatenate((spans, tokens[rs_:re_]))
        

    return spans.tolist()[: max_seq_length - special_tokens_count]
