import logging

import torch


def trim_lpad_confidence(batch_of_seq_actual, batch_of_seq_pred_conf, pad_index=0):
    """
    Trims pad at the left, based on the ground truth / actual labels.
    """
    pred_batch_labels = torch.max(batch_of_seq_pred_conf, dim=-1)[1]
    result_actual = []
    result_pred_conf = []

    for bi, (s_actual, s_pred_label) in enumerate(zip(batch_of_seq_actual, pred_batch_labels)):
        # Assume first 0 is SEP, hence start with token after
        i = 1
        s_actual_i = s_actual[i]

        # 0 is the pad index
        while s_actual_i.item() == pad_index and i < len(s_actual):
            i += 1
            s_actual_i = s_actual[i]

        result_actual.append(s_actual[i:])
        result_pred_conf.append(batch_of_seq_pred_conf[bi][i:])

    # Change the 2 d to 1 dimensional array as each seq in the batch has a diff length.
    result_actual, result_pred_conf = torch.cat(result_actual, dim=0), torch.cat(result_pred_conf, dim=0)

    return result_actual, result_pred_conf


def trim_lpad(batch_of_seq_x, batch_of_seq_y):
    """
       Trims pad at the left, based on the ground truth (batch_of_seq_x)
    """
    logger = logging.getLogger(__name__)
    result_x = []
    result_y = []
    for s_x, s_y in zip(batch_of_seq_x, batch_of_seq_y):
        # first char is a sep
        init = 1
        i = init
        xi = s_x[i]

        # 0 is the pad index
        while xi.item() == 0 and i < len(s_x):
            i += 1
            xi = s_x[i]

        trimmed_s_x = s_x[(i):]
        result_x.append(trimmed_s_x)
        trimmed_s_y = s_y[(i):]
        result_y.append(trimmed_s_y)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prediction \n\t{trimmed_s_y}")
            logger.debug(f"Actual  \n\t{trimmed_s_x}")

    # Change the 2 d to 1 dimentional array as each seq in the batch has a diff length.
    result_x, result_y = torch.cat(result_x, dim=0), torch.cat(result_y, dim=0)

    return result_x, result_y


def align_predicted_raw_text(pred_tokens, input_tokens_len):
    predicted_raw_text = ""
    assert input_tokens_len <= len(pred_tokens)

    pred_tokens_without_pad = pred_tokens[(-1 * input_tokens_len - 1):-1]
    for i in reversed(range(len(pred_tokens_without_pad))):

        token = pred_tokens_without_pad[i]
        space = "" if i == (input_tokens_len - 1) else " "
        if token.startswith("##"):
            space = ""
            token = token[2:]

        predicted_raw_text = predicted_raw_text + space + token

    return predicted_raw_text


def get_raw_token_len_without_pad(text_token_tensor, pad_index=0):
    i = len(text_token_tensor) - 2
    while text_token_tensor[i] == pad_index:
        i = i - 1
    return i
