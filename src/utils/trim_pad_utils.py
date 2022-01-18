import torch


def trim_lpad_confidence(batch_of_seq_actual, batch_of_seq_pred_conf, pad_index=0):
    """
    Trims pad at the left, provided both predicted and actual match in their pad positions
    """
    pred_batch_labels = torch.max(batch_of_seq_pred_conf, dim=-1)[1]
    result_actual = []
    result_pred_conf = []

    for bi, (s_actual, s_pred_label) in enumerate(zip(batch_of_seq_actual, pred_batch_labels)):
        # Assume last 0 is SEP, hence start with token before
        i = len(s_actual) - 2
        s_actual_i = s_actual[i]

        # 0 is the pad index
        while s_actual_i.item() == pad_index and i > 0:
            i -= 1
            s_actual_i = s_actual[i]

        result_actual.append(s_actual[:(i + 1)])
        result_pred_conf.append(batch_of_seq_pred_conf[bi][-(i + 1):])

    # Change the 2 d to 1 dimensional array as each seq in the batch has a diff length.
    result_actual, result_pred_conf = torch.cat(result_actual, dim=0), torch.cat(result_pred_conf, dim=0)

    return result_actual, result_pred_conf


def trim_lpad(batch_of_seq_x, batch_of_seq_y):
    """
        Trims pad at the left, provided both  match in their pad positions
    """
    result_x = []
    result_y = []
    for s_x, s_y in zip(batch_of_seq_x, batch_of_seq_y):
        # Last char is a sep
        init = len(s_x) - 2
        i = init
        xi = s_x[i]

        # 0 is the pad index
        while xi.item() == 0 and i > 0:
            i -= 1
            xi = s_x[i]

        result_x.append(s_x[:(i + 1)])
        result_y.append(s_y[-(i + 1):])

    # Change the 2 d to 1 dimentional array as each seq in the batch has a diff length.
    result_x, result_y = torch.cat(result_x, dim=0), torch.cat(result_y, dim=0)

    return result_x, result_y
