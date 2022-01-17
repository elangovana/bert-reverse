import torch


def trim_lpad_confidence(batch_of_seq_actual, batch_of_seq_pred_conf, pad_index=0):
    """
    Trims pad at the left, provided both predicted and actual match in their pad positions
    """
    pred_batch_labels = torch.max(batch_of_seq_pred_conf, dim=-1)[1]
    result_actual = []
    result_pred_conf = []

    for bi, (s_actual, s_pred_label) in enumerate(zip(batch_of_seq_actual, pred_batch_labels)):
        # Assume 0 is SEP, hence start with 1
        i = 1
        s_actual_i = s_actual[i]
        s_pred_label_i = s_pred_label[i]

        # 0 is the pad index
        while s_actual_i.item() == s_pred_label_i.item() and s_pred_label_i.item() == pad_index and i < len(
                s_pred_label):
            i += 1
            s_actual_i = s_actual[i]
            s_pred_label_i = s_pred_label[i]

        if i > 1:
            result_actual.append(s_actual[i:])
            result_pred_conf.append(batch_of_seq_pred_conf[bi][i:])
        else:
            result_actual.append(s_actual)
            result_pred_conf.append(batch_of_seq_pred_conf[bi])

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
        i = 1
        xi = s_x[i]
        yi = s_y[i]

        # 0 is the pad index
        while xi.item() == yi.item() and yi.item() == 0 and i < len(s_y):
            i += 1
            xi = s_x[i]
            yi = s_y[i]

        if i > 1:
            result_x.append(s_x[i:])
            result_y.append(s_y[i:])
        else:
            result_x.append(s_x)
            result_y.append(s_y)

    # Change the 2 d to 1 dimentional array as each seq in the batch has a diff length.
    result_x, result_y = torch.cat(result_x, dim=0), torch.cat(result_y, dim=0)

    return result_x, result_y
