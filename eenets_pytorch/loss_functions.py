"""
Edanur Demir
Loss functions used in EENet training
"""
import sys
import torch
import torch.nn.functional as F


def loss(model, exit_tag, pred, target, conf, cost, device):
    """loss function

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This function switches between the loss functions.
    """
    if model.loss_func == "v0":
        return loss_v0(model.num_ee, pred, target, conf, cost)
    if model.loss_func == "v1":
        return loss_v1(model, pred, target, conf, cost)
    if model.loss_func == "v2":
        return loss_v2(model, pred, target, conf, cost)
    if model.loss_func == "v3":
        return loss_v3(model, pred, target, conf, cost)
    if model.loss_func == "v4":
        return loss_v4(model, exit_tag, pred, target, conf, cost, device)


def loss_v0(num_ee, pred, target, conf, cost):
    """loss version 0

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the cumulative loss of all exit points.
    It is used in the first stage of two-stage training.
    """
    pred_loss = 0
    cost_loss = 0
    for i in range(num_ee + 1):
        pred_loss += F.nll_loss(pred[i].log(), target)
    cum_loss = pred_loss

    return cum_loss, pred_loss, cost_loss


def loss_v1(model, pred, target, conf, cost):
    """loss version 1

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the fusion loss of the cross_entropy loss and cost loss.
    These loss parts are calculated in a recursive way as following:
    Prediction'_i = confidence_i * prediction_i + (1 - confidence_i) * Prediction'_(i+1)
    Cost'_i       = confidence_i * cost_i       + (1 - confidence_i) * Cost'_(i+1)
    """
    cum_pred = pred[model.num_ee]
    cum_cost = cost[model.num_ee]
    for i in range(model.num_ee - 1, -1, -1):
        cum_pred = conf[i] * pred[i] + (1 - conf[i]) * cum_pred
        cum_cost = conf[i] * cost[i] + (1 - conf[i]) * cum_cost
    pred_loss = F.nll_loss(cum_pred.log(), target)
    cost_loss = cum_cost.mean()
    cum_loss = pred_loss + model.lambda_coef * cost_loss

    return cum_loss, pred_loss, cost_loss


def loss_v2(model, pred, target, conf, cost):
    """loss version 2

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function is the cumulative loss of loss_v1 by recursively.
    It aims to provide a more fair training.
    """
    target = target.flatten()
    cum_pred = [None] * model.num_ee + [pred[model.num_ee]]
    cum_cost = [None] * model.num_ee + [cost[model.num_ee]]
    pred_loss = F.smooth_l1_loss(cum_pred[-1].log(), target)
    cum_loss = pred_loss + model.lambda_coef * cum_cost[-1].mean()
    for i in range(model.num_ee - 1, -1, -1):
        cum_pred[i] = conf[i] * pred[i] + (1 - conf[i]) * cum_pred[i + 1]
        cum_cost[i] = conf[i] * cost[i] + (1 - conf[i]) * cum_cost[i + 1]
        pred_loss = F.smooth_l1_loss(cum_pred[i].log(), target)
        cost_loss = cum_cost[i].mean()
        cum_loss += pred_loss + model.lambda_coef * cost_loss

    return cum_loss, 0, 0


def loss_v3(model, pred, target, conf, cost):
    """loss version 3

    Arguments are
    * args:     command line arguments entered by user.
    * pred:     prediction result of each exit point.
    * target:   target prediction values.
    * conf:     confidence value of each exit point.
    * cost:     cost rate of the each exit point.

    This loss function uses the normalized confidence values.
    """
    conf_sum = 0
    for i in range(len(conf)):
        conf_sum += conf[i].mean()
    norm_conf = [conf[i].mean() / conf_sum for i in range(len(conf))]

    cum_loss = 0
    for i in range(model.num_ee + 1):
        pred_loss = F.nll_loss(pred[i].log(), target)
        cost_loss = cost[i].mean()
        cum_loss += norm_conf[i] * (pred_loss + model.lambda_coef * cost_loss)

    return cum_loss, 0, 0


def loss_v4(model, exit_tag, pred, target, conf, cost, device):
    """loss version 4
    Arguments are
    * args:       command line arguments entered by user.
    * exit_tag:   exit tag of examples in the batch.
    * pred:       prediction result of each exit point.
    * target:     target prediction values.
    * conf:       confidence value of each exit point.
    * cost:       cost rate of the each exit point.
    This loss function uses the exit tags of examples pre-assigned by the model.
    """
    cum_pred = pred[model.num_ee]
    cum_cost = cost[model.num_ee]
    conf_loss = 0
    for i in range(model.num_ee + 1):
        exiting_examples = (exit_tag == i).to(device, dtype=torch.float)
        not_exiting_examples = (exit_tag != i).to(device, dtype=torch.float)
        cum_pred = exiting_examples * pred[i] + not_exiting_examples * cum_pred
        cum_cost = exiting_examples * cost[i] + not_exiting_examples * cum_cost

        exiting_rate = exiting_examples.sum().item() / len(exit_tag)
        not_exiting_rate = not_exiting_examples.sum() / len(exit_tag)
        conf_weights = (
            exiting_examples * not_exiting_rate + not_exiting_examples * exiting_rate
        )

        conf_loss += F.binary_cross_entropy(conf[i], exiting_examples, conf_weights)
    # cum_pred = cum_pred.type(torch.LongTensor)
    #     state_action_values, expected_state_action_values.unsqueeze(1)
    # )
    cum_pred = torch.flatten(cum_pred)
    pred_loss = F.smooth_l1_loss(cum_pred, target)
    cost_loss = cum_cost.mean()
    cum_loss = pred_loss + model.lambda_coef * cost_loss + conf_loss

    return cum_loss, pred_loss, cost_loss


def update_exit_tags(model, batch_size, pred, target, cost, device):
    """loss version 4

    Arguments are
    * args:         command line arguments entered by user.
    * batch_size:   current size of the batch.
    * pred:         prediction result of each exit point.
    * target:       target prediction values.
    * cost:         cost rate of the each exit point.

    This function updates and returns the exit tags.
    """
    cum_loss = (torch.ones(batch_size) * sys.maxsize).to(device)
    exit_tag = (torch.ones(batch_size) * model.num_ee).to(device, dtype=torch.int)

    for exit in range(model.num_ee + 1):
        loss = (
            F.smooth_l1_loss(pred[exit].log(), target, reduction="none")
            + model.lambda_coef * cost[exit]
        )
        smaller_values = (loss < cum_loss).to(device, dtype=torch.float)
        greater_values = (loss >= cum_loss).to(device, dtype=torch.float)
        cum_loss = loss * smaller_values + cum_loss * greater_values
        exit_tag = exit * smaller_values.int() + exit_tag * greater_values.int()

    return exit_tag.reshape(-1, 1)
