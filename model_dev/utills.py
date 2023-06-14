import torch
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def load_model(args):

    if args.model == 'dlinear':
        from model_dev.models.dlinear import Model
    elif args.model == 'nlinear':
        from model_dev.models.nlinear import Model
    elif args.model == 'nlinear_attention':
        from model_dev.models.nlinear_attention import Model
    elif args.model == 'dlinear_attention':
        from model_dev.models.dlinear_attention import Model    
    else:
        raise NotImplementedError
    model = Model(args)
    return model


def get_stock_meta(instrument_file, data_path):

    # load instrument config for stock meta
    instrumen_config = json.load(open(instrument_file, "r"))
    instrumen_config = {i['instrument_token']: i for i in instrumen_config}
    # read columns header from data_path which is csv
    columns = pd.read_csv(data_path, nrows=1).columns[1:]
    index_to_column = {i: instrumen_config[int(columns[i])] for i in range(len(columns))}
    return index_to_column


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), "{}/checkpoint_{:.4f}.pth".format(path, val_loss))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def read_default_args():

    parent_directory_of_current_script = os.path.dirname(os.path.realpath(__file__))
    with open("{}/default_args.json".format(parent_directory_of_current_script), "r") as f:
        default_args = json.load(f)

    default_args["use_gpu"] = True if torch.cuda.is_available() and default_args["use_gpu"] else False
    if default_args["use_gpu"] and default_args["use_multi_gpu"]:
        default_args["devices"] = default_args["devices"].replace(' ', '')
        device_ids = default_args["devices"].split(',')
        default_args["device_ids"] = [int(id_) for id_ in device_ids]
        default_args["gpu"] = default_args["device_ids"][0]

    return default_args

def get_dates_between(date1, date2, freq='minute'):
    # Date is in 2023-03-27 09:15:00 kind of format
    # freq is in minute, hour, day, week, month, year
    # Returns a list of dates between date1 and date2
    date1 = datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')
    date2 = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
    if freq == 'minute':
        delta = timedelta(minutes=1)
    elif freq == 'hour':
        delta = timedelta(hours=1)
    elif freq == 'day':
        delta = timedelta(days=1)
    elif freq == 'week':
        delta = timedelta(weeks=1)
    elif freq == 'month':
        delta = timedelta(months=1)
    elif freq == 'year':
        delta = timedelta(years=1)
    else:
        raise Exception("Invalid freq")
    dates = []
    while date1 <= date2:
        dates.append(date1.strftime('%Y-%m-%d %H:%M:%S'))
        date1 += delta
    return dates


def get_stock_heatmap_matrix(model, num_stocks, args, setting_suffix=''):

    heat_map_matrix = np.zeros((num_stocks, args.enc_in))
    for t in range(num_stocks):
        setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}_ch_{}{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], t, args.enc_in, setting_suffix)
        weights = os.listdir("{}/{}".format(args.checkpoints, setting))
        sorted_weights = sorted(weights, key=lambda x: float(x.replace('checkpoint_','').replace('.pth','')), reverse=True)
        model.load_state_dict(torch.load("{}/{}/{}".format(args.checkpoints, setting, sorted_weights[-1])))
        attn_weights = model.Attention.weight.cpu().squeeze().detach().numpy()
        heat_map_matrix[t] = attn_weights

    return heat_map_matrix    
