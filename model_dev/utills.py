import torch
import json
import numpy as np


def load_model(model_name):

    if model_name == 'dlinear':
        from model_dev.models.dlinear import Model
    elif model_name == 'nlinear':
        from model_dev.models.nlinear import Model
    elif model_name == 'nlinear_attention':
        from model_dev.models.nlinear_attention import Model
    elif model_name == 'dlinear_attention':
        from model_dev.models.dlinear_attention import Model    
    else:
        raise NotImplementedError
    return Model


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

    with open("/Users/shiva/Desktop/maestro/model_dev/default_args.json", "r") as f:
        default_args = json.load(f)

    default_args["use_gpu"] = True if torch.cuda.is_available() and default_args["use_gpu"] else False
    if default_args["use_gpu"] and default_args["use_multi_gpu"]:
        default_args["devices"] = default_args["devices"].replace(' ', '')
        device_ids = default_args["devices"].split(',')
        default_args["device_ids"] = [int(id_) for id_ in device_ids]
        default_args["gpu"] = default_args["device_ids"][0]

    return default_args

def detect_constant_price(stock_prices, duration):

    # Create a boolean array which is True where prices are the same as the previous day
    is_constant = np.diff(stock_prices, prepend=stock_prices[0]) == 0

    # Count how many constant days are before each day
    constant_days_count = np.cumsum(is_constant) - np.cumsum(np.pad(is_constant[:-duration+1], (duration-1, 0)))

    # Find the first day where there are `duration` constant days before it
    constant_period_end = np.argmax(constant_days_count >= duration)

    # If there's no period of constant prices long enough, return None
    if constant_days_count[constant_period_end] < duration:
        return None

    return constant_period_end - duration + 1, constant_period_end
