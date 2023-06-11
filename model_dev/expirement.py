import os
import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from dataloader import data_provider
from utills import EarlyStopping, adjust_learning_rate, metric, load_model
warnings.filterwarnings('ignore')


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_name = self.args.model
        self.model = self._build_model().to(self.device)
        self.target = self.args.target

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class ExpMain(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.train_data, self.train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            self.vali_data, self.vali_loader = self._get_data(flag='val')

    def _build_model(self):

        model = load_model(self.args)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of parameters is: {}'.format(params))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            # print num params
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                batch_y = batch_y.to(self.device)

                if self.model_name.endswith('attention'):
                    batch_y = batch_y[:, :, self.target]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            os.system('rm -rf ' + path + '/*')

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(tqdm(self.train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.model_name.endswith('attention'):
                    batch_y = batch_y[:, :, self.target]

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            print("Epoch: {} | time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if not self.args.train_only:
                vali_loss = self.vali(self.vali_data, self.vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch+1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch+1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)



    def test(self, setting, mode='test'):
        
        test_data, test_loader = self._get_data(flag=mode)

        path = os.path.join(self.args.checkpoints, setting)
        weights = os.listdir(path)
        sorted_weights = sorted(weights, key=lambda x: float(x.replace('checkpoint_','').replace('.pth','')), reverse=True)
        weights_path = os.path.join(path, sorted_weights[-1])

        print('loading model from {}'.format(weights_path))
        self.model.load_state_dict(torch.load(weights_path))

        preds = []
        trues = []
        train_loss = []
        # inputx = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        criterion = self._select_criterion()
        self.model.eval()
        with torch.no_grad():

            for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.model_name.endswith('attention'):
                    batch_y = batch_y[:, :, self.target]

                outputs = self.model(batch_x)
                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                local_loss = criterion(torch.from_numpy(pred), torch.from_numpy(true))
                train_loss.append(local_loss)

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        my_mse = criterion(torch.from_numpy(preds), torch.from_numpy(trues))
        print('my mse: ', my_mse)

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, MSPE: {:.4f}, RSE: {:.4f}, CORR: {:.4f}".format(mse, mae, rmse, mape, mspe, rse, corr))

        # print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'pred.npy', preds)
        return preds, trues, train_loss

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
                     columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
