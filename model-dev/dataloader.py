import os
import warnings
import pandas as pd
from utills import detect_constant_price
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, size, flag='train', data_path=None, train_only=False, return_date=False):

        self.return_date = return_date
        self.seq_len, self.pred_len = size[0], size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[flag]
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()
        df_data = pd.read_csv(os.path.join(self.root_path, self.data_path))
        date_col = list(df_data['datetime'])
        df_data.drop(columns=['datetime'], inplace=True)

        num_train = int(len(df_data) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.date_col = date_col[border1:border2]

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]  # seq_len*stocks
        seq_y = self.data_y[r_begin:r_end]

        # iterate over the stocks to check if the price is constant and drop such stocks
        # is_constant = detect_constant_price(seq_x[:,2], duration=5)

        if self.return_date:
            return seq_x, seq_y, [self.date_col[s_begin], self.date_col[s_end-1], self.date_col[r_begin], self.date_col[r_end-1]]
        else:
            return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(args, flag, return_date=False):

    Data = Dataset_Custom
    train_only = args.train_only
    batch_size = args.batch_size
    drop_last = True

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = Data(root_path=args.root_path, data_path=args.data_path, flag=flag,
                    size=[args.seq_len, args.pred_len], train_only=train_only, return_date=return_date)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    return data_set, data_loader
