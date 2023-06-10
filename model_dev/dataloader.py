import os
import warnings
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, size, flag='train', data_path=None, train_only=False, return_date=False, data_segment=None, stocks=None):

        self.stocks = stocks
        self.return_date = return_date
        self.data_segment = data_segment
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
        if not ',' in self.data_path:
            df_data = pd.read_csv(os.path.join(self.root_path, self.data_path))
        else:
            files = self.data_path.split(',')
            # read all the files as df and stack them vertically
            df_data = pd.concat([pd.read_csv(os.path.join(self.root_path, f)) for f in files], axis=0)
            # drop any column containing nan
            df_data.dropna(axis=1, inplace=True)
            # dump this combined data
            # df_data.to_csv(os.path.join(self.root_path, self.data_path), index=False)
            # print(df_data.head())
        

        if self.data_segment is not None:
            df_data = df_data[int(len(df_data) * self.data_segment[0]):int(len(df_data) * self.data_segment[1])]
        
        if self.stocks is not None:
            # except date columns select column with self.stocks index
            df_data = df_data.iloc[:, [0] + [i+1 for i in self.stocks]]

        print("Original data shape:", df_data.shape)    

        date_col = list(df_data['date'])
        # print(df_data.head())

        # cerate a new index mapping
        self.new_index = []
        for i in range(len(date_col)-self.seq_len-self.pred_len):
            # take out the current date
            date = date_col[i].split(' ')[0]
            # add args.seq_len + args.pred_len numutes to the current date
            last_i = i + self.seq_len + self.pred_len
            date_end = date_col[last_i].split(' ')[0]
            if date == date_end:
                self.new_index.append(i)

        print("Data shape after inter-day: ", len(self.new_index))      

        df_data.drop(columns=['date'], inplace=True)

        num_train = int(len(self.new_index) * (0.7 if not self.train_only else 1))
        num_test = int(len(self.new_index) * 0.2)
        num_vali = len(self.new_index) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(self.new_index) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(self.new_index)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        num_train_norm = int(len(df_data) * (0.7 if not self.train_only else 1))
        num_test_norm = int(len(df_data) * 0.2)
        num_vali_norm = len(df_data) - num_train - num_test

        border1s_norm = [0, num_train_norm - self.seq_len, len(df_data) - num_test_norm - self.seq_len]
        border2s_norm = [num_train_norm, num_train_norm + num_vali_norm, len(df_data)]

        
        train_data = df_data[border1s_norm[0]:border2s_norm[0]]
        self.scaler.fit(train_data.values)
        self.data = self.scaler.transform(df_data.values)

        # self.data = df_data.values
        self.date_col = date_col

        self.data_indx = self.new_index[border1:border2]
        # self.data_y = self.new_index[border1:border2]
        # self.date_col = date_col[border1:border2]


    def __getitem__(self, index):
        
        index = self.data_indx[index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data[s_begin:s_end]  # seq_len*stocks
        seq_y = self.data[r_begin:r_end]

        # iterate over the stocks to check if the price is constant and drop such stocks
        # is_constant = detect_constant_price(seq_x[:,2], duration=5)

        if self.return_date:
            return seq_x, seq_y, [self.date_col[s_begin], self.date_col[s_end-1], self.date_col[r_begin], self.date_col[r_end-1]]
        else:
            return seq_x, seq_y

    def __len__(self):
        return len(self.data_indx) - self.seq_len - self.pred_len + 1

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
                    size=[args.seq_len, args.pred_len], train_only=train_only, return_date=return_date,
                    data_segment=args.data_segment, stocks=args.stocks)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers,
                             drop_last=drop_last)
    return data_set, data_loader
