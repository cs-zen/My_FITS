import argparse
import os
import torch
import random
import numpy as np
import time
import warnings
from MyFITS import Model
from torch import optim
import torch.nn as nn
warnings.filterwarnings('ignore')

# 基础实验类
class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(self.args.gpu)) if self.args.use_gpu and torch.cuda.is_available() else torch.device('cpu')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model = self._build_model()
    
    def _build_model(self):
        model = Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        model = model.to(self.device)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        print('!!!!!!!!!!!!!!learning rate!!!!!!!!!!!!!!!')
        print(self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型调用
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting, ft=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型调用
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型调用
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                # 使用extend而不是append来避免形状不均匀的问题
                for p in pred:
                    preds.append(p)
                for t in true:
                    trues.append(t)
                for x in batch_x.detach().cpu().numpy():
                    inputx.append(x)
                
                if i % 20 == 0:
                    try:
                        import matplotlib.pyplot as plt
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        plt.figure(figsize=(15, 5))
                        plt.plot(gt, label='True')
                        plt.plot(pd, label='Predicted')
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, str(i) + '.pdf'))
                        plt.close()
                    except:
                        print('Visualization error')

        # 转换为numpy数组并确保形状一致
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        # 重塑为 [样本数, 序列长度, 特征数]
        preds = preds.reshape(-1, self.args.pred_len, self.args.c_out)
        trues = trues.reshape(-1, self.args.pred_len, self.args.c_out)
        inputx = inputx.reshape(-1, self.args.seq_len, self.args.enc_in)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        # 生成预测图像与真实图像的对比
        self.plot_results(preds, trues, setting)
        
        return
    
    # 添加predict方法
    
    def plot_results(self, preds, trues, setting):
        """
        绘制预测结果与真实值的对比图
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 绘制前几个样本的预测结果
            num_samples = min(5, len(preds))
            for i in range(num_samples):
                plt.figure(figsize=(15, 7))
                
                # 绘制预测值和真实值
                plt.plot(trues[i, :, -1], label='真实值', linewidth=2)
                plt.plot(preds[i, :, -1], label='预测值', linewidth=2, linestyle='--')
                
                # 添加网格和标签
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('时间步', fontsize=14)
                plt.ylabel('值', fontsize=14)
                plt.title(f'预测结果对比 (样本 {i+1})', fontsize=16)
                plt.legend(fontsize=12)
                
                # 保存图像为PNG格式
                plt.savefig(os.path.join(folder_path, f'forecast_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 绘制所有样本的误差分布
            errors = np.abs(preds - trues)
            plt.figure(figsize=(12, 6))
            plt.hist(errors.flatten(), bins=50, alpha=0.7, color='blue')
            plt.xlabel('绝对误差', fontsize=14)
            plt.ylabel('频率', fontsize=14)
            plt.title('预测误差分布', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(folder_path, 'error_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'预测图像已保存到 {folder_path}')
            
        except Exception as e:
            print(f'可视化失败: {e}')
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型调用
                outputs, _ = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

# 数据集类
class Dataset_ETT_hour:
    def __init__(self, config, root_path, data_path, flag='train', size=None, features='M', target='OT', timeenc=0, freq='h'):
        self.config = config
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        
        # 初始化参数
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # 加载数据
        self.__read_data__()
    
    def __read_data__(self):
        import pandas as pd
        # 读取CSV文件
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 处理时间特征
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        
        # 分割数据集
        train_size = int(len(df_raw) * 0.7)
        val_size = int(len(df_raw) * 0.2)
        test_size = len(df_raw) - train_size - val_size
        
        if self.flag == 'train':
            self.data = df_raw[:train_size]
        elif self.flag == 'val':
            self.data = df_raw[train_size:train_size+val_size]
        elif self.flag == 'test':
            self.data = df_raw[train_size+val_size:]
        
        # 根据features参数决定加载单变量还是多变量
        if self.features == 'S':
            # 单变量模式，只提取目标特征
            self.data = self.data[[self.target]].values
        else:
            # 多变量模式，提取所有特征（除了date）
            self.data = self.data[cols].values
        
        # 标准化
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.data = (self.data - self.mean) / (self.std + 1e-8)  # 避免除零错误
    
    def __getitem__(self, index):
        # 获取数据切片
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 确保不越界
        if r_end > len(self.data):
            # 填充
            r_end = len(self.data)
            s_end = r_end - self.label_len - self.pred_len
            s_begin = s_end - self.seq_len
        
        # 准备输入和标签
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return seq_x, seq_y, np.array([0]), np.array([0])  # 返回x, y, 时间标记(占位)
    
    def __len__(self):
        # 计算样本数量
        return len(self.data) - self.seq_len - self.pred_len + 1

# 数据提供器
def data_provider(config, flag):
    from torch.utils.data import DataLoader
    Data = Dataset_ETT_hour
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = config.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config.batch_size
    
    data_set = Data(
        config=config,
        root_path=config.root_path,
        data_path=config.data_path,
        flag=flag,
        size=[config.seq_len, config.label_len, config.pred_len],
        features=config.features,
        target=config.target,
        timeenc=0,
        freq=config.freq
    )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader

# 计算指标
def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / (true + 1e-5)))
    mspe = np.mean(np.square((pred - true) / (true + 1e-5)))
    rse = mse / np.var(true)
    corr = np.corrcoef(pred.reshape(-1), true.reshape(-1))[0, 1]
    return mae, mse, rmse, mape, mspe, rse, corr

# 早停类
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
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
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# 调整学习率
def adjust_learning_rate(optimizer, epoch, args):
    # 学习率调整策略
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: args.learning_rate * 0.5, 4: args.learning_rate * 0.1, 6: args.learning_rate * 0.05, 8: args.learning_rate * 0.01}
    elif args.lradj == 'type3':
        # 更灵活的学习率调度
        lr = args.learning_rate
        if epoch >= 3:
            lr = args.learning_rate * 0.5
        if epoch >= 6:
            lr = args.learning_rate * 0.1
        if epoch >= 9:
            lr = args.learning_rate * 0.05
        if epoch >= 12:
            lr = args.learning_rate * 0.01
        lr_adjust = {epoch: lr}
    else:
        lr_adjust = {epoch: args.learning_rate}
        
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer

def main():
    class Args:
        pass
    args = Args()
    
    # basic config
    args.is_training = 1  # 训练状态：1表示训练，0表示测试
    args.model_id = 'ETTh1_360_96'  # 模型ID
    args.model = 'FITS'  # 模型名称
    
    # data loader
    args.data = 'ETTh1'  # 数据集类型
    args.root_path = '../dataset/ETT/'  # 数据文件的根路径
    args.data_path = 'ETTh1.csv'  # 数据文件名
    args.features = 'M'  # 预测任务，选项：[M, S, MS]，M表示多变量
    args.target = 'OT'  # 目标特征
    args.freq = 'h'  # 时间特征编码的频率
    args.checkpoints = './checkpoints/'  # 模型检查点的位置
    
    # forecasting task
    args.seq_len = 360  # 输入序列长度
    args.label_len = 48  # 起始token长度
    args.pred_len = 96  # 预测序列长度
    
    # model config
    args.individual = False  # 是否为每个变量单独使用一个线性层
    args.enc_in = 7  # 编码器输入大小
    args.dec_in = 7  # 解码器输入大小
    args.c_out = 7  # 输出大小
    args.cut_freq = 0  # 截断频率
    args.base_T = 24  # 基本周期
    args.H_order = 6  # 谐波阶数
    
    # optimization
    args.num_workers = 0  # 数据加载器的工作线程数（Windows系统设为0避免多进程问题）
    args.itr = 1  # 实验次数
    args.train_epochs = 10  # 训练轮数
    args.batch_size = 64  # 训练输入数据的批次大小
    args.patience = 20  # 早停耐心值
    args.learning_rate = 0.0005  # 优化器学习率
    args.des = 'Exp'  # 实验描述
    args.lradj = 'type3'  # 学习率调整方式
    
    # GPU
    args.use_gpu = torch.cuda.is_available()  # 是否使用GPU
    args.gpu = 0  # GPU编号
    args.use_multi_gpu = False  # 是否使用多GPU
    args.devices = '0,1,2,3'  # 多GPU的设备ID
    
    # FITS特有参数
    args.train_mode = 1  # 训练模式）：0表示全序列训练，1表示仅预测部分训练，2表示微调
    
    # seed
    args.seed = 514  # 随机种子
    
    # 设置GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # 设置cut_freq
    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
    
    # 设置随机种子
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 处理多GPU
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)
    
    # 执行实验
    Exp = Exp_Main
    
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_H{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.H_order, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.train_mode == 0:
                exp.train(setting, ft=False) # train on xy
            elif args.train_mode == 1:
                exp.train(setting, ft=True) # train on y
            elif args.train_mode == 2:
                exp.train(setting, ft=False)
                exp.train(setting, ft=True) # finetune

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_H{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.H_order, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"发生错误: {type(e).__name__}: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc()