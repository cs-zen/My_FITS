import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from main import Exp_Main

def predict_and_plot():
    # 设置参数
    class Args:
        pass
    args = Args()
    
    # basic config
    args.is_training = 0  # 设置为测试模式
    args.model_id = 'test'
    args.model = 'FITS'
    
    # data loader
    args.data = 'ETTh1'
    args.root_path = '../dataset/ETT/'
    args.data_path = 'ETTh1.csv'
    args.features = 'S'  # 单变量
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    
    # forecasting task
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    
    # model config
    args.individual = False
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.cut_freq = 25
    args.base_T = 24
    args.H_order = 2
    
    # optimization
    args.num_workers = 0  # Windows系统设为0
    args.itr = 1
    args.train_epochs = 10
    args.batch_size = 32
    args.patience = 3
    args.learning_rate = 0.0005
    args.des = 'test'
    args.lradj = 'type3'
    
    # GPU
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'
    
    # FITS特有参数
    args.train_mode = 0
    
    # seed
    args.seed = 2021
    
    # 设置GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # 设置cut_freq
    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
    
    # 固定随机种子
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 处理多GPU
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # 设置setting以找到正确的模型路径
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
    
    print(f"使用配置: {setting}")
    print(f"模型路径: {os.path.join('./checkpoints/' + setting, 'checkpoint.pth')}")
    
    # 初始化实验对象
    exp = Exp_Main(args)
    
    # 运行测试并加载训练好的模型
    print('>>>>>>>开始测试并加载模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)
    
    # 额外保存一个汇总的预测曲线
    save_prediction_plot(args, setting)
    
    print('预测和绘图完成！')

def save_prediction_plot(args, setting):
    """保存一个汇总的预测曲线"""
    try:
        # 加载保存的预测结果
        results_dir = f'./results/{setting}'
        pred_file = os.path.join(results_dir, 'pred.npy')
        true_file = os.path.join(results_dir, 'true.npy')
        
        if os.path.exists(pred_file) and os.path.exists(true_file):
            pred = np.load(pred_file)
            true = np.load(true_file)
            
            # 绘制前500个点的预测和真实值对比
            plt.figure(figsize=(20, 8))
            plt.plot(true[:500, -1], label='True Values')
            plt.plot(pred[:500, -1], label='Predicted Values')
            plt.title('FITS Model Predictions vs True Values')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图片
            save_path = os.path.join(results_dir, 'forecast_summary.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'汇总预测曲线已保存至: {save_path}')
            
            # 也保存到根目录results文件夹
            root_results_dir = './results'
            if not os.path.exists(root_results_dir):
                os.makedirs(root_results_dir)
            root_save_path = os.path.join(root_results_dir, 'forecast.png')
            plt.savefig(root_save_path, dpi=300, bbox_inches='tight')
            print(f'预测曲线也已保存至: {root_save_path}')
            
            plt.close()
        else:
            print(f'警告: 找不到预测结果文件 {pred_file} 或 {true_file}')
    except Exception as e:
        print(f'保存预测曲线时出错: {str(e)}')

if __name__ == '__main__':
    predict_and_plot()