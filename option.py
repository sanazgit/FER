import argparse
import time
import datetime

def init():

  parser = argparse.ArgumentParser(description="PyTorch")
  now = datetime.datetime.now()
  time_str = now.strftime("[%m-%d]-[%H-%M]-")

  parser.add_argument('--data', type=str, default='/content/RAFDB/dataset/')
  parser.add_argument('--data_label', type=str, default='/content/drive/MyDrive/Test_FER/Dataset/RAFDB/data_label.txt')
  parser.add_argument('--land_marks', type=str, default='/content/drive/MyDrive/Test_FER/Dataset/RAFDB/land_marks.npy')
  parser.add_argument('--checkpoint_path', type=str, default='/content/drive/MyDrive/Test_FER/checkpoint_cnn/RAFDB/' + time_str +  'model.pth.tar')
  parser.add_argument('--best_checkpoint_path', type=str, default='/content/drive/MyDrive/Test_FER/checkpoint_cnn/RAFDB/' +time_str + 'model_best.pth.tar')
  parser.add_argument('--log_path', type=str, default='/content/drive/MyDrive/Test_FER/log/RAFDB/') #..save_path in ampnet
  parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers') #..num_workers in ampnet
  parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
  parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
  parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
  parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
  parser.add_argument('--beta1',default=0.5,type=float,metavar='M', help='hyper-parameter ')
  parser.add_argument('--af', '--adjust-freq', default=20, type=int, metavar='N', help='adjust learning rate frequency') #..print_freq in ampnet
  parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
  parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
  parser.add_argument('--resume', default=False, type=str, metavar='PATH', help='path to checkpoint')
  parser.add_argument('--range', default=5, type=int, metavar='N', help='Intercept radius of SSR-Module ')
  parser.add_argument('--dataset', type=str, default='RAF')
  parser.add_argument('--evaluate_path', type=str, default='' + time_str + 'model.pth.tar')
  parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')

  args = parser.parse_args(args=[])
  return args
