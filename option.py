import argparse

def init():

  parser = argparse.ArgumentParser(description="PyTorch")

  parser.add_argument('--data', type=str, default='/content/RAFDB/')
  parser.add_argument('--checkpoint_path', type=str, default='/content/drive/MyDrive/FER/checkpoint_cnn/RAFDB/' + time_str +  'model.pth.tar')
  parser.add_argument('--best_checkpoint_path', type=str, default='/content/drive/MyDrive/FER/checkpoint_cnn/RAFDB/' +time_str + 'model_best.pth.tar')
  parser.add_argument('--log_path', type=str, default='/content/drive/MyDrive/FER/log/RAFDB/' + time_str)
  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
  parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
  parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
  parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
  parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
  # 30 is the best
  parser.add_argument('--af', '--adjust-freq', default=20, type=int, metavar='N', help='adjust learning rate frequency')
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
  parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
  parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
  parser.add_argument('--evaluate_path', type=str, default='' + time_str + 'model.pth.tar')
  parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
  #parser.add_argument('--gpu', default='4', type=str)

  args = parser.parse_args(args=[])
  return args


