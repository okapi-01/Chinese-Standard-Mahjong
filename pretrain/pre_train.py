import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_dataset import MahjongGBDataset, MahjongGBDataset_Allload
from torch.utils.data import DataLoader
<<<<<<< HEAD
from scripts.model import CNNModel, TransformerModel, TransformerMultiHeadModel
=======
from scripts.model import *
>>>>>>> d9d6000579cd65624e1177dbf46c18129b045278
import torch.nn.functional as F
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train the model')
    parser.add_argument('--logdir', type=str, default='ckpt/', help='Directory to save the model checkpoints')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--save_interval', type=int, default=20, help='Interval to save the model checkpoints')
    parser.add_argument('--splitratio', type=float, default=0.9, help='Split ratio for training and validation datasets')
    parser.add_argument('--timestamp', type=str, default=None, help='Number of timesteps to consider in the dataset')
    parser.add_argument('--data', type=str, default='pretrain/data', help='Path to the Mahjong GB data file')
    parser.add_argument('--model', type=str, default='CNN2', help='Model to be trained')
    args = parser.parse_args()

    # trainDataset = MahjongGBDataset(0, args.splitratio, True, args)
    # validateDataset = MahjongGBDataset(args.splitratio, 1, False, args)
    trainDataset = MahjongGBDataset_Allload(0, args.splitratio, True, args)
    validateDataset = MahjongGBDataset_Allload(args.splitratio, 1, False, args)
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False)
<<<<<<< HEAD
    #model = CNNModel().to('cuda')
    model = TransformerMultiHeadModel(dropout=0.1).to('cuda')
=======
    if args.model == 'CNN':
        model = CNNModel().to('cuda')
    elif args.model == 'CNN2':
        model = PreHandsModel().to('cuda')
>>>>>>> d9d6000579cd65624e1177dbf46c18129b045278
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    if args.timestamp is not None:
        timestamp = args.timestamp
        args.logdir = os.path.join(args.logdir, timestamp)
        # Find the highest epoch model file in the logdir and load it
        model_files = [f for f in os.listdir(args.logdir) if f.endswith('.pt')]
        if model_files:
            max_epoch = max([int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()])
            model_path = os.path.join(args.logdir, f"{max_epoch}.pt")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print(f"Loaded model from {model_path}")
        else:
            print("No model checkpoint found in the directory.")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.logdir = os.path.join(args.logdir, timestamp)
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)
        
        
    # Train and validate
    for e in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        # t3 = time.time()
        for d in tqdm(loader, desc=f'Epoch {e+1}', unit='batch'):
            # t1 = time.time()
            # data_load_time = t1 - t3
            # print('Data load time:', data_load_time)

            #input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            if args.model == 'CNN':
                input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
                #print(input_dict["observation"].shape)
                logits, _ = model(input_dict)
                loss = F.cross_entropy(logits, d[2].long().cuda())
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # t3 = time.time()
                # compute_time = t3 - t1
                # print('Compute time:', compute_time)
            elif args.model == 'CNN2':
                input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda(), 'oppo_hands': d[3].cuda()}
                #print(input_dict["observation"].shape)
                logits, _ = model(input_dict, mode='decide')
                pred = model(input_dict, mode='predict')
                pred_target = input_dict['oppo_hands'].reshape((-1,3,4,36)).sum(axis=2).float()
                pred_target = pred_target[...,0:34].reshape((-1,102))
                loss = F.cross_entropy(logits, d[2].long().cuda())
                pred_loss = torch.mean(F.mse_loss(pred, pred_target))
                total_loss = loss + pred_loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # t3 = time.time()
                # compute_time = t3 - t1
                # print('Compute time:', compute_time)
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar('Loss/Train', avg_loss, e + 1)

        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            #input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            if args.model == 'CNN':
                input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
                with torch.no_grad():
                    logits, _ = model(input_dict)
                    pred = logits.argmax(dim = 1)
                    correct += torch.eq(pred, d[2].cuda()).sum().item()
            elif args.model == 'CNN2':
                input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda(), 'oppo_hands': d[3].cuda()}
                with torch.no_grad():
                    logits, _ = model(input_dict, mode='decide')
                    pred = logits.argmax(dim = 1)
                    correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        writer.add_scalar('Accuracy/Validate', acc, e + 1)
        print('Epoch', e + 1, 'Validate acc:', acc)

        if (e + 1) % args.save_interval == 0:
            logdir = args.logdir
            os.makedirs(logdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(logdir, f'{e+1}.pt'))
            print('Saving model to', os.path.join(logdir, f'{e+1}.pt'))