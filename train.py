import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import time

from dataloader import SH_Dataset
from model import Sam_Head

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, info, name = 'test'):
    state = {
            'epoch': info['epoch'],
            'state_dict': model.state_dict(),
            }
    torch.save(state, os.path.join(info['base_root'], name+'.pth.tar'))

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
    epoch = state_dict['epoch']+1
    return model,epoch

def multi_save(best_mae_list,best_epoch_list,mae,epoch,base_dir,train_name,data_type,model,config):
    if len(best_mae_list) == 0:
        best_mae_list.append(mae)
        best_epoch_list.append(epoch)
        save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'best_mae_'+data_type+'_0')
    else:    
        for i in range(len(best_mae_list)):
            if mae < best_mae_list[i]:
                if len(best_mae_list) < config['save_num']:
                    for j in range(len(best_mae_list)-1,i-1,-1):
                        os.rename(os.path.join(base_dir,'run','train',train_name,'best_mae_')+data_type+'_'+str(j)+'.pth.tar',
                                os.path.join(base_dir,'run','train',train_name,'best_mae_')+data_type+'_'+str(j+1)+'.pth.tar')
                    best_mae_list.insert(i,mae)
                    best_epoch_list.insert(i,epoch)
                    save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'best_mae_'+data_type+'_'+str(i))
                    break
                else:
                    os.remove(os.path.join(base_dir,'run','train',train_name,'best_mae_')+data_type+'_'+str(len(best_mae_list)-1)+'.pth.tar')
                    best_mae_list.pop()
                    best_epoch_list.pop()
                    for j in range(len(best_mae_list)-1,i-1,-1):
                        os.rename(os.path.join(base_dir,'run','train',train_name,'best_mae_')+data_type+'_'+str(j)+'.pth.tar',
                                os.path.join(base_dir,'run','train',train_name,'best_mae_')+data_type+'_'+str(j+1)+'.pth.tar')
                    best_mae_list.insert(i,mae)
                    best_epoch_list.insert(i,epoch)
                    save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'best_mae_'+data_type+'_'+str(i))
                    break
            elif len(best_mae_list) < config['save_num'] and i == len(best_mae_list)-1:
                best_mae_list.append(mae)
                best_epoch_list.append(epoch)
                save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'best_mae_'+data_type+'_'+str(i+1))
                break
    return best_mae_list, best_epoch_list

def train_one_epoch(model, train_loader, epoch, optimizer, scheduler, dev):
    model.train()
    losses = AverageMeter()
    criterion = nn.MSELoss(reduction='sum')

    with tqdm(train_loader) as tbar:
        for embeddings, masks, name in tbar:
            tbar.set_description("epoch {}".format(epoch))
            embeddings = embeddings.to(torch.float32).to(dev)
            masks = masks.to(dev)

            optimizer.zero_grad()
            outputs = model(embeddings)

            loss = criterion(outputs.float(), masks.float())
            '''
            loss = 0
            threshold = 0.5
            right_map_pred_tmp1 = torch.where((outputs>=threshold)&(masks>=threshold),outputs,0)
            right_map_pred_tmp2 = torch.where((outputs<threshold)&(masks<threshold),outputs,0)
            right_map_pred = right_map_pred_tmp1+right_map_pred_tmp2
            right_map_gt_tmp1 = torch.where((outputs>=threshold)&(masks>=threshold),masks,0)
            right_map_gt_tmp2 = torch.where((outputs<threshold)&(masks<threshold),masks,0)
            right_map_gt = right_map_gt_tmp1+right_map_gt_tmp2
            false_map_pred_tmp1 = torch.where((outputs>=threshold)&(masks<threshold),outputs,0)
            false_map_pred_tmp2 = torch.where((outputs<threshold)&(masks>=threshold),outputs,0)
            false_map_pred = false_map_pred_tmp1+false_map_pred_tmp2
            false_map_gt_tmp1 = torch.where((outputs>=threshold)&(masks<threshold),masks,0)
            false_map_gt_tmp2 = torch.where((outputs<threshold)&(masks>=threshold),masks,0)
            false_map_gt = false_map_gt_tmp1+false_map_gt_tmp2
            loss = criterion(right_map_pred.float(),right_map_gt.float())+\
                    criterion(false_map_pred.float(),false_map_gt.float())*3
            '''
            losses.update(loss.item(), embeddings.shape[0])
            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss="{:.4f}".format(losses.avg), cur_loss="{:.4f}".format(loss))

            del embeddings, masks, outputs, loss

    scheduler.step()

    return losses


if __name__ == '__main__':
    config = {'max_epoch':300,
              'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              'lr_start':1e-4,
              'lr_finish':1e-5,
              'save_num':3,
              }
    
    base_dir = sys.path[0]
    data_dir = os.path.join(base_dir, 'data')
    train_dataset = SH_Dataset(data_dir, split = 'trainval')
    test_dataset = SH_Dataset(data_dir, split = 'test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = Sam_Head().to(config['dev'])
    #model_path = os.path.join(base_dir,'run','train','2024-04-08-23-30-12','best_mae_trainval_0.pth.tar')
    #model,_ = load_checkpoint(model,model_path)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, config['lr_start'])
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'], eta_min=config['lr_finish'])

    train_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.mkdir(os.path.join(base_dir, 'run', 'train',train_name))

    best_losses = []
    best_epochs = []

    outputfile = open(os.path.join(base_dir,'run','train',train_name,"log.txt"), 'w')
    outputfile.close()
    for epoch in range(config['max_epoch']):
        outputfile = open(os.path.join(base_dir,'run','train',train_name,"log.txt"), 'a')
        train_loss = train_one_epoch(model, train_loader, epoch, optimizer, schedular, config['dev'])

        text = 'epoch: ' + str(epoch) + ' train_loss: ' + str(train_loss.avg) + ' lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr'])
        save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name)},'last')

        best_losses, best_epochs = multi_save(best_losses,best_epochs,train_loss.avg,epoch,base_dir,train_name,'trainval',model,config)

        print(text)
        print(text,file=outputfile)
        outputfile.close()