import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

def train_std(modelfns,model_config,dataroot,dataset,train_model,data_config,train_config,optim_para,optimizer):
    for modelfn in modelfns:
        train_dataset = dataset(dataroot,data_config['datasize'],mode='train')
        val_dataset = dataset(dataroot,data_config['datasize'],mode='val')
        train_loader = DataLoader(train_dataset,batch_size=data_config['batch_size'],shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=data_config['batch_size'])
        model = train_model(**model_config)
        train(model=model,modelfn=modelfn,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,optim_para=optim_para,**train_config)
    return

def train(model,modelfn,train_loader,val_loader,loss_func,metric_func,optim_para
          ,device='cuda:0',epochs=100,val_mode='max',optimizer=torch.optim.Adam):
    assert val_mode in ['max','min'] 
    model.to(device)
    optimizer = optimizer(model.parameters(), **optim_para)

    if val_mode == 'max':
        best_val_score = 0.0
    else:
        best_val_score = 1e6
    early_stop = 0
    best_epoch = 0

    logfn = modelfn.replace('pth','txt').replace('model','log')
    with open(logfn,'w') as f:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_cls_loss = 0.0
            train_score = 0.0
            for data in tqdm(train_loader):
                data = [t.to(device) for t in data]
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(data,output)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_score += metric_func(data,output)

            train_loss /= len(train_loader)
            train_score /= len(train_loader)
            
            val_loss,val_score = test(model,val_loader,device,loss_func,metric_func)

            print('epoch: %d, train_loss: %.6f, train_score: %.6f, val_loss: %.6f, val_score: %.6f' % (epoch, train_loss, train_score, val_loss, val_score))
            f.write('epoch: %d, train_loss: %.6f, train_score: %.6f, val_loss: %.6f, val_score: %.6f\n' % (epoch, train_loss, train_score, val_loss, val_score))

            #early stopping
            if val_mode == 'max':
                res = val_score > best_val_score
            else:
                res = val_score < best_val_score

            if res:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), modelfn)
                print('epoch: %d, best_val_score: %.6f' % (epoch, best_val_score))
                print('save model')
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > 10 and best_val_score>0.1:
                    print('early stop, best epoch is %d, score score is %s' % (best_epoch,best_val_score))
                    break
    return

def test(model,dataloader,device,critetion,mertic_func):
    model.eval()
    val_loss = 0.0
    val_score = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = [t.to(device) for t in data]
            output = model(data)
            loss = critetion(data,output)
            val_loss += loss.item()
            val_score += mertic_func(data,output)
    val_loss /= len(dataloader)
    val_score /= len(dataloader)
    return val_loss,val_score

def evaluate(model,dataloader,device,mertic_func):
    model.to(device)
    model.eval()
    val_score = 0.0
    with torch.no_grad():
        for data  in tqdm(dataloader):
            data = [t.to(device) for t in data]
            output = model(data)
            val_score += mertic_func(data,output)
    val_score /= len(dataloader)
    return val_score

def inference(model,dataloader,device):
    model.eval()
    output = []
    with torch.no_grad():
        for data  in tqdm(dataloader):
            data = [t.to(device) for t in data]
            output.append(model(data))
    output = torch.cat(output,0)
    return output