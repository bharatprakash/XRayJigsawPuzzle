import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

sys.path.append('Dataset')
from xNet import Network

from TrainingUtils import adjust_learning_rate, compute_accuracy
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser(description='Train Classifier on XRay14')
parser.add_argument('data', type=str, help='Path to XRay14 folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=2, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='xCheckpoints/exp1', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=0, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

dtype = torch.FloatTensor

from xImageLoader import DataLoader

ln = None

def main():
    if args.gpu is not None:
        print('Using GPU %d'%args.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')

    print ('Process number: %d'%(os.getpid()))

    trainpath = args.data
    train_data = DataLoader(trainpath,args.data+'/Data_Entry_2017.csv', 10)

    # trainpath = args.data + "/train_img"
    # train_data = DataLoader(trainpath,args.data+'/train', 10)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_data.weights, train_data.N)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch,
                                            shuffle=False,
                                            num_workers=args.cores,
                                            sampler=sampler)

    valpath = args.data
    val_data = DataLoader(valpath, args.data+'/Data_Entry_2017.csv', 5)

    # valpath = args.data + "/test_img"
    # val_data = DataLoader(valpath, args.data+'/test', 5)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            num_workers=args.cores)
    N = train_data.N

    print(train_data.labelNames)
    print(val_data.labelNames)

    iter_per_epoch = train_data.N/args.batch
    print( 'Images: train %d, validation %d'%(train_data.N,val_data.N))

    # Network initialize
    net = Network(train_data.classes)
    if args.gpu is not None:
        net.cuda()

    ############## Load from checkpoint if exists, otherwise from model ###############
    if args.model is not None:
        net.load(args.model)

    criterion = nn.MultiLabelSoftMarginLoss()
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)

    logger = Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint+'/test')
    writer = SummaryWriter(log_dir=args.checkpoint+'/train')
    writer_test = SummaryWriter(log_dir=args.checkpoint+'/test')

    ############## TESTING ###############
    if args.evaluate:
        test(net,criterion,None,val_loader,0)
        return

    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,train_data.classes))
    print('Checkpoint: '+args.checkpoint)

    # Train the Model
    batch_time, net_time = [], []
    steps = args.iter_start
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        print("------------------------------------------")
        if epoch%5==0 and epoch>0:
            test(net,criterion,writer_test,val_loader,steps,train_data.labelNames)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        print("epoch - " , epoch)
        meanHammingLoss = 0
        hammLosses = []

        precisions = []
        recalls = []
        f1Scores = []
        accuracies = []

        end = time()
        for i, (images, labels) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]

            images = Variable(images).type(dtype)
            labels = Variable(labels).type(dtype)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            #correct += (sample['label'] == preds).sum(1).eq(11).sum()
            L = np.array(labels.cpu().data)
            O = np.array(torch.sigmoid(outputs).cpu().data > 0.5)

            for l, o in zip(L, O):
                hammLosses.append(hamming_loss(l, o))

            LT = L.T
            OT = O.T

            prec = []
            recall = []
            f1 = []
            acc = []

            for lt, ot in zip(LT, OT):
                prec.append(precision_score(lt, ot, average='binary'))
                recall.append(recall_score(lt, ot, average='binary'))
                f1.append(f1_score(lt, ot, average='binary'))
                acc.append(accuracy_score(lt, ot))

            recalls.append(recall)
            precisions.append(prec)
            f1Scores.append(f1)
            accuracies.append(acc)

            if steps%5==0:
                print ('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Mean HL % 2.2f' %(
                            epoch+1, args.epochs, steps,
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,np.mean(hammLosses)))

            if steps%5==0:
                #logger.scalar_summary('hamming_loss', np.mean(hammLosses), steps)
                #logger.scalar_summary('loss', loss, steps)

                writer.add_scalar('hamming_loss', np.mean(hammLosses), steps)
                writer.add_scalar('loss', loss, steps)

                meanPrec = np.mean(precisions, axis=0)
                meanRecall = np.mean(recalls, axis=0)
                meanF1Scores = np.mean(f1Scores, axis=0)
                meanAcc = np.mean(accuracies, axis=0)

                d1 = {}
                for i in range(15):
                    d1[train_data.labelNames[i]] = meanPrec[i]

                d2 = {}
                for i in range(15):
                    d2[train_data.labelNames[i]] = meanRecall[i]

                d3 = {}
                for i in range(15):
                    d3[train_data.labelNames[i]] = meanF1Scores[i]

                d4 = {}
                for i in range(15):
                    d4[train_data.labelNames[i]] = meanAcc[i]

                writer.add_scalars('Precisions', d1, steps)
                writer.add_scalars('Recalls', d2, steps)
                writer.add_scalars('F1Scores', d3, steps)
                writer.add_scalars('Accuracies', d4, steps)

            steps += 1

            if steps%1000==0:
                filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
                net.save(filename)
                print ('Saved: '+args.checkpoint)

            end = time()

        meanHammingLoss = np.mean(hammLosses)
        print("Mean HammingLoss: ", meanHammingLoss)
        meanPrec = np.mean(precisions, axis=0)
        print("Precisions: ", meanPrec)
        meanRecall = np.mean(recalls, axis=0)
        print("Recalls: ", meanRecall)
        meanF1Scores = np.mean(f1Scores, axis=0)
        print("F1Scores: ", meanF1Scores)
        meanAcc = np.mean(accuracies, axis=0)
        print("Accuracies: ", meanAcc)

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test(net,criterion,logger,val_loader,steps,ln):
    print('Evaluating network.......')

    hammLosses = []
    precisions = []
    recalls = []
    f1Scores = []
    accuracies = []

    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = Variable(images).type(dtype)
        labels = Variable(labels).type(dtype)
        if args.gpu is not None:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)

        L = np.array(labels.cpu().data)
        O = np.array(torch.sigmoid(outputs).cpu().data > 0.5)

        # -----------------------METRICS-----------------
        for l, o in zip(L, O):
            hammLosses.append(hamming_loss(l, o))
        LT = L.T
        OT = O.T

        prec = []
        recall = []
        f1 = []
        acc = []

        for lt, ot in zip(LT, OT):
            prec.append(precision_score(lt, ot, average='binary'))
            recall.append(recall_score(lt, ot, average='binary'))
            f1.append(f1_score(lt, ot, average='binary'))
            acc.append(accuracy_score(lt, ot))

        recalls.append(recall)
        precisions.append(prec)
        f1Scores.append(f1)
        accuracies.append(acc)
        # -----------------------METRICS-----------------

    if logger is not None:
        logger.add_scalar('hamming_loss', np.mean(hammLosses), steps)

        meanPrec = np.mean(precisions, axis=0)
        meanRecall = np.mean(recalls, axis=0)
        meanF1Scores = np.mean(f1Scores, axis=0)
        meanAcc = np.mean(accuracies, axis=0)

        d1 = {}
        for i in range(15):
            d1[ln[i]] = meanPrec[i]

        d2 = {}
        for i in range(15):
            d2[ln[i]] = meanRecall[i]

        d3 = {}
        for i in range(15):
            d3[ln[i]] = meanF1Scores[i]

        d4 = {}
        for i in range(15):
            d4[ln[i]] = meanAcc[i]

        logger.add_scalars('Precisions', d1, steps)
        logger.add_scalars('Recalls', d2, steps)
        logger.add_scalars('F1Scores', d3, steps)
        logger.add_scalars('Accuracies', d4, steps)


    print('TESTING: %d), Hamming Loss %.2f%%' %(steps,np.mean(hammLosses)))
    meanPrec = np.mean(precisions, axis=0)
    print("Precisions: ", meanPrec)
    meanRecall = np.mean(recalls, axis=0)
    print("Recalls: ", meanRecall)
    meanF1Scores = np.mean(f1Scores, axis=0)
    print("F1Scores: ", meanF1Scores)
    meanAcc = np.mean(accuracies, axis=0)
    print("Accuracies: ", meanAcc)
    print("------------------------------------------")

    net.train()

if __name__ == "__main__":
    main()
