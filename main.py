import random
import torch.nn.functional as F
import torch.nn as nn
from model import *

from utils import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score,f1_score
from sklearn import metrics
from torch.utils.data import DataLoader

SEED=0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

def train(model, device, loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, (data1, data2, y) in enumerate(loader_train):
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(loader_train.dataset),
                                                                           100. * batch_idx / len(loader_train),
                                                                           loss.item()))


def predicting(model, device, loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for data1, data2, y in loader_test:
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1,1)), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()







modeling = AttenSyn
print(modeling.__name__)


TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

device = torch.device('cuda:2')


dataset = MyDataset()

lenth = len(dataset)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)


random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    data_train = dataset.get_data(train_num)
    data_test = dataset.get_data(test_num)
    loader_train = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)


    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    file_result = './result/fold' + str(i) +  '.csv'
    AUCs = ('Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,Precision,F1')
    with open(file_result, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, loader_train,  optimizer, epoch + 1)
        T, S, Y = predicting(model, device, loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)
        precision = precision_score(T, Y)
        F1 = f1_score(T, Y)
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall, precision, F1]

        # save data

        if best_auc < AUC:
            best_auc = AUC
            save_AUCs(AUCs, file_result)


        print('best_auc', best_auc)
