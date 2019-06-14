"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd



def tester(feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network

    

    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    tgt_correct = 0.0
    src_correct = 0.0
    num1=0

    for batch_idx, sdata in enumerate(source_dataloader):
        if(batch_idx> 100):
            break

        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        input1, label1, filenames = sdata
        
        input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
        src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
        
        feature=feature_extractor(input1)
        feature=feature.view(-1,2048)
        output1 = class_classifier(feature)

        pred1 = output1.data.max(1, keepdim = True)[1]
        
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()
        num1+=64
        
        src_preds = domain_classifier(feature, constant)
        src_preds = src_preds.data.max(1, keepdim= True)[1]
        src_correct += src_preds.eq(src_labels.data.view_as(src_preds)).cpu().sum()
    acc1=100. * float(source_correct) / num1
    #len(source_dataloader.dataset)
    print('\nSource Accuracy: {}/{} ({:.4f}%)\n'.format(source_correct, num1,acc1 ))          
          
    num2=0  
    for batch_idx, tdata in enumerate(target_dataloader):
        if(batch_idx> 100):
            break

        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2,filenames = tdata
        
        input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        
        feature=feature_extractor(input2)
        feature=feature.view(-1,2048)
        output2 = class_classifier(feature)
        pred2 = output2.data.max(1, keepdim=True)[1]
        
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()
        num2+=64
        tgt_preds = domain_classifier(feature, constant)
        tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()
    
    acc2=100. * float(target_correct) / num2
    #len(target_dataloader.dataset)
    print("\nTarget Accuracy: {}/{} ({:.4f}%)\n".format(target_correct, num2, acc2))
    
              
    domain_correct = tgt_correct + src_correct
    


    print('Domain Accuracy: {}/{} ({:.4f}%)\n'.format(domain_correct, num1+num2,
        100. * float(domain_correct) / (num1+num2)))

    return (acc1,acc2)



