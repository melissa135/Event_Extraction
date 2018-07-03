import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from word_set import Word_Set
from torch.autograd import Variable
from define_net import Char_CNN_pretrain
from dataloader_modified import DataLoader


def testset_loss(dataset,net):

    loader = DataLoader(dataset,batch_size=1,num_workers=4)

    all_loss = 0.0
    criterion = nn.NLLLoss() # weight=loss_weight

    for i,batch in enumerate(loader,0):
        loss = 0
            
        for data in batch: # due to we have modified the defination of batch, the batch here is a list
            inputs,targets = data
            inputs,targets = Variable(inputs),Variable(targets)
	    outputs = net(inputs)   
	    loss = loss + criterion(outputs,targets)

        all_loss += loss.data[0]

    return all_loss/i


if __name__ == '__main__':
	
    path_ = os.path.abspath('.')

    trainset = Word_Set(path_+'/table_train/',new_dict=True)
    char_dim = trainset.get_char_dim()
    event_dim = trainset.get_event_dim()
    print 'Total %d samples.' % trainset.__len__()
    print 'Char dimension : ' , char_dim
    print 'Event dimension : ' , event_dim
	
    # the length of samples here are different, so we can't directly use DataLoader provided by PyTorch 
    trainloader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=2)

    net = Char_CNN_pretrain(char_dim,event_dim)
    #net.load_state_dict(torch.load(path_+'/net.pth'))
    print net

    testset = Word_Set(path_+'/table_test/',new_dict=False)
    '''
    loss_weight = [10 for i in range(0,43)]
    loss_weight[26] = 1
    loss_weight = torch.FloatTensor(class_weight)
    '''
    criterion = nn.NLLLoss() # weight=loss_weight

    optimizer = optim.Adam(net.parameters())

    for epoch in range(20): #

        running_loss = 0.0
        
        for i,batch in enumerate(trainloader,0):

            loss = 0
            optimizer.zero_grad()

            for data in batch: # due to we have modified the defination of batch, the batch here is a list
                inputs,targets = data
                inputs,targets = Variable(inputs),Variable(targets)
		outputs = net(inputs)
		#print outputs,targets
		loss = loss + criterion(outputs,targets)

	    loss.backward()
	    optimizer.step()

            running_loss += loss.data[0]
            if i%100 == 99:
                print('[%d, %3d] loss: %.5f' % (epoch+1,i+1,running_loss/6400)) # step is 10 and batch_size is 16
                running_loss = 0.0

        test_loss = testset_loss(testset,net)
        print('[%d ] test loss: %.5f' % (epoch+1,test_loss))

    print('Finished Training')
    torch.save(net.state_dict(),path_+'/char_rnn_pretrain.pth')
