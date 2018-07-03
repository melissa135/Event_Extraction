import os.path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

if __name__ == '__main__':

    path_ = os.path.abspath('.')

    fname = path_ + '/loss_sequence'
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    Y_train = []
    Y_train_a,Y_train_b,Y_train_c = [],[],[]
    #Y_test = []
    for l in lines:
        l = l.replace('  ',' ')
	y = float(l.split(' ')[3])
	ya = float(l.split(' ')[4])
	yb = float(l.split(' ')[5])
	yc = float(l.split(' ')[6])
	Y_train.append(y)
	Y_train_a.append(ya)
	Y_train_b.append(yb)
	Y_train_c.append(yc)

    X = range(0,len(Y_train))
    plt.figure(figsize=(20,8),dpi=80)
    plt.plot(X,Y_train,color='black',linewidth=1,linestyle='-',label='train loss')
    plt.plot(X,Y_train_a,color='green',linewidth=1,linestyle='-',label='TR loss')
    plt.plot(X,Y_train_b,color='red',linewidth=1,linestyle='-',label='RC loss')
    plt.plot(X,Y_train_c,color='blue',linewidth=1,linestyle='-',label='EE loss')
    
    ax = plt.subplot(111)
    xmajorLocator = MultipleLocator(50) 
    xmajorFormatter = FormatStrFormatter('%d')
    ymajorLocator = MultipleLocator(0.5) 
    ymajorFormatter = FormatStrFormatter('%1.1f')
    yminorLocator = MultipleLocator(0.1)

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')

    plt.legend(loc='upper right', frameon=False, fontsize=16)
    #plt.savefig(path_+'/loss_seq.png')
    plt.show()
