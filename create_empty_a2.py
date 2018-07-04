import os

if __name__ == '__main__':
        
    txt_path = 'E:/corpus/tmp2/a1'

    for fpath,_,files in os.walk(txt_path):
        for fl in files:
            print fl
        
            file_path = os.path.join(fpath,fl)
            file_path = file_path.replace('a1','a2')
            f = open(file_path,'w')
            f.close()
