import os
import csv

if __name__ == '__main__' :
    
    path_ = os.path.abspath('.')
    a2_path = path_.replace('preprocess','a2')
    target_path = path_.replace('preprocess','a2_table')

    for root,_,fnames in os.walk(a2_path):
        for fname in fnames :
            print fname

            f_path = os.path.join(a2_path,fname)
            write_path = os.path.join(target_path,fname.replace('.a2','.csv'))

            csvfile = open(write_path,'wb')
            writer = csv.writer(csvfile)
            writer.writerow(['event_id','event_type','src','relation','dst'])
            csvfile.close()

            csvfile = open(write_path,'ab+')
            writer = csv.writer(csvfile)

            file_a2 = open(f_path,'r')
            line = file_a2.readline()
            
            while line :
                if line[0] != 'E' :
                    line = file_a2.readline()
                    continue

                line = line.replace('\n','')
                line = line.replace('\t',' ')
                line_list = line.split(' ')

                e_id = line_list[0]
                e_type = line_list[1].split(':')[0]
                sc = line_list[1].split(':')[1]

                if line_list[-1] == '' : # an event with no argument
                    writer.writerow([e_id,e_type,sc])
                    line = file_a2.readline()
                    continue

                for i in range(2,len(line_list)):
                    rlt = line_list[i].split(':')[0]
                    dt = line_list[i].split(':')[1]
                    writer.writerow([e_id,e_type,sc,rlt,dt])

                line = file_a2.readline()

            file_a2.close()
            csvfile.close()
