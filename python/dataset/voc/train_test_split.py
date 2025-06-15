import os  
import random  
import time  
import shutil

xmlfilepath='/media/ai/AI/technology/algorithm/data/1/xml/'  
saveBasePath="/media/ai/AI/technology/develop/python/Annotations"

trainval_percent=0.8  
train_percent=0.8  
total_xml = os.listdir(xmlfilepath)  
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
print("train and val size",tv)  
print("train size",tr) 

start = time.time()

test_num=0  
val_num=0  
train_num=0  

for i in list:  
    name=total_xml[i]
    if i in trainval:  #train and val set 
        if i in train: 
            directory="train"  
            train_num += 1  
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
            if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
            filePath=os.path.join(xmlfilepath,name)  
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
            shutil.copyfile(filePath, newfile)
        else:
            directory="validation"  
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
            if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
            val_num += 1  
            filePath=os.path.join(xmlfilepath,name)   
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
            shutil.copyfile(filePath, newfile)

    else:
        directory="test"  
        xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))  
        if(not os.path.exists(xml_path)):  
                os.mkdir(xml_path)  
        test_num += 1  
        filePath=os.path.join(xmlfilepath,name)  
        newfile=os.path.join(saveBasePath,os.path.join(directory,name))  
        shutil.copyfile(filePath, newfile)

end = time.time()  
seconds=end-start  
print("train total : "+str(train_num))  
print("validation total : "+str(val_num))  
print("test total : "+str(test_num))  
total_num=train_num+val_num+test_num  
print("total number : "+str(total_num))  
print( "Time taken : {0} seconds".format(seconds))
