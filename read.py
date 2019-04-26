from scipy.io import loadmat
"""
A database of 41,368 images of 68 people, each person under 
13 different poses, 43 different illumination conditions, and with 4 
different expressions.    
说明：此处提供5种姿态数据供大家使用，isTest字段为1时，为测试图， 不参与训练
"""
class ReadHandler():
    """
    读取训练与测试数据
    """
    def __init__(self,file):
        """
        读取file文件，将训练集和测试集的id取出
        """
        m=loadmat(file)
        self.istest=m['isTest']
        self.gnd=m['gnd']
        self.fea=m['fea']
        self.total_len=len(self.istest)
        self.train_id=[]
        self.test_id=[]
        for i in range(self.total_len):
            if int(self.istest[i][0])==0:
                self.train_id.append(i)
            else:
                self.test_id.append(i)
        #print(len(self.train_id),len(self.test_id))

    def read_train(self):
        """
        读取file文件的训练集数据，返回图像特征fea和对应人物gnd，都为list类型
        """
        train_gnd=[]
        train_fea=[]
        for id in self.train_id:
            train_gnd.append(int(self.gnd[id][0]))
            train_fea.append(self.fea[id])
        return train_fea,train_gnd

    def read_test(self):
        """
        读取file文件的测试集数据，返回图像特征fea和对应人物gnd，都为list类型
        """
        test_gnd=[]
        test_fea=[]
        for id in self.test_id:
            test_gnd.append(int(self.gnd[id][0]))
            test_fea.append(self.fea[id])
        return test_fea,test_gnd
   
"""
read=ReadHandler('PIE dataset/Pose29_64x64.mat')
train_fea,train_gnd=read.read_train()
test_fea,test_gnd=read.read_test()
print(len(test_gnd),len(train_gnd))
"""