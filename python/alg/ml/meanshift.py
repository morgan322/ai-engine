
import numpy as np
import random
DISTANCE_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1

def distance(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def Gaussian_kernal(distance,sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*distance/(sigma**2))

class MeanShift(object):
    def __init__(self,kernal = Gaussian_kernal):
        self.kernal = kernal
    def shift_points(self,center_point,whole_points,Gaussian_sigma):         ##计算center_point点移动后的坐标
        shifting_px = 0.0
        shifting_py = 0.0
        sum_weight = 0.0
        for each_point in whole_points:#遍历每一个点
            dis = distance(center_point,each_point)#计算当前点与中心点的距离
            Gaussian_weight = self.kernal(dis,Gaussian_sigma)#计算当前点距离中心点的高斯权重
            shifting_px += Gaussian_weight * each_point[0]
            shifting_py += Gaussian_weight * each_point[1]
            sum_weight += Gaussian_weight
        shifting_px /= sum_weight          #归一化
        shifting_py /= sum_weight
        return [shifting_px,shifting_py]
    
    #根据shift之后的点坐标shifting_points获得聚类id
    def cluster_points(self,shifting_points):
        clusterID_points = []#用于存放每一个点的类别号
        cluster_id=0#聚类号初始化为0
        cluster_centers = []#聚类中心点
        for i,each_point in enumerate(shifting_points):#遍历处理每一个点
            if i==0:#如果是处理的第一个点
                clusterID_points.append(cluster_id)#将这个点归为初始化的聚类号(0)
                cluster_centers.append(each_point)#将这个点看作聚类中心点
                cluster_id+=1#聚类号加1
            else:#处理的不是第一个点的情况
                for each_center in cluster_centers:#遍历每一个聚类中心点
                    dis = distance(each_center,each_point)#计算当前点与该聚类中心点的距离
                    if dis < CLUSTER_THRESHOLD:#如果距离小于聚类阈值
                        clusterID_points.append(cluster_centers.index(each_center))#就将当前处理的点归为当前中心点同类（聚类号赋值）
                if(len(clusterID_points)<i+1):#如果上面那个for，所有的聚类中心点都没能收纳一个点，说明是时候开拓一个新类了
                    clusterID_points.append(cluster_id)#把当前点置为一个新类，因为此时的cluster_idx以前谁都没用过
                    cluster_centers.append(each_point)#将这个点作为这个这个新聚类的中心点
                    cluster_id+=1#聚类号加1以备后用
        return clusterID_points
        
    #whole_points：输入的所有点
    #Gaussian_sigma:Gaussian核的sigma
    def fit(self,whole_points,Gaussian_sigma):
        shifting_points = np.array(whole_points)
        need_shifting_flag = [True] * np.shape(whole_points)[0]#每一个点初始都标记为需要shifting
        while True:
            distance_max = 0.0
            #每一轮迭代都对每一个点进行处理
            for i in range(0,np.shape(whole_points)[0]):
                if not need_shifting_flag[i]:#如果这个点已经被标记为不需要继续shifting，就continue
                    continue
                shifting_point_init = shifting_points[i].copy()#将初始的第i个点的坐标备份一下
                #shifting_points[i]由第i个点的坐标更新为第i个点移动后的坐标
                shifting_points[i] = self.shift_points(shifting_points[i],whole_points,Gaussian_sigma)
                #计算第i个点移动的距离
                dis = distance(shifting_point_init,shifting_points[i])
                #如果该点移动的距离小于停止阈值，标记need_shifting_flag[i]为False，下一轮迭代对该点不做处理
                need_shifting_flag[i] = dis > DISTANCE_THRESHOLD
                #本轮迭代中最大的距离存储到distance_max中
                distance_max = max(distance_max,dis)
            #如果在一轮迭代中，所有点移动的最大距离都小于停止阈值，就停止迭代
            if(distance_max < DISTANCE_THRESHOLD):
                break
        #根据shift之后的点坐标shift_points获得聚类id
        cluster_class_id = self.cluster_points(shifting_points.tolist())
        return shifting_points,cluster_class_id
        
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 


#按照均匀分布随机产生n个颜色，每个颜色都由R、G、B三个分量表示
def colors(n):
    ret = []
    for i in range(n):
        ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def main():
    centers = [[0, 1], [-1, 2], [1, 2], [-2.5, 2.5], [2.5,2.5], [-4,1], [4,1], [-3,-1], [3,-1], [-2,-3], [2,-3], [0,-4]]#设置一些中心点
    X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.3)#产生以这些中心点为中心，一定标准差的n个samples

    mean_shifter = MeanShift()
    shifted_points, mean_shift_result = mean_shifter.fit(X, Gaussian_sigma=0.3)#Gaussian核设置为0.5,对X进行mean_shift

    np.set_printoptions(precision=3)
#    print('input: {}'.format(X))
#    print('assined clusters: {}'.format(mean_shift_result))
    color = colors(np.unique(mean_shift_result).size)

    for i in range(len(mean_shift_result)):
        plt.scatter(X[i, 0], X[i, 1], color = color[mean_shift_result[i]])
        plt.scatter(shifted_points[i,0],shifted_points[i,1], color = 'r')
    plt.xlabel("M")
    plt.savefig("../../../data/result/result_meanshift.png")
    plt.show()

if __name__ == '__main__':
    main()