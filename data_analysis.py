import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

def get_max_of_csv(path,num = 200):
    l = (pd.read_csv(path)).loc[:num-1,"min_loss"].to_list()
    res = [min(l[:i+1]) for i in range(len(l))]
    # print(len(l))
    return res

def get_csv(path,num=200):
    l = (pd.read_csv(path)).loc[:num-1,"min_loss"].to_list()
    # res = [max(l[:i+1]) for i in range(len(l))]
    
    return l

def get_max_of_res(res_max,res_tmp):
    res_max = list(res_max)
    res = []
    for i in range(len(res_max)):
        if res_max[i] > res_tmp[i]:
            max = res_max[i]
        else:
            max = res_tmp[i]
        res.append(max)
    return res

def get_min_of_res(res_min,res_tmp):
    res_min = list(res_min)
    res = []
    
    for i in range(len(res_min)):
        if res_min[i] < res_tmp[i]:
            min = res_min[i]
        else:
            min = res_tmp[i]
        res.append(min)
    return res


def plot(my_data_path_list = []):
    plt.rcParams["font.family"] = "Times New Roman" # 字体样式
    plt.rcParams['font.size'] = 14 # 字体大小
    plt.rcParams['figure.figsize'] = (8,6) # 6，8分别对应画布宽和高
    plt.rcParams['figure.dpi'] = 300

    num = 200
    for data_path_list in my_data_path_list:
        # print(list(data_path_list))
        res_tmp = []
        res_mean = []
        res_sum = np.zeros(num)
        res_max = np.zeros(num)
        res_min = np.ones(num)            
        count = 0
        for path in data_path_list:
            count = count + 1
            res_tmp = get_max_of_csv(path=path,num=num)
            res_sum = res_sum + np.array(res_tmp)
            # print(res_sum)
            res_max = get_max_of_res(res_max,res_tmp)
            res_min = get_min_of_res(res_min,res_tmp)

            label = Path(path).stem
            plt.scatter(range(num),res_tmp,s = 100,alpha=0.01)

        # print(res_sum)
        # print(count)
        res_mean = res_sum/count
        plt.plot(range(num),res_mean,label = Path(path).stem,linewidth=3,alpha=0.8)
        plt.fill_between(range(num), res_min, res_max,edgecolor = None,alpha=0.1)

    # plt.ylim((0.5,0.8))
    plt.legend(frameon = False)
    plt.ylabel('Min Loss So Far')
    plt.xlabel('Number of Trials')
    plt.savefig('res.png')
    plt.cla()


if __name__ == "__main__":
    from pathlib import Path
    cur_file_path = Path(__file__).absolute()
    parent_cur_dir = Path(cur_file_path).parent

    # target_path = parent_cur_dir / 'database' 
    # data_path_list = target_path.glob('random_3.csv')
    
    target_p2 = Path('/home/wangfz/wksp/NNCompression_run_anneal_loss/database')
    data_path_list2 = target_p2.glob('anneal_0.csv')

    target_p3 = Path('/home/wangfz/wksp/NNCompression_run_tpe_loss/database')
    data_path_list3 = target_p3.glob('tpe_0.csv')

    # print(list(data_path_list))
    # print(list(data_path_list2)) 
    # print(list(data_path_list3))
    my_data_path_list = []
    # my_data_path_list.append(data_path_list)
    my_data_path_list.append(data_path_list2)
    my_data_path_list.append(data_path_list3)
    plot(my_data_path_list)