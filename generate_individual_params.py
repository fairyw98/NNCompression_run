import yaml
import pandas as pd

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        content = yaml.safe_load(file)
    return content

def change_yaml(content,scheme):
    tmp = content['backbone'][scheme[0]]
    tmp[2] = 'wfz_Conv_compression'
    param = []

    param.extend([tmp[3][-3]])
    param.extend(scheme[1:])
    param.extend(tmp[3][-2:])

    tmp[3] = param

    return content

def write_yaml(yaml_path,content):
    with open(yaml_path,'w') as file:
        yaml.dump(content,file)
    # return content
    return

def read_csv(csv_path):
    df1 = pd.read_csv(csv_path).copy()
    df2 = df1.loc[(df1['partition_id'] == 3) &
                (df1['best_acc'] > 0.8 )
                ,:]
    df2.to_csv('./utils/params.csv',index=False)
    return

if __name__ =="__main__":
    csv_path = "database.csv"
    read_csv(csv_path)