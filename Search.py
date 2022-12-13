import argparse
import csv
import os
import utils.config as config
import train as train
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval,rand,anneal
import pandas as pd

def trick1(database_path,scheme):
    df1 = pd.read_csv(database_path).copy()
    df2 = df1.loc[(df1['partition_id'] == scheme[0]) &
                (df1['quant_bits'] == scheme[1] ) &
                (df1['coder_channels'] == scheme[2] ) &
                (df1['en_stride'] == scheme[3] )
                ,:]
    
    # print(df2)
    if df2.empty:
        return -1
    else:
        df2 = df2.reset_index(drop=True)
        # return float(df2['min_loss'][0])
        return (float(df2['best_acc'][0]),float(df2['min_loss'][0]))

# define an objective function
def objective(args):
    scheme = args
    database_path = config.database_csv

    trick_res = trick1(database_path,scheme)
    if trick_res!=-1:
        # print(trick_res)
        if os.path.exists(database_path) is True: 
            with open(database_path,'a+',newline='') as f:
                writer = csv.writer(f)
                row_list = [scheme[0],scheme[1],scheme[2],scheme[3],trick_res[0],trick_res[1]]
                writer.writerow(row_list)
        return trick_res[-1]
    # return 0
    train_args = config.train_args(partition_id=scheme[0],
                quantization=scheme[1],
                coder_channels=scheme[2],
                en_stride=scheme[3])
    train.main(train_args)

    if os.path.exists(database_path) is True: 
        with open(database_path,'a+',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scheme[0],scheme[1],scheme[2],scheme[3],config.best_acc,config.min_loss])

    # return -config.best_acc
    return config.min_loss

# define a search space
space = [
    hp.choice('partition_id',[2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]), # 3
    hp.choice('quant_bits',[1,2,4,8,16,32]), # 6
    hp.choice('coder_channels',[1,2,4,8,16,32,64]), # 7
    hp.choice('en_stide',[1,2,3,5,6,7,9]), # 7
]

def main(args):
    database_path = config.database_csv

    alg_methods = {}
    alg_methods['random'] = rand.suggest
    alg_methods['anneal'] = anneal.suggest
    alg_methods['tpe'] = tpe.suggest

    alg = alg_methods[args.algo]
    max_evals = args.max_evals

    with open(database_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['partition_id','quant_bits','coder_channels','en_stride','best_acc','min_loss'])

    # minimize the objective over the space
    best = fmin(objective, 
                space, 
                algo=alg, 
                max_evals=max_evals)

    print(best)
    print(space_eval(space, best))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='random')
    parser.add_argument('--max_evals', type=int, default=200)
    opt = parser.parse_args()
    main(opt)

        