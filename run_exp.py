import os
import Search as Search
import utils.config as config

def main(algo):
    search_args = config.search_args(algo=algo)
    Search.main(search_args)

if __name__ == "__main__":
    # algo = ['random','tpe','anneal']
    algo = ['anneal']
    # algo = ['tpe','anneal','random']
    num = 10

    if os.path.exists("./database") is False:
        os.makedirs("./database")

    for algo_method in algo:
        # random.shuffle(algo)
        for i in range(num):
            config.database_csv = f'./database/{algo_method}_{i}.csv'
            main(algo_method)