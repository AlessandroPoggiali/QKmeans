from QKmeans import QKMeans
from itertools import product

def test_iris():
    
    params_iris = {

        'dataset_name': ['iris'],
        'K': [2,3],
        'M1': [150],
        'shots': [150000],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    keys, values = zip(*params_iris.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print("Total configurations: " + str(len(params_list)))
    
    for i, params in enumerate(params_list):

        QKMEANS = None
        print("Configuration: " + str(i) +"\n")

        conf = {
            "dataset_name": params['dataset_name'],
            "K": params['K'],
            "M1": params['M1'],
            "shots": params['shots']
            "sc_tresh": params['sc_tresh'],
            "max_iterations": params['max_iterations'] 
        }
    
        QKMEANS = QKMeans(conf)
        QKMEANS.run()
        QKMEANS.print_result('result/iris.csv')    


if __name__ == "__main__":
    
    test_iris()
    
    