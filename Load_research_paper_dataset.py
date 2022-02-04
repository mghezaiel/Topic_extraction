
import pandas as pd 


class loadDataset(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
        self.load_dataset(path = self.path)
        self.sample_data(weights = self.weights)
        self.get_documents()
        
    def load_dataset(self,path): 
        self.dataset = pd.read_csv(path)
        
    def sample_data(self,frac = 0.1,weights = True):
        labels = ["Computer Science","Physics","Mathematics","Statistics","Quantitative Biology","Quantitative Finance"]
        dataset_labels = self.dataset.filter(labels,axis = 1).idxmax(axis = 1).values.tolist()
        if weights:
            weights = [dataset_labels.count(label)/len(dataset_labels) for label in dataset_labels]
        
        else: 
            weights = None 
        self.dataset = self.dataset.sample(frac = frac,weights = weights)
        self.dataset_labels = self.dataset.filter(labels,axis = 1).idxmax(axis = 1).values.tolist()
        
    def get_documents(self): 
        self.documents = self.dataset["ABSTRACT"]

    

