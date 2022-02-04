from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy import trapz
from tqdm import tqdm_notebook
import os

class TopicModelling: 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
        
    def LDA(self,token_counts,n_components): 
        lda = LatentDirichletAllocation(n_components=n_components,random_state=0)
        return lda.fit_transform(token_counts)
    
    def PCA(self,features,n_components): 
        pass 
    
    def TSNE(self,features): 
        pass 
    
    def get_kmeans(self,features,n_clusters): 
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, algorithm = "full").fit(features)
        return kmeans.labels_ 
    
    def get_silhouette_score(self,features,labels): 
        return silhouette_score(features,labels)
    
    def get_kmeans_curve(self,features,max_clusters):
        labels = dict()
        silhouettes = dict()
        for n_clusters in tqdm_notebook(range(2,max_clusters)): 
            labels[n_clusters] = self.get_kmeans(features,n_clusters)
            silhouettes[n_clusters] = self.get_silhouette_score(features,labels[n_clusters])
            
        return labels, silhouettes 
    
    def get_lda_clusters(self,features,max_clusters,min_topic,max_topics): 
        self.topic_modelling_results = dict()
        for n_components in tqdm_notebook(range(min_topic,max_topics)): 
            topics = self.LDA(features,n_components)
            self.topic_modelling_results[n_components] = self.get_kmeans_curve(topics,max_clusters)
        
        
    def plot_topics_separability(self): 
        assert hasattr(self,"topic_modelling_results")
        f,ax = plt.subplots()
        for n_components in self.topic_modelling_results.keys(): 
            labels, silhouettes = self.topic_modelling_results[n_components]
        
            ax.plot(silhouettes.keys(),silhouettes.values(),label = f"{n_components} topics")
            plt.xlabel("n_clusters")
            plt.ylabel("Silhouette coefficient")
        plt.legend()
        plt.show()
        
    def plot_topics(self,features,labels): 
        f,ax = plt.subplots(figsize = (10,10))
        sns.scatterplot(data = np.log(features+1) , x = f"Component_1", y = "Component_2",hue = labels)
        plt.show()
        
    def get_area_under_curve(self,feature): 
        return np.trapz(feature)
        
    def select_best_number_of_topics(self,n_topics = False, n_clusters = False): 
        if not n_topics and not n_clusters: 
            labels = [self.topic_modelling_results[n_topics][0] for n_topics in self.topic_modelling_results.keys()]
            silhouettes = [self.topic_modelling_results[n_topics][1] for n_topics in self.topic_modelling_results.keys()]
            
            silhouettes = pd.DataFrame(silhouettes, columns = [f"{n}_topics" for n in self.topic_modelling_results.keys()], 
                                                               index = [f"{n}_clusters" for n in range(len(silhouettes))])
            
            aucs = [(col_id+min(self.topic_modelling_results.keys()),self.get_area_under_curve(silhouettes.iloc[:,col_id])) for col_id in range(silhouettes.shape[1])]
            best_topic_number = max(aucs, key = lambda x: x[1])[0]
            best_cluster_number = max(silhouettes.filter([f"{best_topic_number}_topics"],axis = 1))
            
            return best_topic_number,int(best_cluster_number.split("_")[0])
        
    def get_best_topics(self,counts): 
        topic_number, cluster_number = self.select_best_number_of_topics()
        self.plot_topics_separability()
        topics = self.LDA(counts,topic_number)
        topics = pd.DataFrame(topics, columns = [f"Component_{n}" for n in range(1,topic_number+1)])
        labels = self.get_kmeans(topics,cluster_number)
        self.plot_topics(topics,labels)
        
        