from sklearn.cluster import DBSCAN

class CustomDBScan:
    def __init__(self, points: list, eps: int, samples: int):
        self.points = points
        self.eps = eps
        self.samples = samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)
        
    def get_dbscan(self):
        clusters = self.dbscan.fit_predict(self.points)
        return clusters