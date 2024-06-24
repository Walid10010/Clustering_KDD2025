
class DPoint:


    def __init__(self, coordinates, name, idx):
        self.coordinates = coordinates
        self.idx = idx
        self.neighbours = None
        self.rknn = set([idx])
        self.rknn_dist = 0
        self.symmetric_knn = set([idx])
        self.densityPeak = 0
        self.avg_k_distance = None
        self.name = name
        self.label = None




