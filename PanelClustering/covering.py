from clustering import panel

def minimalCovering(admissable, cluster):
    if cluster.isPanel() or admissable(cluster):
        return [cluster]
    return sum([ minimalCovering(admissable, c) for c in cluster.children ], [])