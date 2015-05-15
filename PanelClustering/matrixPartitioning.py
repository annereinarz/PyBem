class panelPair:
    def __init__(self,t1,t2):
        self.t1 = t1
        self.t2 = t2
       
    def getPanels(self):
        return [self]
    def isPanel(self):
        return True
    def __str__(self):
        return '('+ str(self.p1) +', '+ str(self.p2) +')'
    def format(self, depth):
        return '  '*depth + str(self)

class clusterPair:
    def __init__(self, t1, t2, children):
       self.t1 = t1
       self.t2 = t2
       self.children = children
    def __str__(self):
        return self.format()
    def format(self, depth=0):
        return '  '*depth + ' '.join(str(p) for p in self.getPanels()) + '\n' + '\n'.join(c.format(depth+1) for c in self.children)
    def getPanels(self):
        ps = []
        for c in self.children:
            for p in c.getPanels():
                ps.append(p)
        return ps
    def isPanel(self):
        return False


def transform((t1,t2)):
    if t1.isPanel() or t2.isPanel():
        return panelPair(t1,t2)
    t11,t12 = t1.children
    t21,t22 = t2.children
    
    return clusterPair(t1, t2, children = map(transform,[(t11,t21),(t11,t22),(t12,t21),(t12,t22)]))

