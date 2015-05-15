class panel:
    def __init__(self, cube):
        self.cube = cube
    def getPanels(self):
        return [self]
    def getBounds(self, axis):
        return self.cube[axis]
    def isPanel(self):
        return True
    def __str__(self):
        return str(self.cube)
    def format(self, depth):
        return '  '*depth + str(self.cube)

class cluster:
    def __init__(self, children, cube):
        self.children = children
        self.cube = cube
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
    def getBounds(self, axis):
        return self.cube[axis]