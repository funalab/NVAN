class normalize(object):
    def __init__(self):
        pass
    def __call__(self, x):
        
        return (x - x.min()) / (x.max() - x.min())
