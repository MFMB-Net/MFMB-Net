"""
AIO -- All Trains in One
"""

from trains.missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # missing-task
            'mfmb_net': MFMB_NET,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
