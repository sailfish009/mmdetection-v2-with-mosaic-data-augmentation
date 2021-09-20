from .mosaiccoco import MosaicCocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class MyMosaicDataset(MosaicCocoDataset):

    CLASSES = ('letters', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')
    
