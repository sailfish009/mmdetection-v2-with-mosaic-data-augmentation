from .mosaiccoco import MosaicCocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class MyMosaicDataset(MosaicCocoDataset):

    CLASSES = ('wheat', 'background')