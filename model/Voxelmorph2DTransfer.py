from enum import Enum

from base import BaseModel
from model.FCDenseNet import FCDenseNet, FCDenseNetEncoder
from model.utils.layers import *


class EvalutionMode(Enum):
    WARP = 'warp'
    SEG = 'seg'


class Voxelmorph2DTransfer(BaseModel):
    def __init__(self, mode=EvalutionMode.WARP.value, in_channels=8, resolution=(217, 217)):
        super().__init__()
        self.mode = EvalutionMode(mode)
        self.encoder = FCDenseNetEncoder(in_channels=in_channels)
        self.densenet = FCDenseNet(in_channels=in_channels, n_classes=2, apply_softmax=(self.mode == EvalutionMode.SEG), encoder=self.encoder)
        self.spatial_transform = SpatialTransformer(resolution)

    def forward(self, input_moving, input_fixed):
        x = torch.cat([input_moving, input_fixed], dim=1)
        flow = self.densenet(x)  # of EvaluationMode.Seg => This is the segmentation output

        if self.mode == EvalutionMode.SEG:
            return flow

        modalities = torch.unbind(input_moving, dim=1)

        y = torch.stack(
            [torch.squeeze(self.spatial_transform(torch.unsqueeze(modality, 1), flow), dim=1) for modality in modalities]
            , dim=1)

        return y, flow
