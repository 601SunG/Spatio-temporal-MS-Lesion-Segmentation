import os

CONFIG = {
    "name": f"{os.path.basename(__file__).split('.')[0]}",
    "n_gpu": 1,

    "arch": {
        "type": "Voxelmorph2DTransfer",
        "args": {
            "mode": "warp",
            "resolution": (217, 217)
        }
    },
    "dataset": {
        "type": "ISBIDatasetLongitudinal",
        "args": {
            "data_dir": "../ISBIMSlesionChallenge/",
            "preprocess": True,
            "modalities": ['flair', 'mprage', 'pd', 't2'],
            "val_patients": [4]
        }
    },
    "data_loader": {
        "type": "ISBIDataloader",
        "args": {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 4,
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.0001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "deformation_loss",
    "metrics": [],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "type": "LongitudinalDeformationTrainer",
        "epochs": 200,
        "save_dir": "../saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": True
    }
}
