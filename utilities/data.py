import datetime
import random
import time

from hy2dl.datasetzoo import get_dataset
from hy2dl.utils.config import Config
from torch.utils.data import DataLoader


class DataHandler:

    def __init__(self, config: Config):
        self.cfg = config
        self.ds_training = None
        self.ds_validation = None
        self.ds_testing = None

    def get_basin_ids(self):
        with self.cfg.path_entities.open("r") as f:
            basin_ids = f.read().splitlines()

        return [int(basin) for basin in basin_ids]

    def load_data(self):
        # Which CAMELS dataset?
        Dataset = get_dataset(self.cfg)
        
        # Load all periods
        self.cfg.logger.info(f"Loading data from {self.cfg.dataset.upper()} dataset...")

        # Tick
        total_time = time.time()

        # Training
        self.cfg.logger.info("Loading training data...")
        self.ds_training = Dataset(self.cfg, time_period="training")
        self.ds_training.calculate_basin_std()
        self.ds_training.calculate_global_statistics(save_scaler=True)
        self.ds_training.standardize_data()
        scaler = self.ds_training.scaler

        # Validation
        if self.cfg.validate_n_random_basins > 0:
            basin_ids = self.get_basin_ids()
            random_basins = random.sample(basin_ids, k=self.cfg.validate_n_random_basins)
            random_basins = [str(basin) for basin in random_basins]
        
        self.cfg.logger.info("Loading validation data...")
        self.ds_validation = Dataset(
            self.cfg,
            time_period="validation",
            entities_ids=random_basins if self.cfg.validate_n_random_basins > 0 else None,
        )
        self.ds_validation.calculate_basin_std()
        self.ds_validation.scaler = scaler
        self.ds_validation.standardize_data(standardize_output=False)
        
        # Report
        self.cfg.logger.info("Number of valid samples")
        self.cfg.logger.info(f"   {'Training:':<12} {len(self.ds_training):>8,}")
        self.cfg.logger.info(f"   {'Validation:':<12} {len(self.ds_validation):>8,}")

        # Tock
        self.cfg.logger.info(f"Time required to process load data: {datetime.timedelta(seconds=int(time.time() - total_time))}")


    def get_loader(self, split: str):
        # Get the data loader with the appropriate settings depending on the
        # split type
        match split:
            case "training":
                loader = DataLoader(
                    dataset=self.ds_training,
                    batch_size=self.cfg.batch_size_training,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=self.ds_training.collate_fn,
                    num_workers=self.cfg.num_workers,
                )
            case "validation":
                loader = DataLoader(
                    dataset=self.ds_validation,
                    batch_size=self.cfg.batch_size_evaluation,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=self.ds_validation.collate_fn,
                    num_workers=self.cfg.num_workers,
                )
        
        # For the training data loader, print its contents to see if everything
        # is working correctly
        if split == "training":
            self.cfg.logger.info(" Details DataLoader ".center(60, "-"))
            self.cfg.logger.info(f"{'Key':^30}|{'Shape':^30}")
            # Loop through the sample dictionary and print the shape of each element
            for key, value in next(iter(loader)).items():
                if key.startswith(("x_d", "x_conceptual")):
                    self.cfg.logger.info(f"{key}")
                    for k, v in value.items():
                        self.cfg.logger.info(f"{k:^30}|{str(v.shape):^30}")
                else:
                    self.cfg.logger.info(f"{key:<30}|{str(value.shape):^30}")
        
        # Print number of batches
        self.cfg.logger.info(f"{split.capitalize()} batches: {len(loader)}")
        return loader