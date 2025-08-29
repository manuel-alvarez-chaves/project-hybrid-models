import datetime
import pickle
import time

import numpy as np
import torch
import xarray as xr
from hy2dl.datasetzoo import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Postprocessor:

    def __init__(self, config):
        self.cfg = config

        # Load basin ids
        self.basin_ids = []
        with self.cfg.path_entities.open("r") as f:
            entities_ids = [basin for basin in f.read().splitlines()]
            self.basin_ids.extend(entities_ids)

    def _load_testing_data(self):
        # Which CAMELS dataset?
        Dataset = get_dataset(self.cfg)

        # Tick
        total_time = time.time()

        # Load scaler
        with open(self.cfg.path_save_folder / "scaler.pickle", "rb") as f:
            scaler = pickle.load(f)

        # For testing data, we have one Dataset object per basin
        self.cfg.logger.info("Loading testing data...")
        data = {}
        for basin in self.basin_ids:
            ds = Dataset(
                cfg=self.cfg,
                time_period="testing",
                check_NaN=False,
                entities_ids=basin
            )
            ds.scaler = scaler
            ds.standardize_data(standardize_output=False)
            data[basin] = ds

        total_time = datetime.timedelta(seconds=int(time.time() - total_time))
        self.cfg.logger.info(f"Testing data loaded in {total_time}")
        
        return data

    def _evaluate(self, model):
        ds_testing = self._load_testing_data()

        model.eval()
        out = {}
        iterator = tqdm(self.basin_ids, desc="Basins", ncols=78, ascii=True, unit="basin")
        for basin in iterator:
            loader = DataLoader(
                dataset=ds_testing[basin],
                batch_size=self.cfg.batch_size_evaluation,
                shuffle=False,
                drop_last=False,
                collate_fn=ds_testing[basin].collate_fn,
                num_workers=self.cfg.num_workers,
            )
            dates, y_obs, y_hat = [], [], []
            
            for sample in loader:
                dates.append(sample["date"][:, -1])
                y_obs.append(sample["y_obs"][:, -1, 0].detach().cpu().numpy())
                pred = model(sample)["y_hat"][:, -1, 0] # many-to-one
                pred = pred * ds_testing[basin].scaler["y_std"] + ds_testing[basin].scaler["y_mean"]
                y_hat.append(pred.detach().cpu().numpy())

                del sample, pred
                torch.cuda.empty_cache()

            out[basin] = {
                "dates": np.concatenate(dates),
                "y_obs": np.concatenate(y_obs),
                "y_hat": np.concatenate(y_hat)
            }
            del dates, y_obs, y_hat
        
        return out

    def _dict_to_xarray(self, out_dict: dict):
        # Get all basin IDs
        basin_ids = list(out_dict.keys())
        B = len(basin_ids)

        # Get dates from first basin (assuming all basins have same dates)
        dates = out_dict[basin_ids[0]]["dates"]
        N = dates.shape[0]

        y_obs_array = np.zeros((B, N))
        y_hat_array = np.zeros((B, N))

        # Fill xarrays
        for idx, basin in enumerate(basin_ids):
            y_obs_array[idx] = out_dict[basin]["y_obs"]
            y_hat_array[idx] = out_dict[basin]["y_hat"]

        # Create coordinate arrays
        coords = {
            "basin": ("basin", basin_ids),
            "date": ("date", dates)
        }

        ds = xr.Dataset(
            {
                "y_obs": (("basin", "date"), y_obs_array),
                "y_hat": (("basin", "date"), y_hat_array)
            },
            coords=coords
        )
        return ds
    
    def postprocess(self, model):
        out = self._evaluate(model)
        ds = self._dict_to_xarray(out)
        path_results = self.cfg.path_save_folder / "results.nc"
        ds.to_netcdf(path_results)
        return ds