import datetime
import json
import pickle
import time

import numpy as np
import torch
import xarray as xr
from hy2dl.datasetzoo import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from unite_toolbox.knn_estimators import calc_knn_entropy


def calc_nse(sim, obs) -> float:
    num = ((obs - sim)**2).mean(skipna=True)
    den = ((obs - obs.mean(skipna=True))**2).mean(skipna=True)
    return float(1 - num / den)

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
        iterator = tqdm(self.basin_ids, desc="Basins", ascii=True, unit="basin", leave=False)
        self.cfg.logger.info("Post-processing model...")
        for basin in iterator:
            loader = DataLoader(
                dataset=ds_testing[basin],
                batch_size=self.cfg.batch_size_evaluation,
                shuffle=False,
                drop_last=False,
                collate_fn=ds_testing[basin].collate_fn,
                num_workers=self.cfg.num_workers,
            )
            dates, y_obs, y_hat, hs = [], [], [], []
            
            for sample in loader:
                dates.append(sample["date"])
                y_obs.append(sample["y_obs"].detach().cpu().numpy())
                
                pred = model(sample)
                y_hat.append(pred["y_hat"].detach().cpu().numpy())

                hs.append(pred["hs"].detach().cpu().numpy())

                del sample, pred
                torch.cuda.empty_cache()

            out[basin] = {
                "dates": np.concatenate(dates),
                "y_obs": np.concatenate(y_obs),
                "y_hat": np.concatenate(y_hat),
                "hs": np.concatenate(hs)
            }
            del dates, y_obs, y_hat, hs

        return out

    def _dict_to_xarray(self, out_dict: dict):
        self.cfg.logger.info("Saving to netCDF...")

        # Get all basin IDs
        basin_ids = list(out_dict.keys())
        B = len(basin_ids)

        # Get dates from first basin (assuming all basins have same dates)
        dates = out_dict[basin_ids[0]]["dates"]
        D = dates.shape[0]

        # Get predict_last_n and num_targets from the observations
        N = out_dict[basin_ids[0]]["y_obs"].shape[1]
        T = out_dict[basin_ids[0]]["y_obs"].shape[2]

        # Get number of hidden states
        HS = out_dict[basin_ids[0]]["hs"].shape[2]

        y_obs_array = np.zeros((B, D, N, T))
        y_hat_array = np.zeros((B, D, N, T))
        hs_array = np.zeros((B, D, N, HS))

        # Fill xarrays
        for idx, basin in enumerate(basin_ids):
            y_obs_array[idx] = out_dict[basin]["y_obs"]
            y_hat_array[idx] = out_dict[basin]["y_hat"]
            hs_array[idx] = out_dict[basin]["hs"]

        # Create coordinate arrays
        coords = {
            "basin": ("basin", basin_ids),
            "date": ("date", dates[:, -1]), # date of last prediction
            "last_n": ("last_n", np.arange(1, N + 1)),
            "hidden_state": ("hidden_state", np.arange(1, HS + 1)),
            "target": ("target", np.arange(1, T + 1))
        }

        ds = xr.Dataset(
            {
                "y_obs": (("basin", "date", "last_n", "target"), y_obs_array),
                "y_hat": (("basin", "date", "last_n", "target"), y_hat_array),
                "hs": (("basin", "date", "last_n", "hidden_state"), hs_array)
            },
            coords=coords
        )
        return ds
    
    def _calc_metrics(self, ds) -> dict:
        metrics = {}
        iterator = tqdm(ds.basin.values, desc="Basins", ascii=True, unit="basin", leave=False)
        self.cfg.logger.info("Computing metrics...")
        for basin in iterator:
            # Compute NSE
            y_obs = ds.sel(basin=basin).y_obs
            y_sim = ds.sel(basin=basin).y_hat
            
            nse = calc_nse(y_sim, y_obs)

            # Compute Entropy
            hs = ds.sel(basin=basin)["hs"].values[:, -1, :]
            hs = float(calc_knn_entropy(hs, k=3))

            metrics[str(basin)] = {
                "nse": nse,
                "h_hs": hs
            }
        return metrics
    
    def postprocess(self, model):
        out = self._evaluate(model)
        ds = self._dict_to_xarray(out)
        path_results = self.cfg.path_save_folder / "results.nc"
        ds.to_netcdf(path_results)
        metrics = self._calc_metrics(ds)
        with open(self.cfg.path_save_folder / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        return ds