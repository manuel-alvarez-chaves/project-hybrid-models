import datetime
import time

import torch
from hy2dl.training.loss import nse_basin_averaged
from hy2dl.utils.optimizer import Optimizer
from hy2dl.utils.utils import upload_to_device
from tqdm import tqdm


class Trainer:
    
    def __init__(self, config, dataloaders, model):
        self.cfg = config
        self.dataloaders = dataloaders or {}
        self.current_epoch = 0
        self.model = model
        self.optimizer = None
        self.criterion = nse_basin_averaged

        self._setup_training()
      
    def _setup_training(self):
        self.optimizer = Optimizer(cfg=self.cfg, model=self.model)

    def run_epoch(self, period: str = "training"):
        is_training = (period == "training")

        # Set model mode
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        # Get appropriate DataLoader
        loader = self.dataloaders.get(period)

        # Context manager for gradients
        context = torch.no_grad() if not is_training else torch.enable_grad()

        # Make the data loader an iterator
        iterator = tqdm(
                    loader,
                    desc=period.capitalize(),
                    ncols=79,
                    unit="batch",
                    ascii=True,
                    position=1,
                    leave=False,
                )

        # Loop
        loss_evol = []
        start_time = time.time()
        with context:
            for idx, sample in enumerate(iterator):
                if self.cfg.max_updates_per_epoch is not None and idx >= self.cfg.max_updates_per_epoch:
                    break

                sample = upload_to_device(sample, self.cfg.device)
                if is_training:
                    self.optimizer.optimizer.zero_grad()

                # Forward pass
                pred = self.model(sample)

                # Calculate loss
                loss = nse_basin_averaged(
                    y_sim=pred["y_hat"],
                    y_obs=sample["y_obs"],
                    per_basin_target_std=sample["std_basin"]
                )

                if is_training:
                    loss.backward()
                    self.optimizer.clip_grad_and_step(self.current_epoch, idx)

                loss_evol.append(loss.item())
                iterator.set_postfix({"loss": f"{sum(loss_evol)/len(loss_evol):.3f}"})

                # Free memory
                del sample, pred
                torch.cuda.empty_cache()
        total_time = time.time() - start_time
        total_time = str(datetime.timedelta(seconds=int(total_time)))

        # Save the model
        path_save_model = self.cfg.path_save_folder / f"model_epoch_{(self.current_epoch + 1):02d}.pt"
        torch.save(self.model.state_dict(), path_save_model)

        if not is_training:
            self.current_epoch += 1

        return sum(loss_evol)/len(loss_evol), total_time