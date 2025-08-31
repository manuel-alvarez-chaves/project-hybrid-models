import datetime
import time

import torch
from hy2dl.training.loss import nse_basin_averaged
from hy2dl.utils.optimizer import Optimizer
from hy2dl.utils.utils import upload_to_device
from tqdm import tqdm


def _mask(*tensors: torch.Tensor) -> tuple[torch.Tensor]:
    masks = []
    for tensor in tensors:
        num_dim = tensor.dim()
        for _ in range(num_dim - 1):
            tensor = tensor.sum(dim=1)
        mask = ~tensor.isnan()
        masks.append(mask)
    mask = torch.stack(masks, dim=1).all(dim=1)

    return tuple(tensor[mask] for tensor in tensors)

def calc_nse(sim: torch.Tensor, obs: torch.Tensor) -> float:
    # sim, obs: B, N, T
    sim, obs = _mask(sim, obs)
    num = (obs - sim).pow(2).mean()
    den = (obs - obs.mean(dim=(1, 2), keepdim=True)).pow(2).mean()
    return (1 - num / den).item()

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
                    unit="batch",
                    ascii=True,
                    leave=False,
                )

        # Loop
        loss_evol, nse_evol = [], []
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

                nse = calc_nse(pred["y_hat"], sample["y_obs"])
                nse_evol.append(nse)

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
        path_save_model = self.cfg.path_save_folder / f"model/model_epoch_{(self.current_epoch + 1):02d}.pt"
        torch.save(self.model.state_dict(), path_save_model)

        if is_training:
            self.current_epoch += 1

        mean_loss = sum(loss_evol)/len(loss_evol)
        mean_nse = sum(nse_evol)/len(nse_evol)

        return mean_loss, mean_nse, total_time