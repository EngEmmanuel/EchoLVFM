import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from torchdiffeq import odeint
from einops import rearrange, repeat
from src.custom_loss import MaskedMSELoss
# Code adapted from https://github.com/lucidrains/rectified-flow-pytorch/blob/main/rectified_flow_pytorch/rectified_flow.py

def identity(t):
    return t

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers
def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))



class LinearFlow(Module):
    def __init__(
            self,
            model,
            data_shape: tuple[int, ...] | None = None,
            clip_values: tuple[float, float] | None = None,
            clip_flow_values: tuple[float, float] | None = None,
            **kwargs
    ):
        super().__init__()
        self.model = model

        self.data_shape = data_shape
        self.noise_schedule = lambda x: x
        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        self.loss_fn = MaskedMSELoss()
        # objective - either flow or noise. CHOSE TO PREDICT FLOW
        # self.predict = predict

    @property
    def device(self):
        return next(self.model.parameters()).device

    def sample_times(self, batch):
        pass

    @torch.no_grad()
    def sample(
        self, 
        encoder_hidden_states: torch.Tensor, 
        batch_size=1, 
        steps=16, 
        noise=None, 
        data_shape: tuple[int, ...] | None = None,
        cond_image=None, 
        mask=None,
        guidance_scale: float = 1.0,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        use_ema: bool = False,
        **model_kwargs
    ):
        
        model = self.model
        data_shape = default(data_shape, self.data_shape)

        print(f'Sampling with steps={steps}, batch_size={batch_size}, guidance_scale={guidance_scale}')

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_values is not None else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_values is not None else identity

        # Backward-compatible lookup for learned null embedding: prefer flow.null_ehs, fallback to base model.null_ehs
        uncond_ehs = getattr(self, "null_ehs", None)
        if uncond_ehs is None:
            uncond_ehs = getattr(self.model, "null_ehs", None)
        if uncond_ehs is not None:
            # Get underlying tensor
            uncond = uncond_ehs.data if isinstance(uncond_ehs, torch.nn.Parameter) else uncond_ehs
            # Try to match encoder_hidden_states shape (excluding batch)
            target_tail = tuple(encoder_hidden_states.shape[1:]) if hasattr(encoder_hidden_states, 'shape') else None
            if target_tail and uncond.shape != target_tail:
                try:
                    uncond = uncond.view(*target_tail)
                except Exception:
                    # leave as-is; expand best-effort below
                    pass
            # Expand along batch dimension
            if uncond.dim() == 0:
                uncond = uncond.view(1, 1)
            if uncond.dim() == 1:
                uncond_ehs = uncond.unsqueeze(0).expand(batch_size, -1)
            elif uncond.dim() == 2:
                uncond_ehs = uncond.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                uncond_ehs = uncond.unsqueeze(0).expand(batch_size, *uncond.shape)

        def _predict(x, t, ehs):
            return self.predict_flow(
                model,
                x,
                times=t,
                encoder_hidden_states=ehs,
                cond_image=cond_image,
                mask=mask,
                **model_kwargs,
            )

        def ode_fn(t, x):
            x = maybe_clip(x)

            if guidance_scale <= 1.0: # No CFG
                flow = _predict(x, t, encoder_hidden_states)
            else:
                if uncond_ehs is None:
                    raise ValueError(
                        "guidance_scale > 1.0 requires a learned null EF embedding. "
                        "Either this model was not trained for CFG or you need to" \
                        "Attach `null_ehs` to the flow (e.g., during checkpoint load)."
                    )

                flow_cond = _predict(x, t, encoder_hidden_states)
                flow_uncond = _predict(x, t, uncond_ehs)
                flow = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
            return maybe_clip_flow(flow)

        # Start with random gaussian noise - y0
        noise = default(noise, torch.randn(batch_size, *data_shape, device=self.device))

        # time steps
        time_steps = torch.linspace(0., 1., steps, device=self.device)

        # ode
        trajectory = odeint(ode_fn, noise, time_steps, **odeint_kwargs)

        sampled_data = trajectory[-1]  # Get the last state as the sampled data
        
        return sampled_data
    
    # Keep model arg in case of ema
    def predict_flow(self,
                    model:Module, 
                    noised, 
                    *, 
                    times, 
                    encoder_hidden_states=None, 
                    cond_image=None, 
                    mask=None, 
                    eps=1e-10, 
                    **model_kwargs
                ):

        batch = noised.shape[0]
        
        # Prepare time conditioning for model
        times = rearrange(times, '... -> (...)') # Flattens times

        if times.numel() == 1:
            times = repeat(times, '1 -> b', b = batch)

        # Unet and STDiT forward(x, timestep, encoder_hidden_states=None, cond_image=None, mask=None, return_dict=True)
        output = self.model(x=noised, timestep=times, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image, mask=mask, **model_kwargs) # predicted flow / velocity field
        if hasattr(output, 'sample'):
            return output.sample
        return output


    def forward(
            self, 
            x, 
            encoder_hidden_states: torch.Tensor, 
            noise: Tensor | None = None,
            cond_image=None, 
            mask=None,
            loss_mask=None,
            **model_kwargs
        ):

        batch, *data_shape = x.shape
        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data
        noise = default(noise, torch.randn_like(x))

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, x.ndim - 1)

        def get_noised_and_flows(model, t):

            # maybe noise schedule
            t = self.noise_schedule(t)
            noised = x * t + noise * (1 - t)

            flow = x - noise
            
            pred_flow = self.predict_flow(model, noised, times=t, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image, **model_kwargs)

            pred_x = noised + pred_flow * (1 - t)

            return flow, pred_flow, pred_x

        # getting flow and pred flow for main model
        flow, pred_flow, pred_x = get_noised_and_flows(self.model, padded_times)

        main_loss = self.loss_fn(pred_flow, flow, loss_mask) #, pred_data = pred_x, times = times, data = x)

        return main_loss