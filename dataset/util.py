import torch 
from pprint import pprint

def make_sampling_collate(n: int, **kwargs):
    def _collate(batch):
        '''
        Returns a batch of size n by repeating the single example in the input batch.
        EF is varied but the cond_image is kept the same.
        '''
        assert len(batch) == 1, "Sampling DataLoader must use batch_size=1"
        sample = batch[0]

        video_name = sample['video_name']
        cond_image = sample['cond_image']  # (C, T, H, W)
        ehs = sample['encoder_hidden_states']  # (1, E)

        ef_gen_range = kwargs.get('ef_gen_range', (0,2, 0.8))

        # Repeat data n times along batch dimension
        cond_rep = cond_image.unsqueeze(0).repeat(n, 1, 1, 1, 1)      # (n, C, T, H, W)

        ehs_dim = ehs.shape[-1]
        ef_orig = ehs.flatten()[0] # scalar EF value used to create the original EHS

        if n > 1:
            grid = torch.linspace(ef_gen_range[0], ef_gen_range[1], steps=n - 1, device=ehs.device, dtype=ehs.dtype)
            ef_values = torch.cat([ef_orig.view(1), grid], dim=0)
        else:
            ef_values = ef_orig.view(1)

        ehs_rep = ef_values.view(n, 1, 1).expand(-1, 1, ehs_dim)       # (n, 1, E)

        reference_batch = {'cond_image': cond_image, 'ef_values': ef_values, 'video_name': video_name, 
                           'observed_mask': sample['observed_mask'], 'not_pad_mask': sample['not_pad_mask']}
        repeated_batch = {'cond_image': cond_rep, 'encoder_hidden_states': ehs_rep}
        return reference_batch, repeated_batch
    return _collate

def default_eval_collate(batch):
    """
    Standard collate that stacks samples into a batch (no repetition).
    Returns (reference_batch, batch) to match your existing code structure.
    """
    assert len(batch) > 0, "Empty batch passed to collate"

    # Stack cond_images: expect cond_image shape (C, T, H, W) per sample -> (B, C, T, H, W)
    conds = torch.stack([sample["cond_image"] for sample in batch], dim=0)

    # Stack masks / other tensors
    observed_masks = torch.stack([sample["observed_mask"] for sample in batch], dim=0)
    not_pad_masks = torch.stack([sample["not_pad_mask"] for sample in batch], dim=0)

    # video_name list kept as Python list (strings)
    video_names = [sample["video_name"] for sample in batch]
    target_ef_bin = [sample.get("target_ef_bin", None) for sample in batch]

    # Encoder hidden states: expect (1, E) or (E,) per sample -> make shape (B, 1, E)
    ehs_list = [torch.as_tensor(sample["encoder_hidden_states"]).reshape(1, -1) for sample in batch]
    encoder_hidden_states = torch.stack(ehs_list, dim=0)  # (B, 1, E)

    # Original EF values (assumed encoded as first element in encoder_hidden_states)
    ef_orig = encoder_hidden_states[:, 0, 0].clone()  # (B,)

    reference_batch = {
        "cond_image": conds,             # (B, C, T, H, W)
        "ef_values": ef_orig,            # (B,)
        "video_name": video_names,       # list[str]
        "observed_mask": observed_masks, # (B, ...)
        "not_pad_mask": not_pad_masks,   # (B, ...)
        "target_ef_bin": target_ef_bin   # list[Optional[int]]
    }


    batch_out = {
        "cond_image": conds,                 # (B, C, T, H, W)
        "encoder_hidden_states": encoder_hidden_states,  # (B, 1, E)
    }

    return reference_batch, batch_out