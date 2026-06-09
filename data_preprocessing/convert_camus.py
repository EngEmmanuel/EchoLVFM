from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import imageio
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", ".*macro_block_size.*", category=UserWarning)

def convert_camus(input_dir, output_dir, split_dir=None, label_map=None):
    """
    Convert CAMUS dataset:
    - Output each view in its own folder: output_dir/{patient}_{view}
    - NIfTI half-sequences -> MP4 videos (rotated + flipped)
    - NIfTI mask sequences -> per-frame PNG masks (rotated, flipped, and optionally relabeled)
    - Extract metadata into metadata.csv with optional split column
    - label_map: dict mapping original mask values -> new values
      e.g. {0:0, 1:2, 2:255, 3:1}

    Only the .mp4 videos are required for the latent-encoding pipeline
    (vae/convert_to_latent_dataset.py); the masks/EF outputs are used elsewhere.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build split lookup
    split_map = {}
    if split_dir:
        split_dir = Path(split_dir)
        for fname, lab in [
            ("subgroup_training.txt","TRAIN"), # text files where each line is 'patientWXYZ'
            ("subgroup_validation.txt","VAL"),
            ("subgroup_testing.txt","TEST")
        ]:
            fp = split_dir / fname
            if fp.exists():
                for line in fp.read_text().splitlines():
                    pid = line.strip()
                    if pid:
                        split_map[pid] = lab

    metadata = []

    for patient_dir in tqdm(list(input_dir.iterdir()), desc="Patients"):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        split_label = split_map.get(pid, "")

        for view in ("2CH","4CH"):
            cfg = patient_dir / f"Info_{view}.cfg"
            seq_nii = patient_dir / f"{pid}_{view}_half_sequence.nii.gz"
            m_nii   = patient_dir / f"{pid}_{view}_half_sequence_gt.nii.gz"
            ed_nii  = patient_dir / f"{pid}_{view}_ED_gt.nii.gz"
            es_nii  = patient_dir / f"{pid}_{view}_ES_gt.nii.gz"
            if not (cfg.exists() and seq_nii.exists() and m_nii.exists()):
                continue

            # parse cfg
            params = {}
            for L in cfg.read_text().splitlines():
                if ":" in L:
                    k,v = L.split(":",1)
                    params[k.strip()] = v.strip()
            ED      = int(params.get("ED",0))
            ES      = int(params.get("ES",0))
            NbFrame = int(params.get("NbFrame",0))
            Sex     = params.get("Sex","")
            Age     = int(params.get("Age",0))
            IQ      = params.get("ImageQuality","")
            EF      = float(params.get("EF",0.0))
            FR      = float(params.get("FrameRate",0.0))

            # load volumes
            seq   = nib.load(str(seq_nii)).get_fdata().astype(np.uint8) #(H,W,T)
            masks = nib.load(str(m_nii)).get_fdata().astype(np.uint8)
            ed_mask = nib.load(str(ed_nii)).get_fdata().astype(np.uint8)
            es_mask = nib.load(str(es_nii)).get_fdata().astype(np.uint8)

            # --- Reversal logic for ES→ED sequences ---
            if ED > ES:
                assert ED == NbFrame, f"Unexpected: ED={ED}, NbFrame={NbFrame} for {pid}-{view}"
                # Reverse temporal axis (last dimension)
                seq   = np.flip(seq,   axis=-1)
                masks = np.flip(masks, axis=-1)

                # Update ED/ES indices after reversal
                ED, ES = 1, NbFrame  # ED now at start, ES at end

            # prepare output paths
            out_dir   = output_dir / f"{pid}_{view}"
            masks_dir = out_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            # write video
            vid_path = out_dir / f"{pid}_{view}.mp4"
            writer = imageio.get_writer(str(vid_path), format="ffmpeg", mode="I", fps=FR)
            for t in tqdm(range(seq.shape[-1]), desc=f"Writing video {pid}-{view}", leave=False):
                frm = seq[...,t]
                # rotate 90° CW then flip L/R
                frm = np.fliplr(np.rot90(frm, k=-1))
                writer.append_data(frm)
            writer.close()

            # write ED/ES masks
            if label_map:
                ed2 = np.zeros_like(ed_mask)
                es2 = np.zeros_like(es_mask)
                for orig, new in label_map.items():
                    ed2[ed_mask == orig] = new
                    es2[es_mask == orig] = new
                ed_mask = ed2
                es_mask = es2
            # rotate & flip
            ed_mask = np.fliplr(np.rot90(ed_mask, k=-1))
            es_mask = np.fliplr(np.rot90(es_mask, k=-1))
            # save ED/ES masks as PNG
            ed_mask_path = masks_dir / f"{pid}_{view}_ED_mask.png"
            es_mask_path = masks_dir / f"{pid}_{view}_ES_mask.png"
            imageio.imwrite(str(ed_mask_path), (ed_mask).astype(np.uint8))
            imageio.imwrite(str(es_mask_path), (es_mask).astype(np.uint8))

            # write masks with optional remapping
            for t in tqdm(range(masks.shape[-1]), desc=f"Writing masks {pid}-{view}", leave=False):
                m = masks[...,t].astype(int)
                # apply mapping in one pass
                if label_map:
                    m2 = np.zeros_like(m)
                    for orig, new in label_map.items():
                        m2[m==orig] = new
                    m = m2
                # rotate & flip
                m = np.fliplr(np.rot90(m, k=-1))
                # save as 8-bit
                out_png = masks_dir / f"{pid}_{view}_mask_{t:03d}.png"
                imageio.imwrite(str(out_png), (m).astype(np.uint8))

            # record metadata
            metadata.append({
                "video_name": f"{pid}_{view}",
                "view": view,
                "ED": ED,
                "ES": ES,
                "NbFrame": NbFrame,
                "Sex": Sex,
                "Age": Age,
                "ImageQuality": IQ,
                "EF": EF,
                "FrameRate": FR,
                "split": split_label
            })


    # write metadata.csv
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Conversion done: metadata at {output_dir/'metadata.csv'}")




if __name__ == "__main__":
    input_directory = ''
    output_directory = ''
    split_directory = ''
    label_map = {0: 0, 1: 2, 2: 255, 3: 1}  # Example mapping
    # Original labels: 0=background, 1=LV, 2=LV_epi, 3=LA
    # My labels: 0=background, 1=LA, 2=LV, 255=LV_epi
    convert_camus(input_directory, output_directory, split_directory, label_map)
