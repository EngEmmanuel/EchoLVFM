# Videos

This folder stores sample and demo videos for EchoLVFM.

## How to Upload Videos Without Cloning

You can add videos directly through GitHub's web interface — no Git installation or coding required:

1. **Navigate to this folder** in the GitHub repository:  
   `https://github.com/EngEmmanuel/EchoLVFM/tree/main/videos`

2. Click **"Add file"** → **"Upload files"** (top-right area of the file list).

3. **Drag and drop** your video files onto the upload area, or click **"choose your files"** to browse.

4. Scroll down to the **"Commit changes"** section, optionally add a description, then click **"Commit changes"**.

Your videos will be added to the repository immediately after committing.

> **Note:** GitHub has a **100 MB per-file limit** for files uploaded via the web interface.  
> For larger files, [Git LFS](https://git-lfs.github.com/) is configured in this repository (see `.gitattributes` at the root) and supports files up to 2 GB.  
> Large files require a local Git + LFS setup to upload.

## Supported Formats

Video files tracked by Git LFS in this repository:

| Extension | Format |
|-----------|--------|
| `.mp4`    | MPEG-4 |
| `.avi`    | AVI    |
| `.mov`    | QuickTime |
| `.mkv`    | Matroska |
| `.gif`    | Animated GIF |
| `.webm`   | WebM   |
