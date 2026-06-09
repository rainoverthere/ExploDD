# A quick start !

This repository provides a quick-start guide for running explosion detection on demo sequences from ExploDD using Prophesee Metavision SDK 4.6.2, our modified site-packages, and a pretrained model.

---

## Demo Video

<video src="demo_25fps_400x_slow_motion.mp4" controls width="100%"></video>

[View the demo video](demo_25fps_400x_slow_motion.mp4)

---

## Environment Setup

1. **Obtain Prophesee SDK 4.6.2**
   - Apply for access to the SDK at [Prophesee SDK Download](https://www.prophesee.ai/metavision-intelligence-sdk-download/) and obtain the installer.

2. **Install the SDK**
   - Run the installer and follow the prompts to install SDK 4.6.2 (for example, on Windows, install to `C:\Program Files\Prophesee`).

3. **Replace Python Site-Packages**
   - Replace the folder:
     ```
     C:\Program Files\Prophesee\lib\python3\site-packages
     ```
     with the provided modified `site-packages` folder.  
     > Note: The provided folder includes custom codes for explosion detection.

4. **Set up Python 3.9.21 and Dependencies**
   - Follow the official instructions to initialize Python environment: [Installation on Windows](https://docs.prophesee.ai/4.6.2/installation/windows.html#chapter-installation-windows)
   - Ensure Python 3.9.21 is installed.
   - Download necessary components (e.g., `ffmpeg`).

5. **Create Conda Environment**
   - Use the provided `requirements.txt` to create the complete environment:
     ```bash
     conda create --name new_env --file requirements.txt
     ```

---

## Prediction (3 real demo sequences from ExploDD)

1. **Run Prediction Script**
   ```bash
   python train_detection.py path/to/out_dir path/to/demo_seqs \
       --num_tbins 4 \
       --batch_size 8 \
       --just_demo \
       --checkpoint path/to/real-ref-epoch=50-step=5424.ckpt
   ```

2. **View Results**
   - Predicted videos are saved in:
     ```
     path/to/out_dir/videos/video#-1.mp4
     ```

   Note: The event frames have an equivalent frame rate of 10,000 FPS, while the output video (`output_video#-1.mp4`) is displayed at 25 FPS, resulting in a 400× slower visualization. 

---

## Notes

- The environment and SDK have been customized for explosion detection. Do **not** use the original `site-packages` folder.
- Make sure all Python dependencies are installed as specified in `requirements.txt`.
- Complete ExploDD has been deposited in Zenodo and is openly accessible at https://doi.org/10.5281/zenodo.20603099.
- All related pre-trained model resources are available at https://drive.google.com/file/d/1kjf3tGkZ5MJaqmkOCa-uCvX_xLcxGKkZ/view?usp=drive_link.
- The `preprocessing` folder contains the code used to preprocess ExploDD raw data: `generate_hdf5.py` converts event stream raw data into HDF5 format, while `from_YOLO_to_npy.py` converts LabelImg raw labels (txt format) into sequence-level npy format. Note that `generate_hdf5.py` defaults to real event stream raw data (`.raw` / EVT3.0 format) as input. If you want to process synthetic raw data, please go to `site-packages/metavision_ml/preprocessing/hdf5.py`, uncomment the `write_to_hdf5_npy` function at line 86, and comment out the default `write_to_hdf5` function at line 71. 
- As this repository and ExploDD is built upon Prophesee SDK 4.6.2, please refer to the [official Prophesee documentation](https://docs.prophesee.ai/4.6.2/index.html) for more details.
