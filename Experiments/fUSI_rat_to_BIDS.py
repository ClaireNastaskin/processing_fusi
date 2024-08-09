from pathlib import Path
from shutil import copyfile
import h5py
from tqdm import tqdm
import numpy as np
import re
import json
import pandas as pd
import nibabel as nib
from contextlib import redirect_stdout

import anise.io.behavior_loader
import anise.process.image_metrics
import anise.utils

import matplotlib.pyplot as plt
from PIL import Image

root = Path('fUS_data') / 'fUSI_Rat_acquisition'
task = 'light'
overwrite = False

out_dir = Path('fUS_data') / 'fUSI_Rat_BIDS'
out_dir.mkdir(exist_ok=True)

# %%
# First find all power doppler processed folders

pd_dirs = [p for p in tqdm(root.rglob('*')) if
           p.is_dir() and p.stem.lower() == 'power_doppler']
    
n_tcs = {pd_dir: set([anise.utils.get_param('num_tissue_components', p.stem)
         for p in pd_dir.glob('*.h5')])
         for pd_dir in tqdm(pd_dirs)}

# %%
# Now, assemble a dataframe with parameters

if (out_dir / 'experiment_data.csv').exists() and not overwrite:
    df = pd.read_csv(out_dir / 'experiment_data.csv')
else:
    df = pd.DataFrame(columns=[
        'power_doppler_path',
        'bmode_fname', 'power_doppler_fname',
        'bmode_img_fname', 'power_doppler_img_fname',
        'dataset', 'rat', 'run', 'sequence', 'plane', 'n_tissue_components',
        'angles', 'frequency', 'az_aperture', 'el_aperture',
        'gain', 'power_mode', 'cycles', 'metadata'])
if (out_dir / 'excluded_data.csv').exists() and not overwrite:
    df_ex = pd.read_csv(out_dir / 'excluded_data.csv')
else:
    df_ex = pd.DataFrame(columns=['power_doppler_path', 'reason'])

with open(out_dir / 'log.txt', "a") as fid:
    for pd_dir in tqdm(pd_dirs):
        if pd_dir in df['power_doppler_path'].values and not overwrite:
            continue
        if pd_dir in df_ex['power_doppler_path'].values and not overwrite:
            continue
            # code to try again
            # df_ex = df_ex.drop(df_ex[df_ex['power_doppler_path'].values == pd_dir].index)
        ses = pd_dir.relative_to(root).parts[0].replace('-', '')
        this_n_tcs = n_tcs[pd_dir]

        # get events
        streams_path = pd_dir / '..' / '..' / '..' / 'streams'
        events_path = streams_path / 'task_1-event_stream.h5'
        if not events_path.exists():
            events_path = streams_path / 'daq_1-event_stream.h5'
        if not events_path.exists():
            for filename in streams_path.glob('*'):
                use_as_events = False
                # search for events structured h5 file
                try:
                    with h5py.File(filename, 'r') as file:
                        use_as_events = file['data'][:][0][0].decode(
                            'utf-8').startswith('pwm')
                except Exception as _:
                    pass
                if use_as_events:
                    events_path = filename
        if not events_path.exists():
            df_ex.loc[len(df_ex.index)] = pd_dir, 'no events'
            continue
        with redirect_stdout(fid):
            events = anise.io.behavior_loader.behavior_loader(events_path)
        events['trial_type'] = events['trial_type'].replace(
            'pwm_enable_command_sent', 'pwm_enabled'
        )

        # get bmode file structure
        with redirect_stdout(fid):
            bmode_info = anise.utils.load_fusi_info(pd_dir / '..' / 'beamformed')
        
        if bmode_info is None:
            df_ex.loc[len(df_ex.index)] = pd_dir, 'no bmodes'
            continue

        power_doppler_infos = dict()
        for n_tc in this_n_tcs:
            with redirect_stdout(fid):
                power_doppler_info = anise.utils.load_fusi_info(
                    pd_dir, num_tissue_components=n_tc)
                power_doppler_infos[n_tc] = power_doppler_info
        
        if any([info is None for info in power_doppler_infos.values()]):
            df_ex.loc[len(df_ex.index)] = pd_dir, 'no power doppler'
            continue

        # get parameters
        rat = anise.utils.match_param('rat', pd_dir.relative_to(root))
        rat = '001' if rat is None else rat  # default
        run = anise.utils.match_param('run', pd_dir.relative_to(root))
        run = '01' if run is None else run  # default
        seq = anise.utils.match_param('seq', pd_dir.relative_to(root))
        seq = pd_dir.parts[-2] if seq is None else seq
        depth_min_match = re.search('[0-9]+Dmin', seq)
        depth_min = int(depth_min_match.group().replace('Dmin', '')) if \
            depth_min_match else 'n/a'
        depth_max_match = re.search('[0-9]+Dmax', seq)
        depth_max = int(depth_max_match.group().replace('Dmax', '')) if \
            depth_max_match else 'n/a'
        planes = [part for part in pd_dir.relative_to(root).parts if 'plane' in part]
        plane = planes[0].replace('plane', '').replace('_', '') \
            if len(planes) == 1 else '1'

        bmode_dir = out_dir / 'rawdata' / f'sub-{rat}' / f'ses-{ses}' / 'fus'
        bmode_img_dir = out_dir / 'derivatives' / 'bmode_images' / \
            f'sub-{rat}' / f'ses-{ses}'
        fus_dir = out_dir / 'sourcedata' / f'sub-{rat}' / f'ses-{ses}' / 'fus'
        fus_img_dir = out_dir / 'derivatives' / 'power_doppler_images' / \
            f'sub-{rat}' / f'ses-{ses}'
        beh_dir = out_dir / 'sourcedata' / f'sub-{rat}' / f'ses-{ses}' / 'beh'
        for this_dir in (bmode_dir, bmode_img_dir, fus_dir, fus_img_dir, beh_dir):
            this_dir.mkdir(parents=True, exist_ok=True)
    
        bmode_base = (f'sub-{rat}_ses-{ses}_task-{task}_acq-{{acq}}'
                      f'_run-{run}_pose-{plane}')
        bmode_fname = None
        acq = -1
        while bmode_fname is None or bmode_fname.exists():
            acq += 1
            bmode_fname = bmode_dir / \
                (bmode_base.format(acq=acq) + f'_idx-0_bmode.h5')

        # check for mismatches
        mismatch = bmode_shape_check = tr_check = tr_compound_check = None
        bad_idxs = list()
        bmode_data = list()
        for i, filename in enumerate(bmode_info['Filename']):
            if not anise.utils.is_valid_hdf5(pd_dir / '..' / 'beamformed' / filename):
                bad_idxs.append(i)
                continue
            with h5py.File(pd_dir / '..' / 'beamformed' / filename, 'r') as file:
                bmode_data.append(file['beamformed'][:].mean(axis=0))
                bmode_shape = file['beamformed'].shape
                if 'metadata' in file and 'sequence' in file['metadata']:
                    tr = file['metadata']['sequence'][
                        'pulse_repetition_interval_s'][()]
                    tr_compound = file['metadata']['sequence'][
                        'compound_bmode_repetition_interval_s'][()]
                else:
                    tr = tr_compound = 'n/a'
                if tr_check is not None:
                    if bmode_shape != bmode_shape_check:
                        mismatch = 'bmode shape'
                    if tr != tr_check:
                        mismatch = 'pulse repetition interval'
                    if tr_compound != tr_compound_check:
                        mismatch = 'compound bmode repetition interval'
                    if mismatch is not None:
                        break
                bmode_shape_check = bmode_shape
                tr_check = tr
                tr_compound_check = tr_compound

        if mismatch is not None:
            df_ex.loc[len(df_ex.index)] = pd_dir, f'inconsistent {mismatch}'
            continue

        # some files corrupted
        if bad_idxs:
            bmode_info = bmode_info.drop(index=bad_idxs)

        # write bmodes
        for idx, from_fname in enumerate(bmode_info['Filename']):
            to_fname = bmode_dir / \
                (bmode_base.format(acq=acq) + f'_idx-{idx}_bmode.h5')
            if overwrite or not to_fname.exists():
                copyfile(pd_dir / '..' / 'beamformed' / from_fname, to_fname)
            bmode_info.loc[idx, 'BIDSFilename'] = to_fname
        bmode_info[['BIDSFilename', 'Experiment Time']].to_csv(
            bmode_dir / f'{bmode_base.format(acq=acq)}_bmode.tsv',
            sep='\t', index=False
        )

        # store bmode image plot
        bmode_idx = anise.process.image_metrics.find_median_image_idx(bmode_data)
        bmode_img_fname = bmode_img_dir / f'{bmode_base.format(acq=acq)}_bmode.png'
        if overwrite or not bmode_img_fname.exists():
            for ext in ('png', 'jpg'):
                bmode_img = pd_dir / '..' / 'beamformed' / \
                    bmode_info.loc[bmode_idx, 'Filename'].replace('.h5', f'.{ext}')
                if bmode_img.exists():
                    break
            bmode_img = Image.open(bmode_img)
            bmode_img.save(bmode_img_fname)

        """
        with redirect_stdout(fid):
            bmode_nii, time_stamps = anise.utils.get_bmode_nii(
                pd_dir / '..' / 'beamformed')

        nib.save(bmode_nii, bmode_fname)

        # add sidecar
        with open(func_dir / (fname.format(acq) + '_bmode.json'), 'w') as fid2:
            fid2.write(json.dumps(sidecar, indent=4))
        """

        for n_tc in this_n_tcs:
            pd_base = (f'sub-{rat}_ses-{ses}_task-{task}_acq-{acq}_'
                       f'run-{run}_pose-{plane}_proc-svd{n_tc}ntc')
            with redirect_stdout(fid):
                power_doppler_nii, time_stamps = \
                    anise.utils.get_power_doppler_nii(pd_dir, n_tc)
            power_doppler_data = np.array([
                h5py.File(pd_dir / pd)['power_doppler'][:]
                for pd in power_doppler_infos[n_tc]['Filename']
            ])
            power_doppler_idx = anise.process.image_metrics.find_median_image_idx(
                power_doppler_data)
            pd_img_fname = fus_img_dir / f'{pd_base}.png'
            # ignore extra elevation kwargs
            from_pd_img_fname = power_doppler_infos[n_tc].loc[
                power_doppler_idx, 'Filename'].replace('.h5', '')
            if overwrite or not pd_img_fname.exists():
                pd_img = Image.open(
                    [f for ext in ('png', 'jpg') for f in
                     pd_dir.glob(f'{from_pd_img_fname}*.{ext}')][0]
                )
                pd_img.save(pd_img_fname)

            if overwrite or not (fus_dir / f'{pd_base}_pwdt.nii.gz').exists():
                # save nii
                nib.save(power_doppler_nii, fus_dir / f'{pd_base}_pwdt.nii.gz')

                # add sidecar
                sidecar = anise.utils.get_sidecar(
                    pd_dir / '..' / 'beamformed' / bmode_info.loc[0, 'Filename'],
                    pd_dir / power_doppler_infos[n_tc].loc[0, 'Filename'],
                    time_stamps, n_tc)
                with open(fus_dir / f'{pd_base}_pwdt.json', 'w') as fid2:
                    fid2.write(json.dumps(sidecar, indent=4))

            df.loc[len(df.index)] = (
                pd_dir,
                bmode_dir / (bmode_base.format(acq=acq) + f'_idx-0_bmode.h5'),
                fus_dir / f'{pd_base}_pwdt.nii.gz',
                bmode_img_fname, pd_img_fname,
                ses, rat, run, seq, plane, n_tc,
                sidecar['PlaneWaveAngles'],
                sidecar['ProbeCentralFrequency'],
                sidecar['TxApertureMask'],
                sidecar['TxElevationMask'],
                sidecar['Gain'],
                sidecar['AFE'],
                sidecar['TxCycles'],
                sidecar
            )

        events_fname = beh_dir / (f'sub-{rat}_ses-{ses}_task-{task}_'
                                  f'run-{run}_pose-{plane}_events.tsv')
        if overwrite or not events_fname.exists():
            events[['onset', 'duration', 'trial_type']].to_csv(
                events_fname, sep='\t', index=False
            )

        df.to_csv(out_dir / 'experiment_data.csv', index=False)
        df_ex.to_csv(out_dir / 'excluded_data.csv', index=False)
