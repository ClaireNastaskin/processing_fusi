from pathlib import Path

# raw to schema
from mangrove.io.metadata import PoseidonMetadata
from mangrove.io.convert.raw_data_to_schema import metadata_to_schema, preprocess_raw_data

from h5py import File

base_path = Path('fUS_data/fUSI_Rat_acquisition/2024-09-10/Rat_3/Functional_recording/run-06/')
sequence = 'seq-RF_reducedTX_ReducedRX_6500000Hz_1a_3c_200loops_-1Gain_3AFE_3rp'
experiment_folder = base_path / 'acquisitions' / sequence

metadata = PoseidonMetadata.from_folder(experiment_folder / 'metadata')
acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
raw_data_files = (experiment_folder / "raw_frame_data").glob("[!.]*.h5")
for raw_data_file in raw_data_files:
    data, scan_timing_metadata = preprocess_raw_data(
        metadata, raw_data_file, transducer_metadata=transducer_metadata
    )