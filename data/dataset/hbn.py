import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne_bids
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame
from tqdm import tqdm

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder
from mne_bids import BIDSPath


logger = logging.getLogger('preproc')


@dataclass
class HBNConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This dataset comprises electroencephalogram (EEG) data and behavioral responses collected during EEG "
        "experiments from >3000 participants (5-21 yo) involved in the HBN project. The data has been released in "
        "11 separate Releases, each containing data from a different set of participants.\n"
        "The HBN-EEG dataset includes EEG recordings from participants performing six distinct tasks, which are "
        "categorized into passive and active tasks based on the presence of user input and interaction in the experiment.\n"
        "Passive Tasks\n"
        "Resting State: Participants rested with their heads on a chin rest, following instructions to open or close "
        "their eyes and fixate on a central cross.\n"
        "Surround Suppression: Participants viewed flashing peripheral disks with contrasting backgrounds, while "
        "event markers and conditions were recorded.\n"
        "Movie Watching: Participants watched four short movies with different themes, with event markers recording "
        "the start and stop times of presentations.\n"
        "Active Tasks\n"
        "Contrast Change Detection: Participants identified flickering disks with dominant contrast changes and received"
        "feedback based on their responses.\n"
        "Sequence Learning: Participants memorized and repeated sequences of flashed circles on the screen, designed "
        "for different age groups.\n"
        "Symbol Search: Participants performed a computerized symbol search task, identifying target symbols "
        "from rows of search symbols.\n"
        "Contents\n"
        "EEG Data: High-resolution EEG recordings capture a wide range of neural activity during various tasks. "
        "Behavioral Responses: Participant responses during EEG tasks, including reaction times and accuracy.\n"
        "This data was originally recorded within the behavior directory of the HBN data. The data is now included "
        "with the EEG data within the events.tsv files.\n")
    citation: Optional[str] = """\
    @article{alexander_open_2017,
	title = {An open resource for transdiagnostic research in pediatric mental health and learning disorders},
	volume = {4},
	issn = {2052-4463},
	url = {https://doi.org/10.1038/sdata.2017.181},
	doi = {10.1038/sdata.2017.181},
	number = {1},
	journal = {Scientific Data},
	author = {Alexander, Lindsay M. and Escalera, Jasmine and Ai, Lei and Andreotti, Charissa and Febre, 
	Karina and Mangone, Alexander and Vega-Potler, Natan and Langer, Nicolas and Alexander, Alexis and Kovacs, 
	Meagan and Litke, Shannon and O'Hagan, Bridget and Andersen, Jennifer and Bronstein, Batya and Bui, 
	Anastasia and Bushey, Marijayne and Butler, Henry and Castagna, Victoria and Camacho, Nicolas and Chan, 
	Elisha and Citera, Danielle and Clucas, Jon and Cohen, Samantha and Dufek, Sarah and Eaves, Megan and Fradera, 
	Brian and Gardner, Judith and Grant-Villegas, Natalie and Green, Gabriella and Gregory, Camille and Hart, 
	Emily and Harris, Shana and Horton, Megan and Kahn, Danielle and Kabotyanski, Katherine and Karmel, 
	Bernard and Kelly, Simon P. and Kleinman, Kayla and Koo, Bonhwang and Kramer, Eliza and Lennon, Elizabeth and Lord, 
	Catherine and Mantello, Ginny and Margolis, Amy and Merikangas, Kathleen R. and Milham, Judith and Minniti, 
	Giuseppe and Neuhaus, Rebecca and Levine, Alexandra and Osman, Yael and Parra, Lucas C. and Pugh, Ken R. and 
	Racanello, Amy and Restrepo, Anita and Saltzman, Tian and Septimus, Batya and Tobe, Russell and Waltz, 
	Rachel and Williams, Anna and Yeo, Anna and Castellanos, Francisco X. and Klein, Arno and Paus, Tomas and 
	Leventhal, Bennett L. and Craddock, R. Cameron and Koplewicz, Harold S. and Milham, Michael P.},
	month = dec,
	year = {2017},
	pages = {170181},
    }
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'hbn'
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    file_ext: str = 'set'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'EGI_128': [
                                        'E22', 'E15', 'E9',
                                'E26', 'E23', 'E16', 'E3', 'E2',
            'E32', 'E33', 'E27', 'E24', 'E19', 'E11', 'E4', 'E124', 'E123', 'E122', 'E1',
            'E38', 'E34', 'E28', 'E29', 'E13', 'E6', 'E112', 'E111', 'E117', 'E116', 'E121',
                'E44', 'E45', 'E41', 'E36', 'E30', 'E105', 'E104', 'E103', 'E108', 'E114',
            'E57', 'E46', 'E47', 'E42', 'E37', 'E55', 'E87', 'E93', 'E98', 'E102', 'E100',
             'E64', 'E58', 'E51', 'E52', 'E60', 'E62', 'E85', 'E92', 'E97', 'E96', 'E95',
                                'E65', 'E67', 'E72', 'E77', 'E90',
                                        'E70', 'E75', 'E83'
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.0
    wnd_div_sec: int = 15
    suffix_path: str = "HBN"
    scan_sub_dir: str = "data"
    writer_batch_size: int = 256

    task: list[str] = field(default_factory=lambda: [
        'thepresent', 'symbolsearch', 'surroundsupp', 'seqlearning6target', 'seqlearning8target',
        'restingstate', 'funwithfractals', 'diaryofawimpykid', 'despicableme', 'contrastchangedetection'
    ])

    # release: list[int] = field(default_factory=lambda: [1])
    release: list[int] = field(default_factory=lambda: [i for i in range(1, 11)])
    category: list[str] = field(default_factory=lambda: [])


class HBNBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = HBNConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        # Correspondence from https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_eeg/BP_EGI_Compatibility_Comparison_V002.pdf
        self.mapping_dict = {
        'E22': 'FP1',
        'E15': 'FPZ',
        'E9': 'FP2',

        'E26': 'AF7',
        'E23': 'AF3',
        'E16': 'AFZ',
        'E3': 'AF4',
        'E2': 'AF8',

        'E32': 'F9',
        'E33': 'F7',
        'E27': 'F5',
        'E24': 'F3',
        'E19': 'F1',
        'E11': 'FZ',
        'E4': 'F2',
        'E124': 'F4',
        'E123': 'F6',
        'E122': 'F8',
        'E1': 'F10',

        'E38': 'FT9',
        'E34': 'FT7',
        'E28': 'FC5',
        'E29': 'FC3',
        'E13': 'FC1',
        'E6': 'FCZ',
        'E112': 'FC2',
        'E111': 'FC4',
        'E117': 'FC6',
        'E116': 'FT8',
        'E121': 'FT10',

        'E44': 'T9',
        'E45': 'T7',
        'E41': 'C5',
        'E36': 'C3',
        'E30': 'C1',
        'E105': 'C2',
        'E104': 'C4',
        'E103': 'C6',
        'E108': 'T8',
        'E114': 'T10',

        'E57': 'TP9',
        'E46': 'TP7',
        'E47': 'CP5',
        'E42': 'CP3',
        'E37': 'CP1',
        'E55': 'CPZ',
        'E87': 'CP2',
        'E93': 'CP4',
        'E98': 'CP6',
        'E102': 'TP8',
        'E100': 'TP10',

        'E64': 'P9',
        'E58': 'P7',
        'E51': 'P5',
        'E52': 'P3',
        'E60': 'P1',
        'E62': 'PZ',
        'E85': 'P2',
        'E92': 'P4',
        'E97': 'P6',
        'E96': 'P8',
        'E95': 'P10',

        'E65': 'PO7',
        'E67': 'PO3',
        'E72': 'POZ',
        'E77': 'PO4',
        'E90': 'PO8',

        'E70': 'O1',
        'E75': 'OZ',
        'E83': 'O2',
    }

    def preproc(self, n_proc: Optional[int] = None):
        if self._is_preproc_cached():
            logger.info(f'Using cached summary info at {self.info_csv_path}')
            return

        if self.config.is_remote_fs:
            self._run_func_parallel(self._s3_link_test, [None], desc='Testing S3')

        np.random.seed(self.config.seed)
        self.clean_disk_cache()
        self.create_dir_structure()

        self._fix_channel_tsv(n_proc)
        data_files = self._walk_raw_data_files()
        info_df = self._gather_data_info(data_files, n_proc)
        info_df = self._exclude_wrong_data(info_df, n_proc)
        split_df = self._divide_split(info_df)
        split_df.to_csv(self.info_csv_path, index=False)

        self._generate_middle_files(split_df, n_proc)

        self._mark_preproc_done()

    def _fix_channel_tsv(self, n_proc: Optional[int] = None):
        target_rels = [10]
        logger.info('Checking HBN BIDS channel config...')
        tsv_list = []
        for rel in target_rels:
            if rel not in self.config.release:
                continue

            release_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir, f"HBN{str(rel)}")
            bids_path = BIDSPath(
                datatype="eeg",
                root=release_path,
                suffix='channels',
                extension=".tsv"
            )
            matched_paths = bids_path.match()
            tsv_list.extend([str(path.fpath) for path in matched_paths])

            # for subject in get_entity_vals(release_path, "subject"):
            #     bids_path = BIDSPath(
            #         subject=subject, datatype="eeg", root=release_path,
            #         extension=".tsv", suffix='channels'
            #     )
            #
            #     for path in matched_paths:
            #         tsv_list.append(path.fpath)
            #         # logger.info(f'Find tsv file {path.fpath}')

        self._run_func_parallel(
            self._fix_tsv,
            tsv_list,
            n_proc=n_proc,
            desc='Fixing tsv files'
        )

    @staticmethod
    def _fix_tsv(path: str):
        df = pd.read_csv(str(path), sep='\t', na_values=["n/a"])
        pattern = r'^E(12[0-8]|1[0-1][0-9]|[1-9]?[0-9])$|^Cz$'
        mask = df["name"].str.match(pattern, na=False)

        df["type"] = df["type"].astype(object)
        df["units"] = df["units"].astype(object)
        df.loc[mask, "type"] = "EEG"
        df.loc[mask, "units"] = "uV"

        df.to_csv(str(path), sep='\t', index=False)
        logger.info(f'Fixing channel tsv file at {path}')

    def _walk_raw_data_files(self):
        raw_data_files = []
        logger.info('Parsing HBN BIDS path...')
        for rel in tqdm(self.config.release, total=len(self.config.release),
                        desc='Searching file system', unit='release'):
            release_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir, f"HBN{str(rel)}")
            bids_path = BIDSPath(datatype="eeg", root=release_path, extension=".set")
            all_paths = bids_path.match()
            filtered_paths = [
                str(path.fpath) for path in all_paths
                if path.task and path.task.lower() in self.config.task
            ]
            raw_data_files.extend(filtered_paths)

        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        bids_path = mne_bids.get_bids_path_from_fname(file_path)
        return {
            'subject': bids_path.subject,
            'session': 1 if bids_path.run is None else bids_path.run,
            'task': bids_path.task,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        json_file = file_path.replace('.set', '.json')

        try:
            with open(json_file, 'r') as f:
                supp = json.load(f)
                time = supp['RecordingDuration']
        except FileNotFoundError as e:
            logger.error(f'Could not find {json_file}: {e}')
        except Exception as e:
            logger.error(f'Could not parse {json_file}: {e}')

        info.update({
            'montage': 'EGI_128',
            'time': time,
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]
        else:
            raise NotImplementedError

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [self.mapping_dict[ch] for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            bids_path = mne_bids.get_bids_path_from_fname(file_path)
            raw = mne_bids.read_raw_bids(bids_path, verbose=verbose)
            return raw
        # bids_path = mne_bids.get_bids_path_from_fname(file_path)
        # raw = mne_bids.read_raw_bids(bids_path, verbose=verbose)
        # return raw


if __name__ == "__main__":
    builder = HBNBuilder('pretrain')
    builder.clean_disk_cache()
    builder.preproc(n_proc=6)
    builder.download_and_prepare(num_proc=6)
    dataset = builder.as_dataset()
    print(dataset)
