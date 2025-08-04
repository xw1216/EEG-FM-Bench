import os.path

 # the middle cache and arrow proc data path can be remote s3

PLATFORM = 'local'
RUN_ROOT = './assets/run'
LOG_ROOT = './assets/run/log'
CONF_ROOT = './assets/conf'

DATABASE_RAW_ROOT = './dataset'
DATABASE_PROC_ROOT = './arrow'
DATABASE_CACHE_ROOT = './cache'


def get_conf_file_path(path):
    if os.path.isabs(path):
        return path
    elif os.sep in path:
        return os.path.join(CONF_ROOT, os.path.normpath(path))
    else:
        return os.path.join(CONF_ROOT, path)


def create_parent_dir(path):
    par_dir = os.path.dirname(path)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir, exist_ok=True)
