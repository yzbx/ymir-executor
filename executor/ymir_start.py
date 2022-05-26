import os
import os.path as osp

from cv2 import UMatData_USER_ALLOCATED
from executor import env
import subprocess
import logging
import yaml

def main():
    # step 1. read config.yaml and clone git_url:git_branch to /app
    logging.info('step::1 runing git clone ... ','*'*50)
    executor_config = env.get_executor_config()
    code_config_file = executor_config.get('code_config','')
    if osp.exists(code_config_file):
        with open(code_config_file,'r') as fp:
            code_config = yaml.safe_load(fp)
            for key in code_config:
                if key not in executor_config:
                    executor_config[key]=code_config[key]
                else:
                    user_cfg=executor_config[key]
                    if isinstance(user_cfg,str):
                        if len(user_cfg) !=0:
                            logging.info(f'overwrite {key}, {code_config[key]} --> {executor_config[key]}')
                        else:
                            logging.info(f'invalid value {user_cfg} for {key}')
                    elif user_cfg is not None:
                        logging.info(f'overwrite {key}, {code_config[key]} --> {executor_config[key]}')
                    else:
                        logging.info(f'invalid value {user_cfg} for {key}')
                    
    git_url=executor_config['git_url']
    git_branch=executor_config['git_branch']

    cmd=f'git clone {git_url} -b {git_branch} /app' 
    subprocess.check_output(cmd.split())

    # step 2. read /app/extra-requirements.txt and install it.
    logging.info('step::2 runing pip install ... ','*'*50)
    if osp.exists('/app/extra-requirements.txt'):
        cmd='pip install -r /app/extra-requirements.txt'
        subprocess.check_output(cmd.split())

    # step 3. run /app/start.py 
    logging.info('runing python start.py ','*'*50)
    cmd = 'cd /app && python3 start.py'
    subprocess.check_output(cmd.split())

if __name__ == '__main__':
    main()