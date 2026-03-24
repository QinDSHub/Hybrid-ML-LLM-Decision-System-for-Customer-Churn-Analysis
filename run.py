#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import os

if __name__ == '__main__':
    subprocess.run(['python', 'data_process.py', '--data_path', '../clean_data', '--save_path', './'])
    subprocess.run(['python', 'llm_lgb.py', '--save_path', './'])
    subprocess.run(['python', 'metrics.py', '--save_path', './'])
    print("All done!")

if __name__ == '__main__':
    data_path = '../clean_data'
    save_path = './'
    
    # run data_process.py
    print("=" * 20)
    print("Running data_process.py...")
    print("=" * 20)
    result1 = subprocess.run([
        'python', 'data_process.py',
        '--data_path', data_path,
        '--save_path', save_path
    ])
    
    if result1.returncode != 0:
        print("data_proces.py failed! Stopping...")
        exit(1)
    
    print("\n" + "=" * 20)
    print("Running llm_lgb.py...")
    print("=" * 20)
    result2 = subprocess.run([
        'python', 'llm_lgb.py',
        '--save_path', save_path
    ])
    
    if result2.returncode != 0:
        print("llm_lgb.py failed! Stopping...")
        exit(1)
    
    print("\n" + "=" * 20)
    print("Running metrics.py...")
    print("=" * 20)
    result3 = subprocess.run([
        'python', 'metrics.py',
        '--save_path', save_path
    ])
    
    if result3.returncode == 0:
        print("\n" + "=" * 20)
        print("All scripts completed successfully!")
        print("=" * 20)

