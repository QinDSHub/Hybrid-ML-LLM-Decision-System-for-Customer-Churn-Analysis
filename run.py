#!/usr/bin/env python
# coding: utf-8

import subprocess
import os

if __name__ == '__main__':
    subprocess.run(['python', 'data_preprocess.py', '--data_path', '../clean_data', '--save_path', './'])
    subprocess.run(['python', 'predict_metrics.py', '--save_path', './'])
    print("All done!")

if __name__ == '__main__':
    data_path = '../clean_data'
    save_path = './'
    
    # run data_process.py
    print("=" * 20)
    print("Running data_preprocess.py...")
    print("=" * 20)
    result1 = subprocess.run([
        'python', 'data_preprocess.py',
        '--data_path', data_path,
        '--save_path', save_path
    ])
    
    if result1.returncode != 0:
        print("data_preproces.py failed! Stopping...")
        exit(1)
    
    print("\n" + "=" * 20)
    print("Running predict_metrics.py...")
    print("=" * 20)
    result2 = subprocess.run([
        'python', 'predict_metrics.py',
        '--save_path', save_path
    ])
    
    if result2.returncode != 0:
        print("predict_metrics.py failed! Stopping...")
        exit(1)
    else:
        print("\n" + "=" * 20)
        print("All scripts completed successfully!")
        print("=" * 20)

