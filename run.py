"""Run from the extracted project folder: python run.py"""
import os
# Avoid excessive BLAS threads when optional orbit propagation uses small arrays.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import uvicorn
if __name__=='__main__':
    print('\nPassRescue — local planning prototype\nOpen http://127.0.0.1:8000\nPress Ctrl+C to stop.\n')
    uvicorn.run('app.main:app',host='127.0.0.1',port=8000,log_level='info')
