#!/bin/bash
# run.sh

# This tells Python to ignore all warnings globally, 
# which silences the joblib/sklearn parallel worker warnings.
export PYTHONWARNINGS="ignore"

python src/mh_pipeline.py "$@"