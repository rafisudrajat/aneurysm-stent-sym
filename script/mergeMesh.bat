@echo off
set "dir_path=%~1"
python mergeMesh.py ^
--experiment_dir="%dir_path%"