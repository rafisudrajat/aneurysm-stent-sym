@echo off
set "dir_path=%~1"
python constructAneuGeom.py ^
--experiment_dir="%dir_path%"