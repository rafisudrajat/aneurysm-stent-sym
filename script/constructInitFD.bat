@echo off
set "dir_path=%~1"
set "stent_pos=%~2"
python constructInitFD.py ^
--experiment_dir="%dir_path%" --stent_pos="%stent_pos%"