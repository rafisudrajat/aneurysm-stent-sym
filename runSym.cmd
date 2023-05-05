@echo off
set "dir_param=experiment\experiment 0"
set "stent_pos1=outer"
set "stent_pos2=inner"
call script\constructAneuGeom.bat "%dir_param%"
call script\constructInitFD.bat "%dir_param%" "%stent_pos1%"
call script\deployStent.bat "%dir_param%" "%stent_pos1%"
call script\mergeMesh.bat "%dir_param%"
call script\constructInitFD.bat "%dir_param%" "%stent_pos2%"
call script\deployStent.bat "%dir_param%" "%stent_pos2%"