import numpy as np
from Utils import *
import time
import argparse
import json

def jsonParser(dir_path):
    with open(dir_path+'/appSettings.json','r') as setting:
        data=json.load(setting)["constructAneuGeom"]
        aneu_geom_param=data["aneu_geom_param"]
        filter_param=data.get("filter",False)
        return aneu_geom_param, filter_param

def main(dir_path:str):
    t0 = time.time()
    aneu_param,filter_param=jsonParser(dir_path)
    dict_aneu_geom = aneu_geom(
        r=aneu_param['r'], 
        h=aneu_param['h'], 
        hstent=aneu_param['hstent'], 
        overlap=aneu_param['overlap'], 
        aneu_rad=aneu_param['aneu_rad'], 
        cyl_res = aneu_param['cyl_res'], 
        sph_res=aneu_param['sph_res'], 
        angle=np.radians(aneu_param['angle']), 
        extension_ratio=aneu_param['extension_ratio'], 
        ext_res=aneu_param['ext_res'],
        get_inlet_outlet=aneu_param.get("get_inlet_outlet",False)
    )
    aneu=dict_aneu_geom["geom"]
    centerline_points=dict_aneu_geom["stent_centerline"]
    inlet_surface=dict_aneu_geom.get("inlet",None)
    outlet_surface=dict_aneu_geom.get("outlet",None)
    experiment_number=dir_path.split('\\')[1].split()[1]
    if inlet_surface!=None and outlet_surface!=None:
        inlet_surface.save("{}/results/inlet_EX{}.stl".format(dir_path,experiment_number))
        outlet_surface.save("{}/results/outlet_EX{}.stl".format(dir_path,experiment_number))

    centerline_wrap=pv.wrap(centerline_points)
    # for centerline, the file to be saved must be in .vtk format
    centerline_wrap.save("{}/results/centerline_EX{}.vtk".format(dir_path,experiment_number))
    bound=aneu
    if(filter_param):
        bound = aneu.subdivide(filter_param['nsub'], subfilter=filter_param['kind'])
    bound.save("{}/results/vessel_EX{}.stl".format(dir_path,experiment_number))
    tend=time.time()
    print("Finished to construct aneurism geometry with time= %.2f ms"%(tend-t0))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to experiment directory'
    )
    args=parser.parse_args()
    directory_path=args.experiment_dir
    main(directory_path)
