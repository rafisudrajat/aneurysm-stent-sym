import PyStenting as ps
from Utils import *
import time
import pickle
import argparse
import json

def jsonParser(dir_path,pos):
    with open(dir_path+'/appSettings.json','r') as setting:
        if(pos!="inner" and pos!="outer"):
            raise ValueError("stent position value must be either 'inner' or 'outer'")
        kindFD=pos
        data=json.load(setting)["deployStent"][kindFD]
        deploy_param=data["deploy_param"]
        render_param=data["render_param"]
        filter_param=data.get("filter",False)
        return kindFD,deploy_param,render_param,filter_param

def loadFDCaseFile(filename:str)->ps.VirtualStenting:
    with open(filename, 'rb') as config_object_file:
        config_object = pickle.load(config_object_file)
        case=config_object
        return case

def main(dir_path:str,stent_pos:str):
    kind_FD,deploy_param,render_param,filter_param=jsonParser(dir_path,stent_pos)
    t1 = time.time()
    experiment_number=dir_path.split('\\')[1].split()[1]
    case= loadFDCaseFile('{}/results/init_{}_stentEX{}.obj'.format(dir_path,kind_FD,experiment_number))
    result = case.deploy( 
                tol=deploy_param["tol"], 
                add_tol=deploy_param["add_tol"],
                step=deploy_param["step"], 
                fstop=deploy_param["fstop"],
                max_iter=deploy_param["max_iter"],
                alpha=deploy_param["alpha"],
                verbose=deploy_param["verbose"], 
                OC=deploy_param["OC"],
                render_gif=deploy_param.get("render_gif"),
                deployment_name=deploy_param.get("deployment_name")
            )

    t2 = time.time()
    print('Deployment simulation done, dt_deploy=%.2f s'%(t2-t1))
    if len(render_param)!=0:
        result= result.render_strut(
                n=render_param["n"],
                h=render_param["h"],
                threshold=render_param["threshold"]
            )
        '''Filtering for stent mesh will be deprecated'''
        if(filter_param):
            result.subdivide(filter_param['nsub'], subfilter=filter_param['kind'])
        result.save("{}/results/deployed_{}_stentEX{}.stl".format(dir_path,kind_FD,experiment_number))
        t3 = time.time()
        print('Rendering done, dt_render=%.2f s'%(t3-t2))
    else:
        result.save("{}/results/deployed_{}_stentEX{}_norender.vtp".format(dir_path,kind_FD,experiment_number))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to experiment directory'
    )
    parser.add_argument(
        '--stent_pos',
        type=str,
        default='inner',
        help="Position of the stent, value must be 'inner' or 'outer'"
    )
    args=parser.parse_args()
    directory_path=args.experiment_dir
    stent_pos=args.stent_pos
    main(directory_path,stent_pos)