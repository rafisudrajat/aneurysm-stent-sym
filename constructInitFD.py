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
        data=json.load(setting)["constructInitFD"][kindFD]
        pattern=data["pattern"]
        stent=data["stent"]
        deploy_pos_param=data["deploy_position_param"]
        filter_param=data.get("filter",False)
        return kindFD,pattern,stent,deploy_pos_param,filter_param

def selectPattern(name:str,param:dict={}):
    '''
    select flow diverter pattern 
    '''
    if name not in ["helical","semienterprise","enterprise","honeycomb"]:
        raise ValueError("pattern name must be either [helical, semienterprise, enterprise, honeycomb]")
    if(name=="helical"):
        if "size" in param.keys(): 
            return ps.helical(param["size"])
        return ps.helical()
    elif(name=="semienterprise"):
        return ps.semienterprise
    elif(name=="enterprise"):
        if "N" in param.keys():
            return ps.enterprise(param["N"])
        return ps.enterprise()
    elif(name=="honeycomb"):
        return ps.honeycomb()

def saveFDCase(filename:str,case:ps.VirtualStenting):
    '''
    save case object to a file
    '''
    with open(filename, 'wb') as config_object_file:
        pickle.dump(case, config_object_file)
    

def main(dir_path:str,stent_pos:str):
    t0 = time.time()
    kind_FD,pattern_param,stent_param,deploy_pos_param,filter_param=jsonParser(dir_path,stent_pos)
    pattern = selectPattern(pattern_param["name"],pattern_param.get("parameter",{}))
    stent = ps.FlowDiverter(
                pattern, 
                radius=stent_param["radius"], 
                height=stent_param["height"], 
                tcopy=stent_param["tcopy"], 
                hcopy=stent_param["hcopy"], 
                strut_radius=stent_param["strut_radius"],
                offset_angle=stent_param.get("offset_angle",0)
            )
    # for centerline, the file to be loaded must be in .vtk format
    experiment_number=dir_path.split('\\')[1].split()[1]
    centerline_load=pv.read("{}/results/centerline_EX{}.vtk".format(dir_path,experiment_number))
    path_to_bound="{}/results/vessel_EX{}.stl".format(dir_path,experiment_number) if kind_FD=="outer"\
        else "{}/results/stented1x_vessel_EX{}.stl".format(dir_path,experiment_number)
    bound=pv.read(path_to_bound)
    if(filter_param):
        bound.subdivide(filter_param['nsub'], subfilter=filter_param['kind'])
    centerline = ps.VascCenterline(
                    centerline_load.points, 
                    init_range=deploy_pos_param.get("range",np.array([])), 
                    point_spacing=deploy_pos_param.get("point_spacing",5), 
                    reverse=deploy_pos_param.get("reverse",False)
                )
    case = ps.VirtualStenting(stent = stent, centerline = centerline, boundary = bound)
    # save initial stent before deployed
    filename='{}/results/init_{}_stentEX{}'.format(dir_path,kind_FD,experiment_number)
    case.initial_stent.save("{}.vtp".format(filename))
    # Save case object
    saveFDCase('{}.obj'.format(filename),case)
    tend=time.time()
    print("Finished to construct initial stent with time= %.2f ms"%(tend-t0))

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