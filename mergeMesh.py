import argparse
import trimesh

def main(dir_path:str):
    experiment_number=dir_path.split('\\')[1].split()[1]
    vessel=trimesh.load("{}/results/vessel_EX{}.stl".format(dir_path,experiment_number))
    deployed_stent=trimesh.load("{}/results/deployed_outer_stentEX{}.stl".format(dir_path,experiment_number))
    combined = trimesh.util.concatenate([vessel,deployed_stent])
    combined.export(file_obj="{}/results/stented1x_vessel_EX{}.stl".format(dir_path,experiment_number)) 

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


