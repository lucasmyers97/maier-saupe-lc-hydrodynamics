"""
This script generates a bunch of parameter files based on the `template.toml` file
"""
import os

def make_filename(TER, L2):

    L2_string_raw = '{:.2f}'.format(L2).split('.')
    L2_string = '{}_{}'.format( L2_string_raw[0], L2_string_raw[1] )

    if TER:
        return 'TER_L2_{}_new'.format(L2_string)
    else:
        return 'ER_L2_{}_new'.format(L2_string)



def main():

    with open('template.toml', 'r') as f:
        contents = f.read()

    L2_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    TER_angles = [0.0, 1.0471975512]
    TER_configs = [False, True]
    TER_names = ['ER', 'TER']

    directory_name = '/expanse/lustre/scratch/myers716/temp_project/2024-04-22/{}/{}/'
    file_name = '{}.toml'

    for L2 in L2_list:
        for TER_angle, TER_config, TER_name in zip(TER_angles, TER_configs, TER_names):

            L2_name = make_filename(TER_config, L2)
            cur_directory_name = directory_name.format(TER_name, L2_name)
            cur_contents = contents.format(directory_name=cur_directory_name,
                                           L2=L2,
                                           final_twist_angle=TER_angle)
            os.makedirs(cur_directory_name, exist_ok=True)

            with open(file_name.format(L2_name), 'w') as f:
                f.write( cur_contents )



if __name__ == '__main__':
    main()
