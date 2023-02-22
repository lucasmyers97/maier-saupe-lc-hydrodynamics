import argparse
import tarfile
import os.path

def get_commandline_args():

    description = ('Compresses all subfolders in a parent folder, and moves '
                   'them to some another specified location. '
                   'Does this one folder at a time to conserve disk space.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--parent_folder',
                        dest='parent_folder',
                        help='folder whose subfolders we are compressing')
    parser.add_argument('--destination',
                        dest='destination',
                        help='where the archives will be stored')
    parser.add_argument('--start_num',
                        dest='start_num',
                        type=int,
                        help='file at which to start in case of interrupt')

    args = parser.parse_args()

    return args.parent_folder, args.destination, args.start_num



def get_subfolders(parent_folder):

    return [folder for folder in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, folder))]



def make_tarfile(parent_folder, subfolder, destination):

    full_folder_name = os.path.join(parent_folder, subfolder)
    destination_name = os.path.join(destination, subfolder)
    tarfile_name = destination_name + '.tar.gz'
    with tarfile.open(tarfile_name, 'w:gz') as tar:
        tar.add(full_folder_name, arcname=subfolder)



def main():

    parent_folder, destination, start_num = get_commandline_args()
    subfolders = get_subfolders(parent_folder)

    for i, folder in enumerate(subfolders):
        if i < start_num:
            continue

        print('Tar-ing folder: {}, name: {}'.format(i, folder))
        make_tarfile(parent_folder, folder, destination)


if __name__ == '__main__':
    main()
