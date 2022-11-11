import paraview.simple as ps

import argparse
import time

def get_commandline_args():

    description = ("Testing pvbatch by warping a configuration by a scalar")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_file', dest='data_file',
                        help='which data file is being read in')
    args = parser.parse_args()

    return args.data_file

def main():

    filename = get_commandline_args()
    reader = ps.OpenDataFile(filename)
    warped_by_scalar = ps.WarpByScalar(Input=reader, Scalars=['POINTS', 'S'])
    ps.Show(warped_by_scalar)
    ps.Render()

if __name__ == "__main__":
    main()
