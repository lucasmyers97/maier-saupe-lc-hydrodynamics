import subprocess
import argparse

def get_commandline_args():

    description=('Runs example mpi program')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mpi_program',
                        dest='mpi_program',
                        help='name of mpi executable to be used')
    parser.add_argument('--n_process_arg',
                        dest='n_process_arg',
                        help=('argument to pass to mpi program to specify '
                              'number of processes'))
    parser.add_argument('--n_processes',
                        dest='n_processes',
                        type=int,
                        help='number of processes to run executable on')
    parser.add_argument('--executable',
                        dest='executable',
                        help='name of executable to be executed by mpi')
    args = parser.parse_args()

    return (args.mpi_program, args.n_process_arg, 
            args.n_processes, args.executable)

def main():

    (mpi_program, n_process_arg, 
     n_processes, executable) = get_commandline_args()

    subprocess.run([mpi_program, 
                    '-{}'.format(n_process_arg), 
                    '{}'.format(n_processes), 
                    executable])

if __name__ == "__main__":

    main()
