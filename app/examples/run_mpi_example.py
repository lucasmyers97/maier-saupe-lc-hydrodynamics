import subprocess

def main():

    subprocess.run(['/usr/bin/mpirun', '-np', '6', 
                    './install/bin/NematicSystemMPISim',
                    'parameter-files/two_defect.prm'])

if __name__ == "__main__":

    main()
