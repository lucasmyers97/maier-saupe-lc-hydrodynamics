import paraview.simple as ps

import time
import argparse
import os

from utilities import paraview as pvu

def get_commandline_args():

    description = ("Prints frames of animation of nematic from vtu file")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--configuration_prefix', 
                        dest='configuration_prefix',
                        help='prefix of pvtu file holding configuration')
    parser.add_argument('--output_prefix',
                        dest='output_prefix',
                        help='output prefix (will be numbered by timestep)')
    parser.add_argument('--dolly_zoom',
                        dest='dolly_zoom',
                        type=float,
                        default=1.0,
                        help='how much to zoom the image in by')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    output_prefix = os.path.join(output_folder, args.output_prefix)

    return (args.data_folder, args.configuration_prefix, 
            output_prefix, args.dolly_zoom)



def main():

    start = time.time()

    n_steps = 5
    (data_folder, configuration_prefix, 
     output_prefix, dolly_zoom) = get_commandline_args()
    
    vtu_filenames, times = pvu.get_vtu_files(data_folder, configuration_prefix)
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(data_folder, vtu_filename) )

    Q_configuration = ps.OpenDataFile(vtu_full_path)
    tsteps = Q_configuration.TimestepValues
    print(tsteps)

    eigenvalue_filter = pvu.get_eigenvalue_programmable_filter(Q_configuration)
    
    show_view = ps.Show(eigenvalue_filter)
    ps.ColorBy(show_view, ('POINTS', 'S'))
    show_view.RescaleTransferFunctionToDataRange(True)
    
    source = ps.GetActiveSource()
    view = ps.GetActiveView()
    display = ps.GetDisplayProperties(source, view)
    display.SetScalarBarVisibility(view, True)
    
    view.ViewSize = [500, 500]
    view.ViewTime = tsteps[0]
    render_view = ps.Render()
    camera = ps.GetActiveCamera()
    camera.Dolly(dolly_zoom)
    ps.SaveScreenshot(output_prefix + str(0) + ".png", magnification=5, quality=100, view=render_view)

    for i in range(1, n_steps):
        view.ViewTime = tsteps[i]
        show_view.RescaleTransferFunctionToDataRange(True)
        render_view = ps.Render()
        ps.SaveScreenshot(output_prefix + str(i) + ".png", magnification=5, quality=100, view=render_view)

    stop = time.time()
    print(stop - start)



if __name__ == "__main__":

    main()
