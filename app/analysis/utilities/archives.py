import os
import re

import numpy as np

def get_archive_files(folder, archive_filename_prefix):

    filenames = os.listdir(folder)

    pattern = r'^(' + archive_filename_prefix + r'(\d*))\.mesh.ar$'
    p = re.compile(pattern)

    archive_filenames = []
    times = []
    for filename in filenames:
        matches = p.findall(filename)
        if matches:
            archive_filenames.append(os.path.join(folder, matches[0][0]))
            times.append( int(matches[0][1]) )
        
    archive_filenames = np.array(archive_filenames)
    times = np.array(times)

    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    archive_filenames = archive_filenames[sorted_idx]

    return archive_filenames, times
