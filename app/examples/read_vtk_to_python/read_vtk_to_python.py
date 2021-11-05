# -*- coding: utf-8 -*-
"""
This script reads a vtk file into Python and plots the results usin
matplotlib.

@author: Lucas Myers
"""

import vtk

filename = "system-2d-plus-half.vtu"
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(filename)
reader.Update()  # Needed because of GetScalarRange
output = reader.GetOutput()