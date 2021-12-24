# coding: utf-8

import pyvista
from PIL import Image

def load_stl(path):
	return pyvista.PolyData(path)

def project2D(mesh, elev, azim, **kwargs):
	mesh = mesh.copy()
	mesh.rotate_z(-90-azim)
	mesh.rotate_x(-90+elev)
	plotter = pyvista.Plotter(off_screen=True)
	plotter.set_background('black')
	plotter.add_mesh(mesh, color="white")
	img = plotter.screenshot(return_img=True)
	img = Image.fromarray(img, 'RGB')
	return img