# coding: utf-8

import numpy as np
import scipy.ndimage
import skimage.measure
import pydicom
import stl.mesh
import os, glob, argparse

def parse_dicom(dir_path):
	slices = [pydicom.dcmread(path)
				for path in glob.glob(os.path.join(dir_path, '*'))
				if os.path.splitext(path)[1] in ['.dcm','.DCM']
				]
	slices.sort(key = lambda s: int(s.InstanceNumber))

	if '0x70051022' in slices[0]:
		zspace = slices[0]['0x70051022'].value
	else:
		zspace = slices[0].SliceThickness
	spacing = np.array(list(map(float, list(slices[0].PixelSpacing)+[zspace])))
	return slices,spacing

def to_HU_scale(voxels, slope=1.0, intercept=0.0):
	voxels = voxels.astype(np.float64)
	voxels *= slope
	voxels += intercept
	voxels = voxels.astype(np.int16)
	return voxels

def filter_voxels(voxels, vmax=1000, vmin=80):
	voxels[voxels<vmin] = -1000
	# voxels[voxels>vmax] = -1000
	return voxels

def resample(voxels, current_spacing, new_spacing=[1,1,1]):
	resize_factor = current_spacing / new_spacing
	new_real_shape = voxels.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / voxels.shape
	new_spacing = current_spacing / real_resize_factor
	
	voxels = scipy.ndimage.interpolation.zoom(voxels, real_resize_factor)
	
	return voxels, new_spacing

def reset_orientation(voxels, slices, sample_idx=None):
	"""
	Reset the orientation to the LPH format.
	That is:
	X = Right -> Left
	Y = Anterior -> Posterior
	Z = inferior -> superior
	"""
	if hasattr(slices[0], 'PatientOrientation'):
		print(sample_idx,slices[0].PatientOrientation)
		if slices[0].PatientOrientation==['R','P']:
			voxels = np.rot90(voxels, k=2, axes=(1,2))
		elif slices[0].PatientOrientation!=['L','P']:
			print(slices[0].PatientOrientation)
	else:
		print('slices.ImageOrientationPatient', slices[0].ImageOrientationPatient)
		voxels = np.rot90(np.rot90(voxels, k=3, axes=(0,2)), k=2, axes=(0,1))
	return voxels


def voxel2mesh(voxels):
	verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(voxels, 0)
	return verts, faces, normals, values

def save_mesh(mesh, faces, save_path):
	obj = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(3):
			obj.vectors[i][j] = verts[f[j],:]
	if os.path.splitext(os.path.basename(save_path))[0] in ['PRI_292','PRI_1871','PRI_2298']:
		center = (obj.max_ + obj.min_) * 0.5
		obj.rotate([0.0, 1.0, 0.0], np.deg2rad(180+30), center)
	obj.save(save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('dicom_dir', type=str, help='Path to the directory containing dicom slices.')
	parser.add_argument('save_path', type=str, help='Path to the (directory of) npy file(s) where 3D verts are saved.')
	parser.add_argument('--is_root', action='store_true', help='If selected, analyze all the subdirectories under dicom_dir.')
	parser.add_argument('--interpolated_spacing', type=float, nargs=3, default=None, help='Voxel spacing in mm after the interpolation.')
	args = parser.parse_args()

	if args.is_root:
		data_dirs = glob.glob(os.path.join(args.dicom_dir, '*/'), recursive=False)
		save_dir = args.save_path
		save_paths = [os.path.join(save_dir, os.path.relpath(d, args.dicom_dir)).rstrip('/')+'.stl'
						for d in data_dirs]
	else:
		data_dirs = [args.dicom_dir]
		save_paths = [args.save_path]
		save_dir = os.path.dirname(args.save_path)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	for data_path,save_path in zip(data_dirs, save_paths):
		slices,spacing = parse_dicom(data_path)
		voxels = np.stack([s.pixel_array for s in slices], axis=-1)
		voxels = to_HU_scale(voxels, slope=float(slices[0].RescaleSlope), intercept=float(slices[0].RescaleIntercept))
		voxels,new_spacing = resample(voxels, spacing)
		sample_idx = os.path.basename(data_path.rstrip('/'))
		voxels = reset_orientation(voxels, slices, sample_idx=sample_idx)
		if not args.interpolated_spacing is None:
			voxels,new_spacing = resample(voxels, new_spacing, args.interpolated_spacing)
		voxels = filter_voxels(voxels)
		verts, faces, normals, values = voxel2mesh(voxels)
		save_mesh(verts, faces, save_path)