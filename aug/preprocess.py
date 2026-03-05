from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Structure
from functools import partial
from tqdm import tqdm
import numpy as np
import os

calculator = xrd.XRDCalculator()

def calc_std_dev(two_theta, tau):
    """
    calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation

    K = 0.9 ## shape factor
    wavelength = calculator.wavelength * 0.1 ## angstrom to nm
    theta = np.radians(two_theta/2.) ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2

def compute_all_spectrum(ref_dir,save_dir,min_angle=10.0,max_angle=80.0):
    all_struc = []

    for fname in sorted(os.listdir(ref_dir)):
        fpath = '%s/%s' % (ref_dir, fname)
        struc = Structure.from_file(fpath)
        all_struc.append((struc))

    func = partial(generate_single_ref_spectra,max_angle=max_angle,min_angle=min_angle)
    # Iterate through all reference structures
    ref_patterns = []
    with tqdm(total=len(all_struc)) as pbar:
        for struc in all_struc:
            ref_patterns.append(func(struc))
            pbar.update(1)

    np.save(os.path.join(save_dir,'ref_patterns.npy'), ref_patterns)

    return 

def compute_all_pdf(ref_dir,save_dir,min_angle=10.0,max_angle=80.0):
    all_struc = []

    for fname in sorted(os.listdir(ref_dir)):
        fpath = '%s/%s' % (ref_dir, fname)
        struc = Structure.from_file(fpath)
        all_struc.append((struc))

    func = partial(generate_single_ref_spectra,max_angle=max_angle,min_angle=min_angle)
    # Iterate through all reference structures
    ref_patterns = []
    with tqdm(total=len(all_struc)) as pbar:
        for struc in all_struc:
            ref_patterns.append(func(struc))
            pbar.update(1)

    np.save(os.path.join(save_dir,'ref_patterns.npy'), ref_patterns)

    return 

def generate_single_ref_spectra(struc,min_angle,max_angle):
    pattern = calculator.get_pattern(struc, two_theta_range=(min_angle, max_angle))
    angles = pattern.x
    intensities = pattern.y

    steps = np.linspace(min_angle, max_angle, 4501)
    signals = np.zeros([len(angles), steps.shape[0]])

    for i, ang in enumerate(angles):
        # Map angle to closest datapoint step
        idx = np.argmin(np.abs(ang-steps))
        signals[i,idx] = intensities[i]

    # Convolute every row with unique kernel
    # Iterate over rows; not vectorizable, changing kernel for every row
    domain_size = 25.0
    step_size = (max_angle - min_angle)/4501
    for i in range(signals.shape[0]):
        row = signals[i,:]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size)
        # Gaussian kernel expects step size 1 -> adapt std_dev
        signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size, mode='constant')

    # Combine signals
    signal = np.sum(signals, axis=0)

    # Normalize signal
    norm_signal = 100 * signal / max(signal)
    return norm_signal

def generate_single_pdf(struc,min_angle,max_angle):
    pattern = calculator.get_pattern(struc, two_theta_range=(min_angle, max_angle))
    angles = pattern.x
    intensities = pattern.y
    return [angles,intensities]

def get_all_struct(ref_dir):
    all_struc = []
    for fname in os.listdir(ref_dir):
        fpath = '%s/%s' % (ref_dir, fname)
        struc = Structure.from_file(fpath)
        all_struc.append((struc))
    return all_struc

def compute_ref_spectrum(ref_fn,ref_dir,save_dir,all_struc,all_spec):
    struct = Structure.from_file(os.path.join(ref_dir,ref_fn))
    ref_lattice = struct.lattice.abc
    ref_spec = []
    for i,s in enumerate(all_struc):
        if False in np.isclose(ref_lattice, s.lattice.abc, atol=0.01):
            ref_spec.append(all_spec[i])
    np.save(os.path.join(save_dir,ref_fn.replace('cif','npy')),ref_spec)
    return 

def compute_ref_pdf(ref_fn,ref_dir,save_dir,all_struc,all_spec):
    struct = Structure.from_file(os.path.join(ref_dir,ref_fn))
    ref_lattice = struct.lattice.abc
    ref_spec = []
    for i,s in enumerate(all_struc):
        if False in np.isclose(ref_lattice, s.lattice.abc, atol=0.01):
            ref_spec.append(all_spec[i])
    np.save(os.path.join(save_dir,ref_fn.replace('cif','npy')),ref_spec)
    return 

def main(ref_dir):
    cif_dir = os.path.join(ref_dir,'cif')
    filenames = os.listdir(cif_dir)
    save_dir = cif_dir+'_spec'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # compute_all_spectrum(ref_dir,save_dir)
    all_spec = np.load(os.path.join(ref_dir,'ideal_patterns.npy'))
    all_struc = get_all_struct(cif_dir)
    with tqdm(total=len(filenames)) as pbar:
        for fn in filenames:
            compute_ref_spectrum(fn,cif_dir,save_dir,all_struc,all_spec)
            pbar.update(1)
    return