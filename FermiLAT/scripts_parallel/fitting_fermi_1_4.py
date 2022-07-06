from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
import matplotlib.pyplot as plt
import numpy as np
from regions import CircleSkyRegion
import yaml

from gammapy.catalog import CATALOG_REGISTRY
from gammapy.data import EventList
from gammapy.datasets import MapDataset
from gammapy.irf import PSFMap, EDispKernelMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
    TemplateSpatialModel,
    PowerLawNormSpectralModel,
    Models,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.modeling import Fit

##################################
# parameters I can change

sig_threshold = 50 # the significance value threshold in which I want to include sources
dataset_idx = 1
n_bin_met = 4

###################################
# general definitions

dataset_names = ['low_energy', 'med_energy', 'hi_energy']
main_path = '/home/woody/caph/mppi062h/woody_output/final/fermilat/'
iso_list = ['iso_P8R3_ULTRACLEANVETO_V3_v1.txt', 'iso_P8R3_ULTRACLEANVETO_V3_v1.txt', 'iso_P8R3_SOURCE_V3_v1.txt']

N_bin_met = 8
radius_list = [12, 5, 5]
dataset_e_edges = [100, 600, 4e3, 1e6] #u.MeV

def initializing(dataset_idx_master, n_bin_met_master):
    #####################################
    # loading datasets

    datasets_binmet = []
    for dataset_idx, dataset_name in enumerate(dataset_names):
        datasets_binmet.append([])
        if dataset_idx != 0:
            for n_bin_met in range(N_bin_met-1):
                folder = f'{main_path}/{dataset_name}/{n_bin_met}_bin_met'
                dataset = MapDataset.read(f'{folder}/fermi_dataset.fits')

                dataset.mask_safe = Map.from_geom(geom=dataset.counts.geom, data=np.ones_like(dataset.counts.data).astype(bool))   
                dataset.mask_safe &= dataset.counts.geom.region_mask(f"galactic;circle(0, 0, {radius_list[dataset_idx]})")

                m1 = (dataset.counts.geom.axes['energy'].center.value > dataset_e_edges[dataset_idx])
                m2 = (dataset.counts.geom.axes['energy'].center.value < dataset_e_edges[dataset_idx+1])
                mask = m1 & m2
                dataset.mask_safe.data[~mask] = 0

                datasets_binmet[dataset_idx].append(dataset)

    ###########################
    # models

    geom = datasets_binmet[dataset_idx][n_bin_met].counts.geom
    fgl = CATALOG_REGISTRY.get_cls("4fgl")()
    inside_geom = (geom.drop('energy')).contains(fgl.positions)
    idx = np.where(inside_geom)[0]

    fermi_models = []
    for i in idx:
        sig = float(fgl[i].info('more').split()[11])
        mo = fgl[i].sky_model()
        if (sig > sig_threshold):
            if mo.name != '4FGL J1745.6-2859':
                mo.parameters.freeze_all()
                mo.parameters['amplitude'].frozen=False
            fermi_models.append(mo)
            print(i)
    initial_model = fermi_models

    filename='/home/vault/caph/mppi062h/repositories/GC/FermiLAT/gll_iem_v07.fits'
    diffuse_galactic_fermi = Map.read(filename)
    diffuse_galactic_fermi.unit = "cm-2 s-1 MeV-1 sr-1"
    template_diffuse = TemplateSpatialModel(diffuse_galactic_fermi, normalize=False, filename=filename)

    diffuse_iem = SkyModel(
        spectral_model=PowerLawNormSpectralModel(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )
    initial_model.append(diffuse_iem)

    for dataset_idx, dataset_name in enumerate(dataset_names):
        initial_model_copy = initial_model.copy()

        filename = f'{main_path}/{dataset_name}/{iso_list[dataset_idx]}'
        diffuse_iso = create_fermi_isotropic_diffuse_model(
        filename=filename, interp_kwargs={"fill_value": None})
        initial_model_copy.append(diffuse_iso)
        if dataset_idx != 0:
            for n_bin_met in range(N_bin_met-1):
                datasets_binmet[dataset_idx][n_bin_met].models = initial_model_copy       
    ################################################
    
    return datasets_binmet[dataset_idx_master][n_bin_met_master]

dataset_fermi = initializing(dataset_idx, n_bin_met)
result0= Fit().run(datasets=[dataset_fermi])
print(result0)

print(dataset_fermi.models)

with open(f'{main_path}/{dataset_names[dataset_idx]}/{n_bin_met}_bin_met/fitted_model.yml', 'w') as outfile:
        yaml.dump(dataset_fermi.models.to_dict(), outfile, default_flow_style=False)
            
copy_dataset = dataset_fermi.copy()
copy_dataset.background = dataset_fermi.npred().copy()

sources_to_exclude = ['4FGL J1745.6-2859', 'diffuse-iem', 'fermi-diffuse-iso']
for source in sources_to_exclude:
    copy_dataset.background.data -= dataset_fermi.npred_signal(source)

# save for now in this folder
dataset_fermi.write(f'{main_path}/{dataset_names[dataset_idx]}/{n_bin_met}_bin_met/fitted_dataset.fits', overwrite = True)

from gammapy.estimators import FluxPoints, FluxPointsEstimator
fp_central = FluxPointsEstimator(
    energy_edges=dataset_fermi.counts.geom.axes['energy'].edges,
    source="4FGL J1745.6-2859",
).run([dataset_fermi])

name = f'{main_path}/{dataset_names[dataset_idx]}/{n_bin_met}_bin_met/fp.fits'
fp_central.write(name, overwrite=True)