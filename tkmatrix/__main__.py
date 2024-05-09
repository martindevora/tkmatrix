import os
import sys
import pickle
import yaml
from argparse import ArgumentParser
from os import path
from lcbuilder.star.starinfo import StarInfo
import importlib.util
from tkmatrix.tkmatrix_class import MATRIX
import datetime
from pathlib import Path


def load_module(module_path):
    spec = importlib.util.spec_from_file_location("customs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def extract_custom_class(module_path):
    class_module = None
    if module_path is not None:
        class_module = load_module(module_path)
        class_name = Path(module_path.replace(".py", "")).name
        class_module = getattr(class_module, class_name)
        globals()[class_name] = class_module
        pickle.dumps(class_module)
        class_module = class_module()
    return class_module

# Check the variable CPUS from user-properties.
# If the value is greater than the real cpus number, we replace it.
def get_cpus():
    user_properties_cpus = matrix_user_properties["CPUS"]
    cpu_count = os.cpu_count()
    if user_properties_cpus > cpu_count:
        final_cpus = cpu_count
        print("Property change [CPUS]: You only have " + str(cpu_count) + " CPUs.")
    else:
        final_cpus = matrix_user_properties["CPUS"]

    return final_cpus

def get_star_info(properties, id):
    input_star_info = None
    if properties["STAR"] is not None:
        if id in properties["STAR"] and properties["STAR"][id] is not None: #TODO to support old STAR -> TIC XXXX -> properties parameters
            print("Setting star parameters under STAR -> TIX XXXX is deprecated. Please nest all the properties directly under the 'STAR' key.")
            star_properties = properties["STAR"][id]
        else:
            star_properties = properties['STAR']
        input_star_info = StarInfo(id,
                            tuple(star_properties["LD_COEFFICIENTS"]) if "LD_COEFFICIENTS" in star_properties else None,
                             star_properties["TEFF"] if "TEFF" in star_properties else None,
                             star_properties["LUM"] if "LUM" in star_properties else None,
                             star_properties["LOGG"] if "LOGG" in star_properties else None,
                             star_properties["LOGG_ERR"] if "LOGG_ERR" in star_properties else None,
                             star_properties["RADIUS"] if "RADIUS" in star_properties else None,
                             star_properties["RADIUS_LOWER_ERROR"] if "RADIUS_LOWER_ERROR" in star_properties else None,
                             star_properties["RADIUS_UPPER_ERROR"] if "RADIUS_UPPER_ERROR" in star_properties else None,
                             star_properties["MASS"] if "MASS" in star_properties else None,
                             star_properties["MASS_LOWER_ERROR"] if "MASS_LOWER_ERROR" in star_properties else None,
                             star_properties["MASS_UPPER_ERROR"] if "MASS_UPPER_ERROR" in star_properties else None,
                             star_properties["RA"] if "RA" in star_properties else None,
                             star_properties["DEC"] if "DEC" in star_properties else None)
    return input_star_info


if __name__ == '__main__':
    # We save the start time:
    start_time = datetime.datetime.now()
    current_path = os.path.dirname(os.path.realpath(__file__))
    ap = ArgumentParser(description='Sherlock Inject&Recovery tool')
    ap.add_argument('--dir', default="./", help="Working directory (if empty your current dir will be assumed)",
                    required=False)
    ap.add_argument('--properties', help="Configuration file", required=True)
    ap.add_argument('--preserve', help="Preserve the inject file. By default they should be removed and only kept if the flag is enabled.",
                    action="store_true", required=False)
    args = ap.parse_args()
    resources_dir = path.join(path.dirname(__file__))
    file_dir = resources_dir + "/" + 'properties.yaml' if resources_dir != "" and resources_dir is not None \
        else 'properties.yaml'
    print("The resource dir is: " + str(resources_dir))
    matrix_user_properties = yaml.load(open(file_dir), yaml.SafeLoader)
    user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
    preserve = args.preserve
    matrix_user_properties.update(user_properties)
    matrix_user_properties["CPUS"] = get_cpus()
    target = matrix_user_properties["TARGET"]
    file = matrix_user_properties["FILE"]
    author = matrix_user_properties["AUTHOR"]
    star_info = get_star_info(matrix_user_properties, target)
    custom_search = extract_custom_class(matrix_user_properties["CUSTOM_SEARCH_ALGORITHM"])
    prepare_algorithm = extract_custom_class(matrix_user_properties["PREPARE_ALGORITHM"])
    initial_mask = matrix_user_properties["INITIAL_MASK"]
    initial_smooth_enabled = matrix_user_properties["INITIAL_SMOOTH_ENABLED"]
    initial_transit_mask = matrix_user_properties["INITIAL_TRANSIT_MASK"]
    rv = matrix_user_properties["RV"]
    auto_detrend_period = matrix_user_properties["AUTO_DETREND_PERIOD"]
    auto_detrend_ratio = matrix_user_properties["AUTO_DETREND_RATIO"]
    auto_detrend_method = matrix_user_properties["AUTO_DETREND_METHOD"]
    auto_detrend_enabled = matrix_user_properties["AUTO_DETREND_ENABLED"]
    oscillation_reduction = matrix_user_properties["SIMPLE_OSCILLATIONS_REDUCTION"]
    oscillation_min_snr = matrix_user_properties["OSCILLATIONS_MIN_SNR"]
    oscillation_amplitude_threshold = matrix_user_properties["OSCILLATIONS_AMPLITUDE_THRESHOLD"]
    oscillation_ws_percent = matrix_user_properties["OSCILLATIONS_WS_PERCENT"]
    oscillation_min_period = matrix_user_properties["OSCILLATIONS_MIN_PERIOD"]
    oscillation_max_period = matrix_user_properties["OSCILLATIONS_MAX_PERIOD"]
    high_rms_bin_hours = matrix_user_properties["INITIAL_HIGH_RMS_BIN_HOURS"]
    high_rms_threshold = matrix_user_properties["INITIAL_HIGH_RMS_THRESHOLD"]
    high_rms_enabled = matrix_user_properties["INITIAL_HIGH_RMS_MASK"]
    outliers_sigma = matrix_user_properties["OUTLIERS_SIGMA"]
    exptime = matrix_user_properties["EXPOSURE_TIME"]
    eleanor_corr_flux = matrix_user_properties["ELEANOR_CORRECTED_FLUX"]
    cache_dir = matrix_user_properties["CACHE_DIR"]
    search_engine = matrix_user_properties["SEARCH_ENGINE"]
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + "/"
    ir = MATRIX(target, matrix_user_properties["SECTORS"], author, args.dir, args.preserve, star_info, file, exptime,
                initial_mask, initial_transit_mask, eleanor_corr_flux, outliers_sigma, high_rms_enabled,
                high_rms_threshold, high_rms_bin_hours, initial_smooth_enabled, auto_detrend_enabled,
                auto_detrend_method, auto_detrend_ratio, auto_detrend_period, prepare_algorithm, cache_dir,
                oscillation_reduction, oscillation_min_snr, oscillation_amplitude_threshold, oscillation_ws_percent,
                oscillation_min_period, oscillation_max_period, matrix_user_properties["CPUS"],
                search_engine=search_engine)
    if matrix_user_properties['PHASES'] is None and ((rv is None) or (rv is not None and rv['FILE'] is None)):
        print("You either need to define a photometry or a RV injection and recovery scenario.")
        exit(1)
    inject_dir = None
    if rv is not None and rv['FILE'] is not None:
        inject_dir = ir.recovery_rv_periods(rv['FILE'], rv["MAX_PERIOD_SEARCH"],
                                            rv['INITIAL_MASK'] if 'INITIAL_MASK' in rv else None,
                                            rv['OVERSAMPLING'] if 'OVERSAMPLING' in rv else 1,
                                            matrix_user_properties["CPUS"])
        if rv['PHASES'] is not None:
            inject_dir, period_grid, mass_grid = ir.inject_rv(inject_dir, rv['FILE'], rv["PHASES"], rv['MIN_PERIOD'],
                         rv["MAX_PERIOD"], rv["STEPS_PERIOD"], rv["MAX_MASS"], rv["STEPS_MASS"],
                         period_grid=rv['PERIOD_GRID'] if 'PERIOD_GRID' in rv else None,
                         mass_grid=rv['MASS_GRID'] if 'MASS_GRID' in rv else None,
                         period_grid_geom=rv["PERIOD_GRID_GEOM"],
                         mass_grid_geom=rv["MASS_GRID_GEOM"])
            ir.recovery_rv(inject_dir, rv['INITIAL_MASK'] if 'INITIAL_MASK' in rv else None,
                           rv['SNR_THRESHOLD'], rv['RUN_LIMIT'],
                           rv['MAX_PERIOD_SEARCH'], rv['OVERSAMPLING'] if 'OVERSAMPLING' in rv else 1)
            ir.plot_results(target, inject_dir, is_rv=True)
    inject_dir, period_grid, radius_grid = ir.inject(matrix_user_properties["PHASES"],
                           matrix_user_properties["MIN_PERIOD"], matrix_user_properties["MAX_PERIOD"],
                           matrix_user_properties["STEPS_PERIOD"],
                           matrix_user_properties["MIN_RADIUS"], matrix_user_properties["MAX_RADIUS"],
                           matrix_user_properties["STEPS_RADIUS"],
                           period_grid=matrix_user_properties['PERIOD_GRID'],
                           radius_grid=matrix_user_properties['RADIUS_GRID'],
                           period_grid_geom=matrix_user_properties["PERIOD_GRID_GEOM"],
                           radius_grid_geom=matrix_user_properties["RADIUS_GRID_GEOM"],
                           inject_dir=inject_dir)
    ir.recovery(inject_dir, matrix_user_properties["SNR_THRESHOLD"],
                matrix_user_properties["DETREND_METHOD"],
                matrix_user_properties["DETREND_WS"], matrix_user_properties["FIT_METHOD"],
                matrix_user_properties["RUN_LIMIT"],
                custom_search, matrix_user_properties["MAX_PERIOD_SEARCH"], matrix_user_properties["OVERSAMPLING"],
                matrix_user_properties["SIGNAL_SELECTION_MODE"],
                use_search_cache=matrix_user_properties["USE_SEARCH_CACHE"])
    ir.plot_results(target, inject_dir)
    print("Execution time: " + str(datetime.datetime.now() - start_time))
