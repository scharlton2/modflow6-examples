# ## One-Dimensional Steady Flow with Transport
#
# phreeqc ex11
#
#


# ### One-Dimensional Steady Flow with Transport Problem Setup

# Imports

# +
import shutil
import sys
from pathlib import Path

import flopy
import git
import matplotlib.pyplot as plt
import modflowapi
import numpy as np
import pooch
from flopy.plot.styles import styles
from modflow_devtools.misc import get_env, timed


import os
import sys


import modflowapi
from modflowapi import Callbacks

import phreeqcrm
import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
import shutil

sim_name = "ex-gwt-phreeqc"
try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()
data_path = root / "data" / sim_name if root else Path.cwd()

# Settings from environment variables
write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)
# -


# # Import common functionality

# import analytical
# import config
# from figspecs import USGSFigure

# Set figure properties specific to this problem

# +
figure_size = (5, 3)
# -

# Base simulation and model name and workspace

#ws = config.base_ws
ws = root
## example_name = "moc"

from contextlib import contextmanager
@contextmanager
def change_directory(new_dir):
    original_dir = os.getcwd()  # Get the current working directory

    # Convert relative path to an absolute path
    new_dir = os.path.abspath(new_dir)
    
    # Check if the directory exists; if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    try:
        os.chdir(new_dir)  # Change to the specified directory
        yield
    finally:
        os.chdir(original_dir)  # Change back to the original directory when done


# Scenario parameters - make sure there is at least one blank line before next item

parameters = {
    "ex-gwt-phreeqc": {
        "longitudinal_dispersivity": 0.002,
    },
}

# Scenario parameter units - make sure there is at least one blank line before next item
# add parameter_units to add units to the scenario parameter table

parameter_units = {
    "longitudinal_dispersivity": "$m$",
}

# Model units

length_units = "METERS"    # m
time_units = "seconds"     # s

# Table of model parameters

nper = 1                      # Number of periods
nlay = 1                      # Number of layers
nrow = 1                      # Number of rows
ncol = 40                     # Number of columns
system_length = .08           # Length of system ($m$)
delr = 0.002                  # Column width ($m$)
delc = 1.0                    # Row width ($m$)
top = 1.0                     # Top of the model ($m$)
botm = 0                      # Layer bottom elevation ($m$)
hydraulic_conductivity = 1.0  # Hydraulic conductivity ($m s^{-1}$)
porosity = 0.2                # Porosity of mobile domain (unitless)
specific_discharge = 1./720.*porosity  # Specific discharge ($m s^{-1}$)
total_time = 72000.0          # Simulation time ($s$)
concentration_factor = 1000.  # Scale factor ($millmoles/mole$)


# Create yaml for phreeqcrm initialize()

def setup_phreeqcrm(sim_folder):
    
    ##os.chdir(sys.path[0])
    #notebook_path = os.path.abspath("ex11-bmi-mmols-single_dlp.ipynb")

    #notebook_directory = os.path.dirname(notebook_path)
    #assert(os.getcwd() == notebook_directory)

    #assert(sim_folder == "ex11-bmi-mmols-single_dlp")
    
    # get abspath before change_directory
    #sim_ws = os.path.abspath(os.path.join(ws, sim_folder, "phreeqcrm"))
    #sim_ws = os.path.abspath(os.path.join(workspace, sim_folder, "phreeqcrm"))
    sim_ws = workspace / sim_folder / "phreeqcrm"

    with change_directory(sim_ws):
        
        # copy phreeqc.dat to sim_ws
        #source_file = os.path.join(notebook_directory, 'phreeqc.dat')
        #source_file = os.path.join(data_path, 'phreeqc.dat')
        source_file = data_path / 'phreeqc.dat'
        shutil.copy(source_file, sim_ws)

        # copy advect.pqi to sim_ws
        #source_file = os.path.join(notebook_directory, 'advect.pqi')
        #source_file = os.path.join(data_path, 'advect.pqi')
        source_file = data_path / 'advect.pqi'
        shutil.copy(source_file, sim_ws)
    
        # Create YAMLPhreeqcRM document
        yrm = phreeqcrm.YAMLPhreeqcRM()

        # Number of cells
        nxyz = nlay*nrow*ncol

        # Set GridCellCount
        yrm.YAMLSetGridCellCount(nxyz)

        # Set some properties
        yrm.YAMLSetErrorHandlerMode(1)
        yrm.YAMLSetComponentH2O(False)
        yrm.YAMLSetRebalanceFraction(0.5)
        yrm.YAMLSetRebalanceByCell(True)
        yrm.YAMLUseSolutionDensityVolume(False)
        yrm.YAMLSetPartitionUZSolids(False)

        # Set concentration units
        yrm.YAMLSetUnitsSolution(2)           # 1, mg/L; 2, mol/L; 3, kg/kgs
        yrm.YAMLSetUnitsPPassemblage(1)       # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        yrm.YAMLSetUnitsExchange(1)           # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        yrm.YAMLSetUnitsSurface(1)            # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        yrm.YAMLSetUnitsGasPhase(1)           # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        yrm.YAMLSetUnitsSSassemblage(1)       # 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        yrm.YAMLSetUnitsKinetics(1)           # 0, mol/L cell; 1, mol/L water; 2 mol/L rock

        # Set conversion from seconds to user units (days) Only affects one print statement
        time_conversion = 1.0 / 86400.0
        yrm.YAMLSetTimeConversion(time_conversion)

        # Set representative volume
        rv = [1] * nxyz
        yrm.YAMLSetRepresentativeVolume(rv)

        # Set initial density
        density = [1.0] * nxyz
        yrm.YAMLSetDensityUser(density)

        # Set initial porosity
        por = [0.2] * nxyz
        yrm.YAMLSetPorosity(por)

        # Set initial saturation
        sat = [1] * nxyz
        yrm.YAMLSetSaturationUser(sat)

        # Load database
        yrm.YAMLLoadDatabase("phreeqc.dat")

        # Run file to define solutions and reactants for initial conditions, selected output
        workers = True             # Worker instances do the reaction calculations for transport
        initial_phreeqc = True     # InitialPhreeqc instance accumulates initial and boundary conditions
        utility = True             # Utility instance is available for processing
        yrm.YAMLRunFile(workers, initial_phreeqc, utility, "advect.pqi")

        # Clear contents of workers and utility
        initial_phreeqc = False
        input = "DELETE; -all"
        yrm.YAMLRunString(workers, initial_phreeqc, utility, input)
        yrm.YAMLAddOutputVars("AddOutputVars", "true")

        # Determine number of components to transport
        yrm.YAMLFindComponents()

        # initial solutions
        initial_solutions = [1] * nxyz
        yrm.YAMLInitialSolutions2Module(initial_solutions)

        # initial exchanges
        initial_exchanges = [1] * nxyz
        yrm.YAMLInitialExchanges2Module(initial_exchanges)

        # Write YAML file
        yrm.WriteYAMLDoc("ex11-advect.yaml")
        
    return "ex11-advect.yaml"


def build_phreeqcrm(sim_folder):
    print(f"sim_folder={sim_folder}")

    yaml = setup_phreeqcrm(sim_folder)
    
    rm = None
    try:
        # get abspath before change_directory
        # sim_ws = os.path.abspath(os.path.join(workspace, sim_folder, "phreeqcrm"))
        sim_ws = workspace / sim_folder / "phreeqcrm"
        print(f"sim_ws={sim_ws}")
        
        rm = phreeqcrm.BMIPhreeqcRM()
        with change_directory(sim_ws):
            rm.initialize(yaml)
            
        rm.solutes = rm.get_value_ptr("Components")
        ncomps     = rm.get_value_ptr("ComponentCount")[0]

        nxyz = nlay*nrow*ncol
        assert nxyz == rm.get_value_ptr("GridCellCount")[0]
        
        # Get initial concentrations
        rm.initial_solution = rm.get_value_ptr("Concentrations")[::nxyz]
        assert len(rm.initial_solution) == ncomps

        # Set boundary condition
        bc1 = [0]           # solution 0 from Initial IPhreeqc instance
        rm.influent_concentration = rm.InitialPhreeqc2Concentrations(bc1)

        # Convert to millimoles
        rm.initial_solution       = rm.initial_solution * concentration_factor
        rm.influent_concentration = rm.influent_concentration * concentration_factor
            
    except:
        raise RuntimeError("build_phreeqcrm failed")
    return rm

# ### Functions to build, write, run, and plot models
#
# MODFLOW 6 flopy GWF simulation object (sim) is returned
#


def build_mf6gwfgwt(sim_folder, solutes, influent_concentration, initial_solution, longitudinal_dispersivity):
    name = "flow"
    # sim_ws = os.path.join(ws, sim_folder, "mf6gwfgwt")
    sim_ws = ws / sim_folder / "mf6gwfgwt"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=sim_ws, exe_name="mf6"
    )
    tdis_ds = ((total_time, 400, 1.0),)    
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
    ims = flopy.mf6.ModflowIms(sim, filename=f"{name}.ims",
        inner_dvclose=1e-9, outer_dvclose=1e-8)
    
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    sim.register_ims_package(ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=0,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, ncol - 1), 1.0]])
    wel_spd = {
        0: [
            [
                (0, 0, 0),
                specific_discharge * delc * delr,   # m^3/s
                *influent_concentration,
            ]
        ],
    }
    
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        pname="WEL-1",
        auxiliary=solutes.tolist()
    )
    head_filerecord = f"{name}.hds"
    budget_filerecord = f"{name}.bud"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    
    # set vars for use in run_model
    sim.sim_ws = sim_ws
    
    # @todo update for other OSs
    sim.libmf6 = Path(os.path.join(os.environ["USERPROFILE"], "AppData\\Local\\flopy\\bin", "libmf6"))
    
    # build transports
    for i, solute in enumerate(solutes):
        name = f"trans.{solute}"
        gwt = flopy.mf6.ModflowGwt(sim, modelname=name, save_flows=True)
        
        imsgwt = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab", filename=f"{gwt.name}.ims",
            inner_dvclose=1e-9, outer_dvclose=1e-8)
        sim.register_ims_package(imsgwt, [gwt.name])
        
        flopy.mf6.ModflowGwtdis(
            gwt,
            length_units=length_units,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
        )
        
        flopy.mf6.ModflowGwtic(gwt, strt=initial_solution[i], filename=f"{gwt.name}.ic")
        
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=porosity,
        )
        
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            diffc=0.0,
            alh=longitudinal_dispersivity,
            ath1=longitudinal_dispersivity,
        )
        
        sourcerecarray = [["WEL-1", "AUX", solute]]
        flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)
        obs_data = {
            f"{solute}.obs.csv": [
                (f"CELL{i:02d}", "CONCENTRATION", (0, 0, i)) for i in range(40)
            ]
        }

        obs_package = flopy.mf6.ModflowUtlobs(
            gwt, digits=10, print_input=True, continuous=obs_data
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt.name}.cbc",
            concentration_filerecord=f"{gwt.name}.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
            printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        
        flopy.mf6.ModflowGwfgwt(
            sim, exgtype="GWF6-GWT6", exgmnamea=gwf.name, exgmnameb=gwt.name, filename=f"{gwf.name}-{gwt.name}.gwfgwt"
        )

    return sim

# MODFLOW 6 flopy GWF simulation object (sim) is returned


def build_model(sim_name, longitudinal_dispersivity):
    sims = None
    #if config.buildModel:
    if build_model:
        # build_phreeqcrm
        sim_phreeqcrm = build_phreeqcrm(sim_name)
        
        solutes                = sim_phreeqcrm.solutes
        initial_solution       = sim_phreeqcrm.initial_solution
        influent_concentration = sim_phreeqcrm.influent_concentration
        
        sim_mf6gwf = build_mf6gwfgwt(sim_name, solutes, influent_concentration, initial_solution, longitudinal_dispersivity)
        sims = (sim_phreeqcrm, sim_mf6gwf)
    return sims


# Function to write model files


def write_model(sims, silent=True):
    #if config.writeModel:
    if write:
        _, sim_mf6gwf = sims
        sim_mf6gwf.write_simulation(silent=silent)
    return


# Function to run the model
# True is returned if the model runs successfully


#@config.timeit
@timed
def run_model(sims, silent=True):
    
    success = True
    #if config.runModel:
    if run:
        phreeqcrm, sim_mf6gwf = sims
        concs = phreeqcrm.get_value_ptr("Concentrations")
        ip = phreeqc_mod.IPhreeqc()
        mf6api = modflowapi.ModflowApi(sim_mf6gwf.libmf6, working_directory=sim_mf6gwf.sim_ws)
        mf6api.initialize()
        
        input_var_names = mf6api.get_input_var_names()
        nxyz = nlay*nrow*ncol
        current_time = mf6api.get_current_time()
        end_time     = mf6api.get_end_time()
        trans_concs = np.full(nxyz*len(phreeqcrm.solutes), 0.0)
        while current_time < end_time:
            mf6api.update()
            for i, solute in enumerate(phreeqcrm.solutes):
                conc = mf6api.get_value_ptr(f"TRANS.{solute.upper()}/X")
                #status = phreeqcrm.SetIthConcentration(i, conc / concentration_factor)
                trans_concs[i*nxyz:(i+1)*nxyz] = conc / concentration_factor
            phreeqcrm.set_value("Concentrations", trans_concs)
            phreeqcrm.update()
            for i, solute in enumerate(phreeqcrm.solutes):
                #conc = mf6api.get_value_ptr(f"TRANS.{solute.upper()}/X") 
                #conc = phreeqcrm.GetIthConcentration(i)
                conc = concs[i*nxyz:(i+1)*nxyz] * concentration_factor
                #conc = conc * concentration_factor
                mf6api.set_value(f"TRANS.{solute.upper()}/X", conc)
            current_time = mf6api.get_current_time()
        mf6api.finalize()
        if not silent:
            print("NORMAL TERMINATION OF SIMULATION")
    return success


def make_script():
    """
    Specify script.

    Transport calculation from example 11.
    """
    script = """
TITLE Example 11.--Transport and cation exchange.
SOLUTION 0  CaCl2
        units            mmol/kgw
        temp             25.0
        pH               7.0     charge
        pe               12.5    O2(g)   -0.68
        Ca               0.6
        Cl               1.2
SOLUTION 1-40  Initial solution for column
        units            mmol/kgw
        temp             25.0
        pH               7.0     charge
        pe               12.5    O2(g)   -0.68
        Na               1.0
        K                0.2
        N(5)             1.2
END
EXCHANGE 1-40
        -equilibrate 1
        X                0.0011
END
TRANSPORT
        -cells           40
        -lengths         0.002
        -shifts          100
        -time_step       720.0
        -flow_direction  forward
        -boundary_conditions   flux  flux
        -diffusion_coefficient 0.0e-9
        -dispersivities  0.002
        -correct_disp    true
        -punch_cells     40
        -punch_frequency 1
        -print_cells     40
        -print_frequency 20
SELECTED_OUTPUT 1
        -file            ex11trn.sel
        -reset           false
        -high_precision true
USER_PUNCH 1
-headings PV Ca Cl K N Na Cl_analytical
10 PUNCH TOTAL_TIME / (40*720)
20 PUNCH TOT("Ca"), TOT("Cl"), TOT("K"), TOT("N"), TOT("Na")
# calculate Cl_analytical...
  50 x = (STEP_NO + 0.5) / cell_no
  60 DATA 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
  70 READ a1, a2, a3, a4, a5, a6
  80 Peclet = 0.08 / 0.002
  90  z = (1 - x) / SQRT(4 * x / Peclet)
  100 PA = 0
  110 GOSUB 2000 # calculate e_erfc = exp(PA) * erfc(z)
  120 e_erfc1 = e_erfc
  130 z = (1 + x) / SQRT(4 * x / Peclet)
  140 PA = Peclet
  150 GOSUB 2000 # calculate exp(PA) * erfc(z)
  160 y = 0.6 * (e_erfc1 + e_erfc)
  #170 PLOT_XY x, y, line_width = 0, symbol = Circle, color = Red
  170 PUNCH y
  180 d = (y - TOT("Cl")*1000)^2
  190 IF EXISTS(10) THEN PUT(d + GET(10), 10) ELSE PUT(d, 10)
  200 IF STEP_NO = 2 * CELL_NO THEN print 'SSQD for Cl after 2 Pore Volumes: ', GET(10), '(mmol/L)^2'
  210 END
  2000 REM calculate e_erfc = exp(PA) * erfc(z)...
  2010 sgz = SGN(z)
  2020 z = ABS(z)
  2050 b = 1 / (1 + a6 * z)
  2060 e_erfc = b * (a1 + b * (a2 + b * (a3 + b * (a4 + b * a5)))) * EXP(PA - (z * z))
  2070 IF sgz = -1 THEN e_erfc = 2 * EXP(PA) - e_erfc
  2080 RETURN
END
        """
    return script


def get_selected_output(phreeqc):
    """Return calculation result as dict.

    Header entries are the keys and the columns
    are the values as lists of numbers.
    """
    output = phreeqc.get_selected_output_array()
    header = output[0]
    conc = {}
    for head in header:
        conc[head] = []
    for row in output[1:]:
        for col, head in enumerate(header):
            conc[head].append(row[col])
    return conc


def run_ex11(sim_folder):
    #sim_ws = os.path.abspath(os.path.join(workspace, sim_folder, "phreeqcrm"))
    sim_ws = workspace / sim_folder / "phreeqcrm"
    with change_directory(sim_ws):
        ip = phreeqc_mod.IPhreeqc()
        ip.load_database("phreeqc.dat")
        script = make_script()
        ip.run_string(script)
        conc = get_selected_output(ip)
    return conc

# Function to plot the model results


def plot_results_ct(
    sims, idx, solutes_idx, solutes, conc, longitudinal_dispersivity
):
    #if config.plotModel:
    if plot:
        _, sim_mf6gwf = sims
        #sim_ws = sim_mf6gwf.simulation_data.mfpath.get_sim_path()
        
        mf6gwt_ra = sim_mf6gwf.get_model(f"trans.{solutes[solutes_idx]}").obs.output.obs().data
        fig, axs = plt.subplots(1, 1, figsize=figure_size, dpi=300, tight_layout=True)
        # fig, axs = plt.subplots(5, 1, figsize=figure_size, dpi=300, tight_layout=True)
        # print(f"len(axs)={len(axs)}")
        #iskip = 4
        iskip = 16
        
        obsnames = ["CELL39"]
        simtimes = mf6gwt_ra["totim"]
        
        dispersion_coefficient = (
            longitudinal_dispersivity * specific_discharge
        )
        colors=["blue", "red", "green"]
        markers=["o","x","+"]
        # Modflow
        i = 0
        axs.plot(
            simtimes[::iskip]/28800.,
            mf6gwt_ra[obsnames[i]][::iskip],
            marker=markers[i],
            ls="none",
            mec=colors[i],
            mfc="none",
            markersize="4",
            label="Modflow",
        )
        i = 1
        var = np.array(conc[solutes[solutes_idx]])
        var = var * concentration_factor
        axs.plot(
            conc["PV"],
            var,
            #var[::iskip],
            marker=markers[i],
            ls="none",
            mec=colors[i],
            mfc="none",
            markersize="4",
            label="Phreeqc",
        )  
        i = 2
        if solutes[solutes_idx] == "Cl":
            var = np.array(conc["Cl_analytical"])
            axs.plot(
                conc["PV"],
                var,
                #var[::iskip],
                marker=markers[i],
                ls="none",
                mec=colors[i],
                mfc="none",
                markersize="4",
                label="Analytical",
            )
        axs.set_xlabel("Pore volumes")
        axs.set_ylabel(f"{solutes[solutes_idx]} Concentration (mmol/kgw)")
        axs.legend()

        # save figure
        if plot_save:
            figs_path.mkdir(exist_ok=True, parents=True)
            fpth = figs_path / f"{sim_name}.png"
            fig.savefig(fpth, dpi=600)

# Function that wraps all of the steps for each scenario
#
# 1. build_model,
# 2. write_model,
# 3. run_model, and
# 4. plot_results.
#


def scenario(idx, silent=True):
    key = list(parameters.keys())[idx]
    parameter_dict = parameters[key]

    sims = build_model(key, **parameter_dict)
    write_model(sims, silent=silent)
    success = run_model(sims, silent=silent)
    assert(success)
    
    if success:
        phreeqcrm, sim_mf6gwf = sims
        solutes = phreeqcrm.solutes

        conc =  run_ex11(sim_name)
        for sidx, solute in enumerate(solutes):
            if sidx > 2:
                plot_results_ct(sims, idx, sidx, solutes, conc, **parameter_dict)
    


def test_01():
    scenario(0, silent=False)


if __name__ == "__main__":
    scenario(0)
