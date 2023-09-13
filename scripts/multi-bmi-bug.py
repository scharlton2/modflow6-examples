# Imports
import os
import sys

import flopy
import matplotlib.pyplot as plt
import numpy as np

import modflowapi
from modflowapi import Callbacks
from pathlib import Path
import shutil

# Append to system path to include the common subdirectory
sys.path.append(os.path.join("..", "common"))

# Import common functionality
import analytical
import config
from figspecs import USGSFigure

# Set figure properties specific to this problem
figure_size = (5, 3)

# Base simulation and model name and workspace
ws = config.base_ws
example_name = "moc"

# Scenario parameters - make sure there is at least one blank line before next item
parameters = {
    "ex11": {
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": 1.0,
        "decay_rate": 0.0,
        "solutes":                ['Na', 'Cl', 'K', 'Ca',  'N'  ],
        "initial_solution":       [ 1.0,  0.0, 0.2,  0.0,  1.2  ],   # mmol/kgw
        "influent_concentration": [ 0.0,  1.2, 0.0,  0.6,  0.0  ],   # mmol/kgw
    },
}

# Scenario parameter units - make sure there is at least one blank line before next item add parameter_units to add units to the scenario parameter table
parameter_units = {
    "longitudinal_dispersivity": "$m$",
    "retardation_factor": "unitless",
    "decay_rate": "$s^{-1}$",
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
specific_discharge = 1./720.  # Specific discharge ($m s^{-1}$)
hydraulic_conductivity = 1.0  # Hydraulic conductivity ($m s^{-1}$)
porosity = 1.0                # Porosity of mobile domain (unitless)
total_time = 14400.0          # Simulation time ($s$)

# Functions to build, write, run, and plot models
# MODFLOW 6 flopy GWF simulation object (sim) is returned

def get_sorption_dict(retardation_factor):
    sorption = None
    bulk_density = None
    distcoef = None
    if retardation_factor > 1.0:
        sorption = "linear"
        bulk_density = 1.0
        distcoef = (retardation_factor - 1.0) * porosity / bulk_density
    sorption_dict = {
        "sorption": sorption,
        "bulk_density": bulk_density,
        "distcoef": distcoef,
    }
    return sorption_dict

def get_decay_dict(decay_rate, sorption=False):
    first_order_decay = None
    decay = None
    decay_sorbed = None
    if decay_rate != 0.0:
        first_order_decay = True
        decay = decay_rate
        if sorption:
            decay_sorbed = decay_rate
    decay_dict = {
        "first_order_decay": first_order_decay,
        "decay": decay,
        "decay_sorbed": decay_sorbed,
    }
    return decay_dict

def build_mf6gwf(sim_folder, solutes, influent_concentration):
    # print(f"Building mf6gwf model...{sim_folder}")
    name = "flow"
    sim_ws = os.path.join(ws, sim_folder, "mf6gwf")
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=sim_ws, exe_name="mf6"
    )
    tdis_ds = ((total_time, 1, 1.0),)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
    flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
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
        auxiliary=solutes
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
    sim.libmf6 = Path(os.path.join(os.environ["USERPROFILE"], "AppData\\Local\\flopy\\bin", "libmf6")) # flow always uses the original
    print(f"sim.libmf6={sim.libmf6}")

    return sim

# MODFLOW 6 flopy GWF simulation object (sim) is returned
def build_mf6gwt(sim_folder, solute, initial_solution, longitudinal_dispersivity, retardation_factor, decay_rate):
    # print(f"Building mf6gwt_{solute} model...{sim_folder}")
    name = "trans"
    sim_ws = os.path.join(ws, sim_folder, f"mf6gwt_{solute}")
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=sim_ws, exe_name="mf6"
    )
    tdis_ds = ((total_time, 240, 1.0),)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwt = flopy.mf6.ModflowGwt(sim, modelname=name, save_flows=True)
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
    flopy.mf6.ModflowGwtic(gwt, strt=initial_solution, filename=f"{name}.ic")
    flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=porosity,
        **get_sorption_dict(retardation_factor),
        **get_decay_dict(decay_rate, retardation_factor > 1.0),
    )
    flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
    flopy.mf6.ModflowGwtdsp(
        gwt,
        xt3d_off=True,
        alh=longitudinal_dispersivity,
        ath1=longitudinal_dispersivity,
    )
    pd = [
        ("GWFHEAD", f"../mf6gwf/flow.hds", None),
        ("GWFBUDGET", "../mf6gwf/flow.bud", None),
    ]
    flopy.mf6.ModflowGwtfmi(gwt, packagedata=pd)
    sourcerecarray = [["WEL-1", "AUX", solute]]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)
    obs_data = {
        f"{solute}.obs.csv": [
            ("CELL00", "CONCENTRATION", (0, 0, 0)),
            ("CELL19", "CONCENTRATION", (0, 0, 19)),
            ("CELL39", "CONCENTRATION", (0, 0, 39)),
        ],
    }
    obs_package = flopy.mf6.ModflowUtlobs(
        gwt, digits=10, print_input=True, continuous=obs_data
    )
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{name}.cbc",
        concentration_filerecord=f"{name}.ucn",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )
    
    # @todo update for other OSs
    source_file = os.path.join(os.environ["USERPROFILE"], "AppData\\Local\\flopy\\bin", "libmf6.dll")
    dest_file = os.path.join(sim_ws, f"libmf6_{solute}.dll")
    shutil.copy(source_file, dest_file)
    
    # set vars for use in run_model
    sim.sim_ws = sim_ws
    sim.solute = solute
    # @todo update for other OSs
    sim.libmf6 = Path(os.path.join(os.environ["USERPROFILE"], "AppData\\Local\\flopy\\bin", "libmf6"))

    if config.copyAndRenameDLL:
        sim.libmf6 = Path(os.path.abspath(os.path.join(sim_ws, f"libmf6_{solute}")))
        print(f"sim.libmf6={sim.libmf6}")

    return sim

def build_model(sim_name, solutes, initial_solution, influent_concentration, longitudinal_dispersivity, retardation_factor, decay_rate):
    sims = None
    if config.buildModel:
        sim_mf6gwf = build_mf6gwf(sim_name, solutes, influent_concentration)
        sims = (sim_mf6gwf,)
        for i, solute in enumerate(solutes):
            sim_mf6gwt = build_mf6gwt(
                sim_name, solute, initial_solution[i], longitudinal_dispersivity, retardation_factor, decay_rate
                )
            sims = sims + (sim_mf6gwt,)
    return sims

def write_model(sims, silent=True):
    if config.writeModel:
        sim_mf6gwf, *sim_mf6gwts = sims
        sim_mf6gwf.write_simulation(silent=silent)
        for sim_mf6gwt in sim_mf6gwts:
            sim_mf6gwt.write_simulation(silent=silent)
    return


@config.timeit
def run_model(sims, silent=True):
    success = True
    if config.runModel:
        sim_mf6gwf, *sim_mf6gwts = sims
        mf6gwf = modflowapi.ModflowApi(sim_mf6gwf.libmf6, working_directory=sim_mf6gwf.sim_ws)
        mf6gwf.initialize()
        
        current_time = mf6gwf.get_current_time()
        end_time = mf6gwf.get_end_time()
        while current_time < end_time:
            mf6gwf.update()
            current_time = mf6gwf.get_current_time()
        mf6gwf.finalize()
        if not silent:
            print("(mf6gwf) NORMAL TERMINATION OF SIMULATION")
        
        for sim_mf6gwt in sim_mf6gwts:
            mf6 = modflowapi.ModflowApi(sim_mf6gwt.libmf6, working_directory=sim_mf6gwt.sim_ws)
            mf6.initialize()
            
            current_time = mf6.get_current_time()
            end_time = mf6.get_end_time()
            while current_time < end_time:
                mf6.update()
                current_time = mf6.get_current_time()
            mf6.finalize()
            if not silent:
                print(f"(mf6gwt({sim_mf6gwt.solute})) NORMAL TERMINATION OF SIMULATION")

    return success


@config.timeit
def run_model_as_list(sims, silent=True):
    success = True
    if config.runModel:
        sim_mf6gwf, *sim_mf6gwts = sims
        mf6gwf = modflowapi.ModflowApi(sim_mf6gwf.libmf6, working_directory=sim_mf6gwf.sim_ws)
        mf6gwf.initialize()
        
        current_time = mf6gwf.get_current_time()
        end_time = mf6gwf.get_end_time()
        while current_time < end_time:
            mf6gwf.update()
            current_time = mf6gwf.get_current_time()
        mf6gwf.finalize()
        if not silent:
            print("(mf6gwf) NORMAL TERMINATION OF SIMULATION")
        
        # initialize
        mf6gwts = [None] * len(sim_mf6gwts)
        for i, sim_mf6gwt in enumerate(sim_mf6gwts):
            mf6gwts[i] = modflowapi.ModflowApi(sim_mf6gwt.libmf6, working_directory=sim_mf6gwt.sim_ws)
            mf6gwts[i].initialize()

        while mf6gwts[0].get_current_time() < mf6gwts[0].get_end_time():
            for i in range(len(mf6gwts)):
                mf6gwts[i].update()
                # phreeqcrm.set_value() <- mf6gwts[i].get_value()
            # phreeqcrm.update()
            # for i in range(len(mf6gwts)):
                # mf6gwts[i].set_value() <- phreeqcrm.get_value()

        for i in range(len(mf6gwts)):
            mf6gwts[i].finalize()
        # phreeqcrm.finalize()


    return success

# Function to plot the model results
def plot_results_ct(
    sims, idx, solutes_idx, solutes, initial_solution, influent_concentration, longitudinal_dispersivity, retardation_factor, decay_rate
):
    if config.plotModel:
        sim_mf6gwf, *sim_mf6gwts = sims
        sim_mf6gwt = sim_mf6gwts[solutes_idx]
        fs = USGSFigure(figure_type="graph", verbose=False)

        sim_ws = sim_mf6gwt.simulation_data.mfpath.get_sim_path()
        mf6gwt_ra = sim_mf6gwt.get_model("trans").obs.output.obs().data
        fig, axs = plt.subplots(1, 1, figsize=figure_size, dpi=300, tight_layout=True)
        iskip = 5
        
        obsnames = ["CELL00", "CELL19", "CELL39"]
        simtimes = mf6gwt_ra["totim"]
        
        dispersion_coefficient = (
            longitudinal_dispersivity * specific_discharge / retardation_factor
        )
        colors=["blue", "red", "green"]
        for i, x in enumerate([0.05, 4.05, 11.05]):
            axs.plot(
                simtimes[::iskip],
                mf6gwt_ra[obsnames[i]][::iskip],
                marker="o",
                ls="none",
                mec=colors[i],
                mfc="none",
                markersize="4",
                label=obsnames[i],
            )
        axs.set_ylim(0, 1.4)
        axs.set_xlim(0, 1.2 * total_time)
        axs.set_xlabel("Time (seconds)")
        axs.set_ylabel(f"{solutes[solutes_idx]} Concentration (mmol/kgw)")            
        axs.legend()

        # save figure
        if config.plotSave:
            sim_folder = os.path.split(sim_ws)[0]
            sim_folder = os.path.basename(sim_folder)
            fname = f"{sim_folder}-{solutes[solutes_idx]}-ct{config.figure_ext}"
            fpth = os.path.join(ws, "..", "figures", fname)
            fig.savefig(fpth)

def plot_results_cd(
    sims, idx, solutes_idx, solutes, initial_solution, influent_concentration, longitudinal_dispersivity, retardation_factor, decay_rate
):
    if config.plotModel:
        #print(f"Plotting {solutes[solutes_idx]} versus x model results...")
        sim_mf6gwf, *sim_mf6gwts = sims
        sim_mf6gwt = sim_mf6gwts[solutes_idx]
        fs = USGSFigure(figure_type="graph", verbose=False)
        
        ucnobj_mf6 = sim_mf6gwt.trans.output.concentration()        
        
        fig, axs = plt.subplots(1, 1, figsize=figure_size, dpi=300, tight_layout=True)
        ctimes = [14400.]
        #x = np.linspace(0.5 * delr, system_length - 0.5 * delr, ncol - 1)
        x = np.linspace(0.5 * delr, system_length - 0.5 * delr, ncol)
        dispersion_coefficient = (
            longitudinal_dispersivity * specific_discharge / retardation_factor
        )
        
        colors=["blue", "red", "green"]
        for i, t in enumerate(ctimes):
            simconc = ucnobj_mf6.get_data(totim=t).flatten()
            assert(len(x) == len(simconc))
            axs.plot(
                x,
                simconc,
                marker="o",
                ls="none",
                mec=colors[i],
                mfc="none",
                markersize="4",
            )
        axs.set_ylim(0, 1.4)
        #axs.set_xlim(0, .08)
        axs.set_xlim(0, 1.2 * system_length)
        if idx in [0]:
            axs.text(.070, 0.7, "t=14400 s")
        axs.set_xlabel("Distance (m)")
        axs.set_ylabel(f"{solutes[solutes_idx]} Concentration (mmol/kgw)")
        #plt.legend()

        # save figure
        if config.plotSave:
            sim_ws = sim_mf6gwt.simulation_data.mfpath.get_sim_path()
            sim_folder = os.path.split(sim_ws)[0]
            sim_folder = os.path.basename(sim_folder)
            fname = f"{sim_folder}-{solutes[solutes_idx]}-cd{config.figure_ext}"
            fpth = os.path.join(ws, "..", "figures", fname)
            fig.savefig(fpth)

# Function that wraps all of the steps for each scenario
#
# 1. build_model,
# 2. write_model,
# 3. run_model, and
# 4. plot_results.

def scenario(idx, silent=True):
    key = list(parameters.keys())[idx]
    parameter_dict = parameters[key]
    
    sims = build_model(key, **parameter_dict)
    write_model(sims, silent=silent)

    if config.runAsList:
        success = run_model_as_list(sims, silent=silent)
    else:
        success = run_model(sims, silent=silent)
    
    if success:
        for sidx, solute in enumerate(parameter_dict["solutes"]):
            plot_results_ct(sims, idx, sidx, **parameter_dict)
            plot_results_cd(sims, idx, sidx, **parameter_dict)

def test_01():
    scenario(0, silent=False)

if __name__ == "__main__":
    print("Start")
    config.copyAndRenameDLL = True      # if True each mf6gwt loads its own dll
    config.runAsList = True             # if True will crash the python kernel
    scenario(0)
    print("Finished")
