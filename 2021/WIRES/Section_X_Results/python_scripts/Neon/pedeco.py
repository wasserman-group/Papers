import n2v
import psi4
import json
import numpy as np

#Grid for plotting
npoints=1000
x = np.linspace(0,5,npoints)[:,None]
y = np.zeros_like(x)
z = y
grid = np.concatenate((x,y,z), axis=1).T

ne = psi4.geometry("""
0 1
Ne 0.0 0.0 0.0
noreorient
nocom
units bohr
symmetry c1
"""
)

list_basis = ['cc-pvtz', 'cc-pvqz', 'cc-pv5z', 
                 'aug-cc-pvtz', 'aug-cc-pvqz',]

pdeco_results = {"basis" : list_basis, "occ_eigs" : [], "density_difference" : [], "vxc" : [], "grid": np.linspace(0,10,npoints).tolist(), "details" : []}

functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=True)[0]

for i in list_basis:

    #Begin Nice and clean for each calculation
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    psi4.set_options({"reference"            : "rhf", 
                      "dft_spherical_points" :   302, 
                      "dft_radial_points"    :    75,
                      "basis"                :     i,})

    psi4.set_num_threads(12)
    psi4.set_options({"save_jk" : True})
    psi4.set_memory(int(2.50e9))

    # Target calculation
    e, wfn = psi4.properties("detci", return_wfn=True, properties=["dipole"], molecule=ne)
    print(f"energy for basis {i}: {e}")

    # WuYang Inversion
    ine = n2v.Inverter(wfn)

    try:
        ine.invert("PDECO", opt_max_iter=200, guide_potential_components=["fermi_amaldi"], gtol=1e-3)

        # DFT grid for density acccuracy
        vpot = psi4.core.VBase.build(ine.basis, functional, "RV")
        vpot.initialize()
        weights =  ine.generate_dft_grid(vpot=vpot)[3,:]

        dt   = ine.on_grid_density(Da=wfn.Da(), Db=wfn.Db(), vpot=vpot)
        dinv = ine.on_grid_density(Da=ine.Da,   Db=ine.Db, vpot=vpot)
        dd = np.sum( (dt - dinv) * weights )

        #VXC on grid
        vH, vFA = ine.on_grid_esp(grid=grid)[1:3]
        vrest = ine.on_grid_ao(ine.v_pbs, grid=grid)
        vxc = vFA + vrest - vH

        pdeco_results["occ_eigs"].append( ine.eigvecs_a[:wfn.nalpha()].tolist() )
        pdeco_results["density_difference"].append( dd )
        pdeco_results["vxc"].append( vxc.tolist() )

    except Exception as exception: 
        pdeco_results["occ_eigs"].append( "FAILED" )
        pdeco_results["density_difference"].append( "FAILED" )
        pdeco_results["details"].append(exception)

    wfn = None
    e   = None

json = json.dumps(pdeco_results)
f = open("results/pdeco.json", "w")
f.write(json)
f.close()
