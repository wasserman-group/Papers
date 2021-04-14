import n2v
import psi4
import json
import numpy as np

#Grid for plotting
npoints=1000
x = np.linspace(-5,5,npoints)[:,None]
y = np.zeros_like(x)
z = y
grid = np.concatenate((x,y,z), axis=1).T #Define grid for plotting

ne = psi4.geometry("""
0 1
H 0.0 0.0 -2.13679345
C 0.0 0.0  2.00561992
N 0.0 0.0  5.8087839cd
noreorient
nocom
units bohr
symmetry c1
"""
)

list_basis = ['6-31G', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z', 
                 'aug-cc-pvtz', 'aug-cc-pvqz',]

staroverov_results = {"basis" : list_basis, "occ_eigs" : [], "density_difference" : [], "vxc":[], "grid": None, "details" : []}

functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=True)[0]

for i in [list_basis[0]]:

    #Begin Nice and clean for each calculation
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    psi4.set_options({"reference"            : "rhf", 
                      "opdm"                 :  True,
                      "tpdm"                 :  True,
                    #   "dft_spherical_points" :   302,
                    #   "dft_radial_points"    :    75,
                      "dft_spherical_points" :   26,
                      "dft_radial_points"    :   26,
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
        ine.invert("mRKS", opt_max_iter=30, frac_old=0.8, init="scan")

        # DFT grid for density acccuracy
        vpot = psi4.core.VBase.build(ine.basis, functional, "RV")
        vpot.initialize()
        x, y, z, weights = vpot.get_np_xyzw()
        mask = np.bitwise_and(np.isclose(y, 0), np.isclose(z,0))
        order = np.argsort(x[mask])
        x_plot = x[mask][order]
        print(x_plot)
        print(x_plot.shape)
        # weights =  ine.generate_dft_grid(vpot=vpot)[3,:]

        dt   = ine.on_grid_density(Da=wfn.Da(), Db=wfn.Db(), vpot=vpot)
        dinv = ine.on_grid_density(Da=ine.Da,   Db=ine.Db, vpot=vpot)
        dd = np.sum( 2 * (dt - dinv) * weights )

        #VXC on grid
        vxc = ine.grid.vxc[mask][order]

        staroverov_results["occ_eigs"].append( ine.eigvecs_a[:wfn.nalpha()].tolist() )
        staroverov_results["density_difference"].append( dd )
        staroverov_results["vxc"].append( vxc.tolist() )

    except Exception as exception: 
        print(exception)
        staroverov_results["occ_eigs"].append( "FAILED" )
        staroverov_results["density_difference"].append( "FAILED" )
        staroverov_results["details"].append(exception)
        staroverov_results["grid"].append(x_plot.tolist())

        

    wfn = None
    e   = None

json = json.dumps(staroverov_results)
f = open("results/staroverov.json", "w")
f.write(json)
f.close()
