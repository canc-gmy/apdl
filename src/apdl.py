from pathlib import Path
from ansys.mapdl.core import Mapdl

def create_model_and_solve_simply_supported_edges(mapdl: Mapdl, x_force: float, y_force: float) -> None:
    """
    Docstring for create_model_and_solve 
    Creates a finite element model in MAPDL, applies boundary conditions and loads, and solves the static analysis.

    This function defines a square plate with specified material properties, element types, and section data.
    It then meshes the plate, applies a simply supported boundary condition at one corner, symmetry conditions
    along two edges (for quarter symmetry), and a concentrated force at a specified (x, y) location.
    Finally, it solves the static analysis.

    :param mapdl: Mapdl Instance
    :type mapdl: Mapdl
    :param x_force: X-coordinate for force application in the global reference frame
    :type x_force: float
    :param y_force: Y-coordinate for force application in the global reference frame
    :type y_force: float
    """

    mapdl.prep7()

    # lenght of a half 1m size square [m]
    l = float(1.0)  

    # Section thickness [m]
    s_thk = 0.002  

    # ==========================================================================
    # MATERIAL DEFINITION
    # ==========================================================================

    m1 = 1
    # Steel Elastic Modulus [Pa]
    ex = float(12.0/(l*s_thk**3))
    prxy = 0
    mapdl.mp("EX", m1, ex)
    mapdl.mp("PRXY", m1, prxy)

    # ==========================================================================
    # GEOMETRY DEFINITION
    # ==========================================================================
    

    k1 = mapdl.k(npt=1, x=0, y=0)
    k2 = mapdl.k(npt=2, x=l, y=0)
    k3 = mapdl.k(npt=3, x=l, y=l)
    k4 = mapdl.k(npt=4, x=0, y=l)

    l1 = mapdl.l(k1, k2)
    l2 = mapdl.l(k2, k3)
    l3 = mapdl.l(k3, k4)
    l4 = mapdl.l(k4, k1)

    a1 = mapdl.a(k1, k2, k3, k4)

    # ==========================================================================
    # ELEMENT DEFINITION
    # ==========================================================================

    e1 = mapdl.et(ename="SHELL181")

    # Element stiffness keyopt(1)
    # Bending and membrane stiffness (default)
    mapdl.keyopt(e1, "1", "0")

    # Integration option keyopt(3)
    # Reduced integration with hourglass control (default)
    mapdl.keyopt(e1, "3", "2")

    # Shell normal orientation option keyopt(4)
    # Calculated from element connectivity (default)
    mapdl.keyopt(e1, "4", "0")

    # Curved shell formulation keyopt(5)
    # Advanced curved-shell formulation
    mapdl.keyopt(e1, "5", "1")

    # Specify layer data storage keyopt(8)
    # Store data for TOP, BOTTOM, and MID for all layers
    mapdl.keyopt(e1, "8", "2")

    # User thickness option keyopt(9)
    # No user subroutine to provide initial thickness (default)
    mapdl.keyopt(e1, "9", "0")

    # Default element x axis (x0) orientation keyopt(11)
    # First parametric direction at the element centroid (default)
    mapdl.keyopt(e1, "11", "0")

    # ==========================================================================
    # SECTION DEFINITION
    # ==========================================================================

    s1 = int(1)
    mapdl.sectype(secid=s1, type_="SHELL")

    # thickness, material, angle
    mapdl.secdata(s_thk, m1, "0")

    # ==========================================================================
    # MESH DEFINITION
    # ==========================================================================

    mapdl.aatt(mat=m1, type_=e1, secn=s1)

    # mesh element size [m]
    mesh_size = 0.002
    mapdl.aesize(a1, mesh_size)

    # mapped mesh
    mapdl.mshkey(key=1)

    mapdl.amesh(a1)

    # ==========================================================================
    # BOUNDARY CONDITION DEFINITION
    # ==========================================================================

    # Simply supported edges (UZ = 0)
    mapdl.dl(l1, a1, "UX", 0)
    mapdl.dl(l1, a1, "UY", 0)
    mapdl.dl(l1, a1, "UZ", 0)
    mapdl.dl(l2, a1, "UZ", 0)
    mapdl.dl(l3, a1, "UZ", 0)
    mapdl.dl(l4, a1, "UZ", 0)

    # ==========================================================================
    # LOAD DEFINITION
    # ==========================================================================

    mapdl.nsel(type_="S", item="LOC", comp="X", vmin=str(x_force))
    mapdl.nsel(type_="R", item="LOC", comp="Y", vmin=str(y_force))

    # Intensity of z-component of force [N]
    fz = 1
    mapdl.f("ALL", "FZ", -fz)

    # ==========================================================================
    # SOLUTION DEFINITION
    # ==========================================================================

    mapdl.slashsolu()
    mapdl.antype("STATIC")

    mapdl.outres("ALL")

    mapdl.allsel()

    mapdl.solve()

# def create_model_and_solve_full(mapdl: Mapdl, x_force: float, y_force: float) -> None:
#     """
#     Docstring for create_model_and_solve
    
#     :param mapdl: Mapdl Instance
#     :type mapdl: Mapdl
#     :param x_force: X-coordinate for force application in the global reference frame
#     :type x_force: float
#     :param y_force: Y-coordinate for force application in the global reference frame
#     :type y_force: float
#     """

#     mapdl.prep7()

#     # ==========================================================================
#     # MATERIAL DEFINITION
#     # ==========================================================================

#     m1 = 1
#     # Steel Elastic Modulus [Pa]
#     ex = float(200e9)
#     prxy = 0.3
#     mapdl.mp("EX", m1, ex)
#     mapdl.mp("PRXY", m1, prxy)

#     # ==========================================================================
#     # GEOMETRY DEFINITION
#     # ==========================================================================
    
#     # lenght of a half 1m size square [m]
#     l = float(1)    

#     k1 = mapdl.k(npt=1, x=0, y=0)
#     k2 = mapdl.k(npt=2, x=l, y=0)
#     k3 = mapdl.k(npt=3, x=l, y=l)
#     k4 = mapdl.k(npt=4, x=0, y=l)

#     l1 = mapdl.l(k1, k2)
#     l2 = mapdl.l(k2, k3)
#     l3 = mapdl.l(k3, k4)
#     l4 = mapdl.l(k4, k1)

#     a1 = mapdl.a(k1, k2, k3, k4)

#     # ==========================================================================
#     # ELEMENT DEFINITION
#     # ==========================================================================

#     e1 = mapdl.et(ename="SHELL181")

#     # Element stiffness keyopt(1)
#     # Bending and membrane stiffness (default)
#     mapdl.keyopt(e1, "1", "0")

#     # Integration option keyopt(3)
#     # Reduced integration with hourglass control (default)
#     mapdl.keyopt(e1, "3", "2")

#     # Shell normal orientation option keyopt(4)
#     # Calculated from element connectivity (default)
#     mapdl.keyopt(e1, "4", "0")

#     # Curved shell formulation keyopt(5)
#     # Advanced curved-shell formulation
#     mapdl.keyopt(e1, "5", "1")

#     # Specify layer data storage keyopt(8)
#     # Store data for TOP, BOTTOM, and MID for all layers
#     mapdl.keyopt(e1, "8", "2")

#     # User thickness option keyopt(9)
#     # No user subroutine to provide initial thickness (default)
#     mapdl.keyopt(e1, "9", "0")

#     # Default element x axis (x0) orientation keyopt(11)
#     # First parametric direction at the element centroid (default)
#     mapdl.keyopt(e1, "11", "0")

#     # ==========================================================================
#     # SECTION DEFINITION
#     # ==========================================================================

#     s1 = int(1)
#     mapdl.sectype(secid=s1, type_="SHELL")

#     # Section thickness [m]
#     s_thk = 0.002

#     # thickness, material, angle
#     mapdl.secdata(s_thk, m1, "0")

#     # ==========================================================================
#     # MESH DEFINITION
#     # ==========================================================================

#     mapdl.aatt(mat=m1, type_=e1, secn=s1)

#     # mesh element size [m]
#     mesh_size = 0.002
#     mapdl.aesize(a1, mesh_size)

#     # mapped mesh
#     mapdl.mshkey(key=1)

#     mapdl.amesh(a1)

#     # ==========================================================================
#     # BOUNDARY CONDITION DEFINITION
#     # ==========================================================================

#     # Simply supported coners (UZ = 0)
#     mapdl.dk(k1, "UZ", 0)
#     mapdl.dk(k2, "UZ", 0)
#     mapdl.dk(k3, "UZ", 0)
#     mapdl.dk(k4, "UZ", 0)
#     mapdl.dk(k1, "UX", 0)
#     mapdl.dk(k2, "UX", 0)
#     mapdl.dk(k3, "UX", 0)
#     mapdl.dk(k4, "UX", 0)
#     mapdl.dk(k1, "UY", 0)
#     mapdl.dk(k2, "UY", 0)
#     mapdl.dk(k3, "UY", 0)
#     mapdl.dk(k4, "UY", 0)

#     # ==========================================================================
#     # LOAD DEFINITION
#     # ==========================================================================

#     mapdl.nsel(type_="S", item="LOC", comp="X", vmin=str(x_force))
#     mapdl.nsel(type_="R", item="LOC", comp="Y", vmin=str(y_force))

#     # Intensity of z-component of force [N]
#     fz = 1
#     mapdl.f("ALL", "FZ", -fz)

#     # ==========================================================================
#     # SOLUTION DEFINITION
#     # ==========================================================================

#     mapdl.slashsolu()
#     mapdl.antype("STATIC")

#     mapdl.outres("ALL")

#     mapdl.allsel()

#     mapdl.solve()

def save_db(mapdl: Mapdl, dir: Path) -> None:
    dir.mkdir(exist_ok=True)
    mapdl.db.save(dir / (mapdl.jobname + ".db"), "ALL")
    mapdl.post1()
    mapdl.reswrite(dir / mapdl.jobname)