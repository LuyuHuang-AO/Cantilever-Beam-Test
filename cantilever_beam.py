import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import SofaRuntime
import Sofa.Gui
import numpy as np
import json
from json import JSONEncoder
import csv
from pathlib import Path
import numpy
from stl import mesh
import matplotlib.pyplot as plt


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# utility functions
def read_simulation_param(file_name):
    curr_path = Path.cwd()
    file_path = str(curr_path) + '/' + file_name  # The input file should be at the same location as the simulator python file

    # Opening JSON file
    f = open(file_path, 'r')
    # returns JSON object as a dictionary
    data = json.loads(f.read())

    # Define parameters for simulation
    paramdict_object = {
        'rod_E': data['physics_parameters']['paramdict_obj']['rod_E'],
        'rod_poisson_ratio': data['physics_parameters']['paramdict_obj']['rod_poisson_ratio'],
        'rod_mass': data['physics_parameters']['paramdict_obj']['rod_mass'],
        'rod_length': data['physics_parameters']['paramdict_obj']['rod_length'],
        'rod_radius': data['physics_parameters']['paramdict_obj']['rod_radius']
    }
    paramdict_move = {
        'force': data['physics_parameters']['paramdict_move']['force']
    }
    paramdict_sim = {
        'dt': data['physics_parameters']['paramdict_sim']['dt'],
        'total_itr': data['physics_parameters']['paramdict_sim']['total_itr'],
        'gravity': data['physics_parameters']['paramdict_sim']['gravity'],
        'CGLinearSolver_thresh': data['physics_parameters']['paramdict_sim']['CGLinearSolver_thresh'],
        'CGLinearSolver_tol': data['physics_parameters']['paramdict_sim']['CGLinearSolver_tol'],
        'CGLinearSolver_iter': data['physics_parameters']['paramdict_sim']['CGLinearSolver_iter'],
    }

    f.close()

    return paramdict_object, paramdict_move, paramdict_sim

def reformat(file_name):
    '''
    This function reformat SOFA's output for position
    '''
    # Collect data by dimension
    data_over_time_x = []
    data_over_time_y = []
    data_over_time_z = []

    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        for row in reader:
            # skip over the first two rows
            if row[0][0] != "#":
                # current time step's position data for all nodes
                curr_time_x = []
                curr_time_y = []
                curr_time_z = []
                # loop over all nodes in each timestep 
                for i in range(1,len(row)-1):
                    # obtain each node's 3D position seperately 
                    curr_time_x.append(float(row[i].split(' ')[0]))
                    curr_time_y.append(float(row[i].split(' ')[1]))
                    curr_time_z.append(float(row[i].split(' ')[2]))
            
                data_over_time_x.append(curr_time_x)
                data_over_time_y.append(curr_time_y)
                data_over_time_z.append(curr_time_z)
    
    # Export in JSON format
    data = {
        "x_dim": np.array(data_over_time_x),
        "y_dim": np.array(data_over_time_y),
        "z_dim": np.array(data_over_time_z)
    }

    # Output
    if 'pos' in file_name:
        with open('positions_over_time.json', 'w') as write_file:
            json.dump(data, write_file, cls=NumpyArrayEncoder)
    if 'force' in file_name:
        with open('forces_over_time.json', 'w') as write_file:
            json.dump(data, write_file, cls=NumpyArrayEncoder) 

def calc_centroid(positions_over_time, num_segment):
    '''
    Given a JSON dictionary of all indices positions over time, calculate the centroids over time by segments
    '''
    total_indices = len(positions_over_time.get('x_dim')[0])
    total_timesteps = len(positions_over_time.get('x_dim'))

    indices_per_seg = int(total_indices / num_segment)

    curr_seg_x = positions_over_time.get('x_dim')[0][0]
    curr_seg_y = positions_over_time.get('y_dim')[0][0]
    curr_seg_z = positions_over_time.get('z_dim')[0][0]

    centroids_over_time_x = []
    centroids_over_time_y = []
    centroids_over_time_z = []

    # run over each time step
    for i in range(total_timesteps):
        # store each segment's centroid position
        curr_time_centroids_x = []
        curr_time_centroids_y = []
        curr_time_centroids_z = []

        # loop over each segment
        for j in range(0,total_indices,indices_per_seg):
           
            # Obtain the x, y, z position arrays for the current segment.
            curr_seg_x = positions_over_time.get('x_dim')[i][j:j+indices_per_seg-1]
            curr_seg_y = positions_over_time.get('y_dim')[i][j:j+indices_per_seg-1]
            curr_seg_z = positions_over_time.get('z_dim')[i][j:j+indices_per_seg-1]

            # Calculate the centroid (average position) for each dimension of the current segment
            curr_time_centroids_x.append(sum(curr_seg_x)/len(curr_seg_x))
            curr_time_centroids_y.append(sum(curr_seg_y)/len(curr_seg_y))
            curr_time_centroids_z.append(sum(curr_seg_z)/len(curr_seg_z))

        centroids_over_time_x.append(curr_time_centroids_x)
        centroids_over_time_y.append(curr_time_centroids_y)
        centroids_over_time_z.append(curr_time_centroids_z)

    # Export in JSON format
    data = {
        "x_dim": np.array(centroids_over_time_x),
        "y_dim": np.array(centroids_over_time_y),
        "z_dim": np.array(centroids_over_time_z)
    }

    # Output
    with open('centroids_over_time.json', 'w') as write_file:
        json.dump(data, write_file, cls=NumpyArrayEncoder) 


def analytical_EndDeflect(rod_radius, rod_length, param_force, E):
    """
    Calculate the analytical solution for the maximum deflection of a beam with a given radius, length, force, and elastic modulus.
    Args:
        rod_radius (float): the radius of the beam.
        rod_length (float): the length of the beam.
        param_force (string): the force applied to the beam in "x y z" form
        E (float): the elastic modulus of the beam material.
    Returns:
        float: the maximum deflection of the beam.
    """
    I = calc_momentOfInertia(float(rod_radius)) # (m^4)
    force = param_force.split(' ')[1] # N
    maxDeflection = calc_maxDeflection(float(force), float(rod_length), float(E), I)
    return maxDeflection

def calc_momentOfInertia(rod_radius):
    """
    Calculate the moment of inertia of an object with a circular base, given its base radius.
    Args:
        base_radius (float): the radius of the object's circular base (m)
    Returns:
        float: the moment of inertia of the object (m^4)
    """
    return (1/4) * np.pi * (rod_radius ** 4)

def calc_maxDeflection(force, rod_length, E, I):
    """
    Calculate the maximum deflection of a beam with a given force, length, elastic modulus, and moment of inertia.
    Args:
        force (float): the force applied to the beam. (N)
        length (float): the length of the beam. (m)
        E (float): the elastic modulus of the beam material. (pa)
        I (float): the moment of inertia of the beam. (m^4)
    Returns:
        float: the maximum deflection of the beam. (m)
    """
    return (force * pow(rod_length,3)) / (3 * E * I)

def simulation_EndDeflec(centroids_over_time):
    """
    Assuming the force is being applied towards y-axis positive direction,
    calculate the deflection of a rod over time, given its positions at the first time step and 
    the time step where it reaches maximum y-value.
    
    Args:
        centroids_over_time (dict): a dictionary containing the positions of the rod over time.
    
    Returns:
        the deflection of the end point
    """
    # extract out the tip position for the first time step
    y_pos_beg = np.array(centroids_over_time.get('y_dim')[0][-1])

    # extract all positions in y_dim and find the maximum y position
    y_pos_all = [np.array(position[-1]) for position in centroids_over_time.get('y_dim')]
    y_pos_max = np.max(y_pos_all)

    # calculate the deflection for rod's tip
    deflection_over_time = [pos - y_pos_beg for pos in y_pos_all]

    # plot deflection over time
    # plt.figure(figsize=(10,6))
    # plt.plot(deflection_over_time)
    # plt.grid(True)
    # plt.title('Deflection over time')
    # plt.xlabel('Time')
    # plt.ylabel('Tip Deflection')
    # plt.show()


    return y_pos_max - y_pos_beg


def physics_validation_analysis(paramdict_object, paramdict_move, pos_output):
    '''
    This function performs a canteliver beam test and analyzes the result
    '''
    # reformat simulation's position output to a new .json file
    print('Reformatting SOFA outputs...')
    reformat(pos_output)

    f = open('positions_over_time.json','r')
    data = json.loads(f.read())  # returns JSON object as a dictionary

    # extract centroids and export new JSON file
    num_seg = 1 # we are only tracking the rod's tip
    calc_centroid(data,num_seg)
    f = open('positions_over_time.json','r') # this is a new positions_over_time.json based on centroids
    centroids_over_time = json.loads(f.read())

    print('Physics validation starting...')

    # simulation result
    Dflec_s = simulation_EndDeflec(centroids_over_time)
    print(f"Simulation result on end point deflection: {Dflec_s}mm")
    
    # analytical result
    Dflec_a = analytical_EndDeflect(paramdict_object['rod_radius'], paramdict_object['rod_length'], paramdict_move['force'], paramdict_object['rod_E'])
    print(f"Analytical result on end point deflection: {Dflec_a}mm")
        
    error = Dflec_a - Dflec_s
    print(f'The error between analytical and simulated results is {error}mm')



def physics_val(root, param_obj, param_move, param_sim):
    '''
    This function is specifcally for physics validation of rod
    The function is called before initialization to define what components are inside the graph
    '''
    root.addObject('RequiredPlugin', name="loadSOFAModules", pluginName="SofaValidation Sofa.Component.Engine.Select Sofa.Component.SolidMechanics.FEM.Elastic Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Detection.Algorithm Sofa.Component.LinearSolver.Iterative Sofa.Component.Mass Sofa.Component.MechanicalLoad Sofa.Component.StateContainer Sofa.Component.ODESolver.Backward Sofa.Component.IO.Mesh")

    root.gravity=param_sim['gravity']
    root.name="root"
    root.dt=param_sim['dt']
    root.addObject('DefaultAnimationLoop', computeBoundingBox=False)
    root.bbox=[[-1,-1,-1],[1,1,1]]
    root.addObject('MeshGmshLoader', name = "meshLoaderCyl", filename="mesh/cylinder_0.3.msh") 

    # Rod
    Rod = root.addChild('Rod')
    Rod.addObject('EulerImplicitSolver')
    Rod.addObject('CGLinearSolver', threshold=param_sim['CGLinearSolver_thresh'], tolerance=param_sim['CGLinearSolver_tol'], iterations=param_sim['CGLinearSolver_iter'])
    Rod.addObject('TetrahedronSetTopologyContainer', name="topo2", src="@../meshLoaderCyl" )
    Rod.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo2")

    Rod.addObject('MechanicalObject', template="Vec3d", name="rodModel", showObject="1")
    Rod.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=param_obj['rod_E'], poissonRatio=param_obj['rod_poisson_ratio'])
    Rod.addObject('UniformMass', totalMass=param_obj['rod_mass'] )

    # Cantiliver Beam Set Up
    # fixed_region = "-10 -10 -35 10 10 -32" #original
    fixed_region = "-10 -10 -1 10 10 0.1"
    Rod.addObject('BoxROI', name="fixed_region", box=fixed_region) # region to be fixed in place
    Rod.addObject('FixedConstraint', indices="@fixed_region.indices")
    Rod.addObject('ConstantForceField', force=param_move['force'],indices="2") # the centroid point
 
    # Rod Visual Model
    VisuCly = Rod.addChild('VisuCly')
    VisuCly.addObject('OglModel', name="VisualCly", src="@../../meshLoaderCyl")
    VisuCly.addObject('IdentityMapping', name="Mapping2", input="@../rodModel", output="@VisualCly" )

    # Record positions over time of the selected vertices (tip of the rod)
    tip_region = "-10 -10 22.95 10 10 25"
    Rod.addObject('BoxROI', name="tip_region", box=tip_region)
    pos_output_file = "sofa_monitor_pos"
    force_output_file = "sofa_monitor_force"
    Rod.addObject('Monitor', template="Vec3d", name=pos_output_file, listening="1", indices="@tip_region.indices", showPositions="1", PositionsColor="1 0 0 1", ExportPositions="true")
    Rod.addObject('Monitor', template="Vec3d", name=force_output_file, listening="1", indices="@tip_region.indices", showForces="1", ExportForces="true")
    return pos_output_file + '_x.txt', force_output_file + '_f.txt'


def main():
    # Make sure to load all SOFA libraries
    SofaRuntime.importPlugin("SofaBaseMechanics")
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("CImgPlugin")

    # Load in input parameters
    paramdict_object, paramdict_move, paramdict_sim = read_simulation_param('physicsval_parameters.txt')

    # Create the root node
    root = Sofa.Core.Node("root")

    # Call the below 'createScene' function to create the scene graph
    pos_output, force_output = physics_val(root,
                                            paramdict_object,
                                            paramdict_move,
                                            paramdict_sim)
    Sofa.Simulation.init(root)
    print("Simulation starts...")

    # Launch the GUI (qt or qglviewer)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    # Initialization of the scene will be done here
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    print("Simulation is completed.")

    # Analysis
    print("Analysis begins...")
    physics_validation_analysis(paramdict_object, paramdict_move, pos_output)
    print("Analysis is completed.")

if __name__ == "__main__":
    # args = parser.parse_args()
    main()