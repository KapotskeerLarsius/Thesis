import vtk
import trimesh

def calculate_point_cloud_volume(points):

    """
    This script creates a triangular mesh of points from a point cloud and
    then creates a convex hull of the triangular mesh and calculates the volume of it.
    The trimesh library was used

    Paramaters: 
        points: np.array of point cloud points
    Returns:
        volume: volume of point cloud
    """

    # Create a trimesh object from the points
    mesh = trimesh.Trimesh(vertices=points)
    # Calculate the volume using the convex hull of the mesh
    volume = mesh.convex_hull.volume

    return volume

def separate_xyz(points):

    """
    Seperates x,y,z values from a np.array of point cloud points.
    This was needed to calculate surface area using the vtk library

    Paramaters: 
        coordinates: np.array of point cloud points
    Returns:
        x_values, y_values, z_values : x,y,z values seperated from the points nparray 
    """

    # Get the  x, y, z values from the point cloud points
    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2]
    
    return x_values, y_values, z_values

def calc_surface_area(points):

    """
    This script creates a triangular mesh of points from a point cloud and calculates surface area of it.
    It also visualizes the triangular mesh.

    The vtk library was used

    Paramaters: 
        points: np.array of point cloud points
    Returns:
        surface area surface area of point cloud
    """
    
    # Extract the  x, y, z values from the point cloud points
    x, y, z = separate_xyz(points)

    # Create points using the coordinates
    vtk_points = vtk.vtkPoints()

    for i in range(len(x)):
        vtk_points.InsertNextPoint(x[i], y[i], z[i])

    # Create a polydata object
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)

    # Triangulate the points to form the surface
    triangulation = vtk.vtkDelaunay2D()
    triangulation.SetInputData(mesh)
    triangulation.Update()

    # Extract the triangles (cells) from the triangulated mesh
    triangles = triangulation.GetOutput().GetPolys()

    # Create a mapper to map the triangulated mesh data to graphical elements
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(triangulation.GetOutput())

    # Create an actor to represent the triangulated mesh visually
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set a property (optional) to customize the appearance, e.g., color
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray

    # Create a renderer to display the actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    # Set the background color (optional)
    renderer.SetBackground(0.2, 0.3, 0.4)  # Dark blue

    # Create a render window to display the renderer
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create an interactor to handle user interaction (optional)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # # Initialize the visualization (start the rendering loop)
    interactor.Initialize()
    interactor.Start()

    # Calculate the surface area
    area = vtk.vtkMassProperties()
    area.SetInputData(triangulation.GetOutput())
    area.Update()

    # Get the surface area
    surface_area = area.GetSurfaceArea()

    return surface_area