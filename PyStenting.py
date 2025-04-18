import numpy as np
import pyvista as pv
from splipy.curve_factory import cubic_curve
from scipy.spatial import ConvexHull, KDTree


'''
Basic Functions
'''

def rotate_layer(origin,tangent,vertices):
    
    '''
    Rotate vertices in the z=0 plane such that it lies in a plane with some 
    normalized vector (tangent) as its normal, then displaces the points
    by some vector (origin)

    origin: A 3D vector representing the translation (displacement) to apply after rotation.

    tangent: A 3D vector representing the normal of the target plane. The vertices will be rotated such that their original plane (z=0) aligns with this new plane.

    vertices: An array of 3D points (shape (n, 3)) to be transformed.
    '''
    
    x = vertices.T
    # The tangent vector is normalized to ensure it's a unit vector.
    t = tangent/np.linalg.norm(tangent)
    # Compute the rotation angle:
    # The angle angle is calculated between the normalized tangent and the [0, 0, 1] vector (the original plane's normal, the z-axis).
    # This gives the angle needed to rotate the z=0 plane to align with the new plane.
    angle = np.arccos(np.dot(t,np.array([0,0,1])))
    
    # If tangent has non-zero x or y components (if t[0] or t[1]), a rotation is required.
    if t[0] or t[1]:
        # The tangent's x and y components are normalized (to compute the in-plane rotation).
        # Suppose that t = [tx,ty,tz]
        # This operation will make t = [tx/sqrt(tx^{2}+ty^{2}), ty/sqrt(tx^{2}+ty^{2}), tz/sqrt(tx^{2}+ty^{2}))]
        t /= np.linalg.norm(t[:-1])
        # M1: Rotates the tangent vector to align its projection in the xy-plane with the x-axis.
        # t[0] = tx/sqrt(tx^{2}+ty^{2}) = cos(theta)
        # t[1] = ty/sqrt(tx^{2}+ty^{2}) = sin(theta)
        # Therefore M1 is a standard rotation matrix along z axis
        M1 = np.array([[t[0],-t[1],0],
                       [t[1],t[0],0],
                       [0,0,1]])
        # M2: Rotates around the y-axis by angle to align the z-axis with the tangent vector.
        # M2 is a standard rotation matrix along y axis
        M2 = np.array([[np.cos(angle),0,np.sin(angle)],
                       [0,1,0],
                       [-np.sin(angle),0,np.cos(angle)]])
        
        # The full rotation M is computed as M = M1^{T} · (M2 · M1) 
        # M1^{T} undoing the initial xy rotation after applying the tilt.
        M = np.dot(M2,M1)
        M = np.dot(M1.transpose(),M)
        x = np.dot(M,x)
        
        return origin + x.T

    else:
        return origin + vertices


'''
Objects
'''        

class FlowDiverter:
    
    def __init__(self, unit_cell, radius, height, tcopy, hcopy, 
                 strut_radius = 0.05, centerline=None, offset_angle:float=0): #, nodes=None, lines=None):
        '''
        unit_cell: A Pattern object defining the base geometry

        radius: Cylinder radius for the stent

        height: Total length of the stent

        tcopy, hcopy: Number of copies in tangential (circumferential) and longitudinal (lengthwise) directions

        strut_radius: Thickness of the stent struts

        centerline: Optional spline path for curved stents

        offset_angle: Rotational offset for pattern alignment

        Key initialization steps:

        1. Stores input parameters and calculates derived dimensions
        2. Computes layer counts and angular spacing between nodes
        3. Generates the initial mesh using pattern_wrap()
        4. Creates connection lists between nodes
        '''
        
        #Input
        self.Pattern = unit_cell
        self.unit_cell = unit_cell.pattern
        self.size_lon = unit_cell.size_lon 
        self.size_tgn = unit_cell.size_tgn                  
        self.radius = radius
        self.height = height
        self.tcopy = tcopy
        self.hcopy = hcopy
        self.strut_radius = strut_radius
        self.centerline = centerline
        
        #Cap
        self.bot_cap = unit_cell.bot_cap
        self.top_cap = unit_cell.top_cap
        self.bot_cap_size = unit_cell.bot_cap_size
        self.top_cap_size = unit_cell.top_cap_size
        
        #Params
        # Represent number of layer in longitudinal direction
        # Using (+ 1) in the beginning due to base layer
        self.layers = 1 + hcopy*(self.size_lon-1) + self.bot_cap_size + self.top_cap_size
        # Represent number of nodes in one circular layer
        # Using size_tgn-1 because, it is circular. 
        # For example if the tangential size of a unit cell is 3, then nodes per layer calculation is like this:
        # 3 + (3-1) + (3-1) + .... (3-2) = (3*(n-1)) -> every cell only needs 3-1 node except first and last cell. 
        # Last cell is like another cell (not first), but have additional -1 due to connection with first cell. 
        self.nodes_per_layer = self.tcopy*(self.size_tgn-1)
        # Distance between each layer in longitudinal direction
        self.layer_height = self.height/(self.layers-1)
        # Distance between each node in tangential direction (represented by degree in radian)
        self.sep_angle = 2*np.pi/self.nodes_per_layer
        # Additional rotational offset for pattern alignment
        self.offset_angle=offset_angle
        # print("offset ang:",offset_angle)
        
        #Initial mesh
        self.mesh = self.pattern_wrap(radius, centerline)
        """
        This processes the line connectivity data into a more usable format:
            self.mesh.lines: The raw connectivity array in VTK format
            Format: [n, pt1, pt2, n, pt3, pt4, ...] where n is number of points in each segment (always 2 for lines)
            .reshape(-1,3): Reshapes into N×3 array where each row is [2, index1, index2]
            [:,1:]: Drops the first column (the 2s), leaving just [index1, index2] pairs

        Result:
            A N×2 numpy array where each row represents one connection between nodes
            Example: [[0, 1], [1, 2], ...] means node 0 connects to 1, 1 connects to 2, etc.

        Purpose:
            Provides a cleaner representation of connections for later processing
            Used in strut generation and connectivity analysis
        """
        self.lines = self.mesh.lines.reshape(-1,3)[:,1:]
        """
        This builds an adjacency list representing the stent's node connections:
        
        Calls connected_list() which:
            For each node, finds all directly connected neighbors
            Returns a list where connected[i] contains indices of nodes connected to node i
        Implementation typically uses self.lines to build these relationships

        Example output:
        [
            [1, 5],    # Node 0 connects to 1 and 5
            [0, 2],    # Node 1 connects to 0 and 2
            [1, 3],    # Node 2 connects to 1 and 3
            ...
        ]
        """
        self.connected = self.connected_list()


    def cylinder_mesh(self, R, centerline):
        
        '''
        Creates a dense cylindrical surface mesh
        '''
        
        #Params
        Nz = self.layers          # Total number of layers (circular rings)
        N = self.nodes_per_layer  # Nodes per layer
        h = self.layer_height     # Vertical distance between layers
        sep_angle = self.sep_angle  # Angular spacing between nodes
        offset_angle = self.offset_angle*np.pi  # Rotational offset in radians
        
        #Unit layer
        circ_nodes = np.zeros((N,3))
        for i in range(N):
            # Creates a reference circle of nodes in the XY plane
            # Nodes are equally spaced with angular separation sep_angle
            # offset_angle rotates the entire pattern if needed
            circ_nodes[i] = R*np.array([np.cos(i*sep_angle+offset_angle),np.sin(i*sep_angle+offset_angle),0])
        
        # Straight Cylinder Mode (no centerline):
        if not centerline:
            #Layer displacement vector
            dz = np.zeros((N,3))
            dz[:,2] = h*np.ones(N) # Z-axis displacement vector
            
            #Generate positional nodes
            nodes = circ_nodes.copy()
            for i in range(1,Nz):
                # Stacks copies of the base ring vertically
                # Each new layer is offset by h in the Z-direction
                nodes = np.append(nodes,circ_nodes+i*dz,axis=0)
        
        # Curved Centerline Mode:
        else:
            #Centerline definition
            c = centerline.interp
            # Samples the centerline at Nz points
            t = np.linspace(c.start()[0],c.end()[0],Nz)
            # Get spline positions
            spline_points = c.evaluate(t)
            # Get spline directions (tangent vector)
            tangents = c.tangent(t)        
            
            #Place layers along centerline
            nodes = np.array([[0,0,0]])
            for i in range(0,Nz):
                # Uses rotate_layer to align the base ring with the tangent
                layer = rotate_layer(spline_points[i],
                                     tangents[i],
                                     circ_nodes)
                # Positions the ring at the spline point
                nodes = np.append(nodes,layer,axis=0)
            nodes = nodes[1:]
        
        # Creates quadrilateral faces between adjacent rings
        # Each quad connects 4 nodes (current + next node in both rings)
        faces = np.array([])
        for i in range(Nz-1):
            for j in range(N):
                f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
                faces = np.append(faces,f)
            
        #faces = faces.astype('int')
        mesh = pv.PolyData(nodes)#,faces)
        
        return mesh
    
    def pattern_wrap(self, R, centerline):
        
        '''
        Wraps a pattern to the side of a dense cylinder surface mesh
        
        1. Generates base cylinder mesh
        2. Processes three sections:
            2.1. Top cap (if exists)
            2.2. Main pattern (repeated along length)
            2.3. Bottom cap (if exists)
        3. Connects nodes according to the pattern's line definitions
        4. Returns a clean wireframe structure
        '''
        
        #Params
        # Number of main layer in longitudinal direction. Not including layer for bot_cap and top_cap.
        Nz = self.layers - self.bot_cap_size - self.top_cap_size
        #Nl = self.unit_cell_size 
        # Number of node in longitudinal direction, in a unit cell.
        Ni = self.size_lon
        # Number of node in tangential direction, in a unit cell.
        Nj = self.size_tgn
        # Nomber of node in one layer
        N = self.nodes_per_layer
        # Generates the 3D wireframe by wrapping the 2D pattern around a cylinder
        mesh = self.cylinder_mesh(R, centerline)
        
        lines = np.array([])
        
        #Top cap
        if self.top_cap.any():
            i = 0
            # Repeats the cap pattern around the circumference (every Nj-1 nodes)
            for j in range(0,N,Nj-1):
                cell_lines = self.top_cap.copy()
                cell_lines += np.array([i,j]) # Offset pattern
                # k represent the line's index
                for k in range(len(cell_lines)):
                    p0 = cell_lines[k,0]
                    p1 = cell_lines[k,1]
                    # Calculates 3D node indices from 2D pattern coordinates
                    # Uses modular arithmetic (%N) for circular connectivity
                    ind0 = p0[0]*N+(p0[1]%N)
                    ind1 = p1[0]*N+(p1[1]%N)
                    lines = np.append(lines,[2,ind0,ind1])
            lines = lines.astype('int')
            
        
        #Shift unit cells, if there is top cap
        start = (self.top_cap_size-1)*self.top_cap.any()
        for i in range(start,Nz-1,Ni-1): # Step by unit cell height
            for j in range(0,N,Nj-1): # Step around circumference
                cell_lines = self.unit_cell.copy()
                cell_lines += np.array([i,j]) # Offset pattern
                for k in range(len(cell_lines)):
                    p0 = cell_lines[k,0]
                    p1 = cell_lines[k,1]
                    # Calculates 3D node indices from 2D pattern coordinates
                    # Uses modular arithmetic (%N) for circular connectivity
                    ind0 = p0[0]*N+(p0[1]%N)
                    ind1 = p1[0]*N+(p1[1]%N)
                    lines = np.append(lines,[2,ind0,ind1])
        lines = lines.astype('int')
        
        #Bottom Cap
        if self.bot_cap.any():
            i += Ni-1 # Position after last main pattern cell
            # Repeats the cap pattern around the circumference (every Nj-1 nodes)
            for j in range(0,N,Nj-1):
                cell_lines = self.bot_cap.copy()
                cell_lines += np.array([i,j]) # Offset pattern
                # k represent the line's index
                for k in range(len(cell_lines)):
                    p0 = cell_lines[k,0]
                    p1 = cell_lines[k,1]
                    # Calculates 3D node indices from 2D pattern coordinates
                    # Uses modular arithmetic (%N) for circular connectivity
                    ind0 = p0[0]*N+(p0[1]%N)
                    ind1 = p1[0]*N+(p1[1]%N)
                    lines = np.append(lines,[2,ind0,ind1])
            lines = lines.astype('int')
        
        
        #Surface and wire structure construction
        edges = pv.PolyData()
        edges.points = mesh.points  # All 3D nodes
        edges.lines = lines  # Connectivity data
        
        edges = edges.clean() #All points have at least one connection
        
        return edges
    
    
    def show(self, cpos=[1,0,0]):
        
        '''
        Show current stent configuration
        '''
        
        p = pv.Plotter()
        p.add_mesh(self.mesh, color='black', line_width=2)
        p.show(cpos=cpos)
        
    def connected_nodes(self,idx):
        
        '''
        Returns array of node connections
        '''
        
        #Connected ids
        cids = [i for i, line in enumerate(self.lines) if idx in line]
        
        #Connected nodes
        connected = np.unique([self.lines[i].ravel() for i in cids])
        
        return np.delete(connected, np.argwhere(connected == idx))
    
    def connected_list(self):
        
        lst = []
        for i in range(len(self.mesh.points)):
            lst.append([p for p in self.connected_nodes(i)])
            
        return lst
            
    
    def save(self, fname):
        
        '''
        Save current mesh configuration
        '''
        

        self.mesh.save(fname)
        
        return None
        
    
    def render_strut(self, n=3, h=1.2, threshold=2, save_as=None):
        
        '''
        Renders strut mesh using wire inflation algorithm
        '''
        
        r = self.strut_radius
        

        node_mesh = pv.PolyData([])
        line_mesh = pv.PolyData([])
        
        polygon = np.array([[r*np.cos(i*2*np.pi/n), r*np.sin(i*2*np.pi/n), 0] for i in range(n)])
        polygon = np.append(np.array([[0,0,0.1*r]]),polygon,axis=0)
        
        for idx in range(len(self.mesh.points)):
            
            pref = self.mesh.points[idx]
            cids = self.connected[idx]
            
            cloud = np.zeros((1,3))
            
            subt = []
            for cid in cids:
                t = self.mesh.points[cid]-pref
                t /= np.linalg.norm(t)
                vertices = rotate_layer(pref+h*r*t, t, polygon)
                cloud = np.append(cloud,vertices,axis=0)
                cone = pv.Cone(center = pref+h*r*t, direction=-t, height = 2*h*r, radius=2*r, resolution=n)
                subt.append(cone)
            
            cloud = cloud[1:]
            hull = ConvexHull(cloud)
            faces = hull.simplices
            faces = np.append(3*np.ones((faces.shape[0],1),'int'),faces,axis=1).ravel()
            
            add = pv.PolyData()
            add.points = cloud
            add.faces = faces
            
            for surf in subt:
                add = add.clip_surface(surf, invert=False)
            
            node_mesh += add
        
        polygon = np.array([[r*np.cos(i*2*np.pi/n), r*np.sin(i*2*np.pi/n), 0] for i in range(n)])
        polygon = np.append(np.array([[0,0,-0.1*r]]),polygon,axis=0)
        
        for line in self.lines:
        
            pref = [self.mesh.points[line[0]], self.mesh.points[line[1]]]
                
            cloud = np.zeros((1,3))
            subt = []
            for i in range(2):
                t = pref[i-1] - pref[i]
                t /= np.linalg.norm(t)
                vertices = rotate_layer(pref[i]+h*r*t, t, polygon)
                cloud = np.append(cloud,vertices,axis=0)
                cone = pv.Cone(center = pref[i]+h*r*t, direction=t, height = 2*h*r, radius=2*r, resolution=n)
                subt.append(cone)
                
            cloud = cloud[1:]
            hull = ConvexHull(cloud)
            faces = hull.simplices
            faces = np.append(3*np.ones((faces.shape[0],1),'int'),faces,axis=1).ravel()
        
            add = pv.PolyData()
            add.points = cloud
            add.faces = faces
            
            for surf in subt:
                add = add.clip_surface(surf, invert=False)
            
            line_mesh += add
        
        strut = pv.PolyData(node_mesh+line_mesh)
        
        # areas = strut.compute_cell_sizes(length=False,volume=False).cell_arrays['Area']
        areas = strut.compute_cell_sizes(length=False,volume=False).cell_data['Area']
        hist,bins = np.histogram(areas,bins=100)
        faces = strut.faces.reshape(-1,4)[:,1:]
        
        delete_cells = [i for i in range(len(faces)) if areas[i] < bins[threshold]]
        faces = np.delete(faces,delete_cells,axis=0)
        strut.faces = np.append(3*np.ones((len(faces),1),dtype='int'),faces,axis=1).ravel()
        
        if save_as:
            strut.save(save_as)
        
        
        return strut
        

class Pattern:
    '''
    The Pattern class stores geometric configurations for stents and their end caps.
    '''
    
    def __init__(self, pattern=np.array([]), bot_cap=np.array([]), top_cap=np.array([])):
        '''
        pattern: A 3D numpy array representing line segments of the stent's main structure.

        Shape of the pattern: (n, 2, 2), where each entry is a line segment defined by two 2D points: [[[x1, y1], [x2, y2]], ...].

        bot_cap/top_cap: Arrays for the bottom/top end caps of the stent (similar structure to pattern).
        '''
        
        self.pattern = pattern
        self.bot_cap = bot_cap
        self.top_cap = top_cap
        
        if pattern.any():
            # size_lon/size_tgn: Longitudinal/tangential (circumferential) dimensions of the stent pattern.
            # Add 1 becasue pattern[:,:,0].max() returning a max index starting from 0.
            self.size_lon = 1 + pattern[:,:,0].max()
            self.size_tgn = 1 + pattern[:,:,1].max()
        
        if bot_cap.any(): 
            # bot_cap_size/top_cap_size: Dimensions of the end caps.
            # Add 1 becasue pattern[:,:,0].max() returning a max index starting from 0.
            self.bot_cap_size = 1 + bot_cap[:,:,0].max()
        else:
            self.bot_cap_size = 0
            
        if top_cap.any():
            # bot_cap_size/top_cap_size: Dimensions of the end caps.
            # Add 1 becasue pattern[:,:,0].max() returning a max index starting from 0.
            self.top_cap_size = 1 + top_cap[:,:,0].max()
        else:
            self.top_cap_size = 0
        

class VascCenterline:
    
    def __init__(self, points, init_range=np.array([]), 
                 point_spacing = 5, reverse = False):
        
        
        self.centerline_full = self.points2lines(points)
        
        if init_range:
            points = points[init_range[0]:init_range[1]+1]
        
        
        self.interp = self.interp_cl(points, point_spacing, reverse)
        
        
        self.init_segment = self.points2lines(points)
        
        
        
    def interp_cl(self, points, point_spacing, reverse):
        
        
        if reverse:
            points = points[::-1]
        
        points = points[::point_spacing]

        return cubic_curve(points)
    
    def points2lines(self, points):
    
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
        cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
        poly.lines = cells
        
        return poly


class VirtualStenting:
    
    def __init__(self, stent=None, centerline=None, boundary=None,
                 initial_stent = None, target_stent = None, crimping = 0.2):
        
        if stent:
            self.stent = stent
            self.result = stent
        
        if centerline:
            self.centerline = centerline
        
        if boundary:
            self.boundary = boundary
        
        if initial_stent:
            self.initial_stent = initial_stent
        else:
            self.initial_stent = self.initial(stent, centerline, crimping)
        
        if target_stent:
            self.target_stent = target_stent
        else:
            self.target_stent = self.initial(stent, centerline, 1)
        
    
    def initial(self, stent, c, crimping):
        
        r = stent.radius
        initial_stent = FlowDiverter(stent.Pattern, r*crimping, stent.height,
                                     stent.tcopy, stent.hcopy, centerline=c,
                                     strut_radius=stent.strut_radius, offset_angle=stent.offset_angle)
        
        return initial_stent
        
    
    def deploy(self, tol = 1e-5, add_tol=0, step = None, fstop = 1, 
               max_iter = 300, alpha = 1, verbose:bool = True, OC:bool = True, 
               render_gif:bool=False, deployment_name:str=""):
        
        #Parameters
        Nz = self.stent.layers
        N = len(self.stent.mesh.points)/Nz
        #Result holder
        result_mesh = pv.PolyData()
        result_mesh.points = self.initial_stent.mesh.points
        result_mesh.lines = self.initial_stent.mesh.lines

        #Initial frame for gif rendering:
        if render_gif:
            frame(init_mesh=result_mesh,init=True,fname="Deploy.gif" if deployment_name=="" else deployment_name)

        
        con_tol = self.stent.strut_radius + add_tol
        
        #Nearest neighbor distance
        tree = KDTree(self.boundary.points)
        def proximity(point):
            d,idx = tree.query(point)
            return d
        
        #Connected points
        connected = self.stent.connected
        #Position initialization
        p_ref = np.array(self.target_stent.mesh.points) #Reference position
        p = np.array(result_mesh.points) #Current positions
        p_new = p.copy() #New positions 
        p_pred = p.copy() #Predicted positions
        p_prev = p.copy() #Previous positions
        
        if step:
            layers = np.arange(step,int(fstop*Nz)+step,step)
            if fstop == 1:
                layers = np.append(layers,Nz)
        
        else:
            layers = [int(fstop*Nz)]

        for l in layers:
            
            err = np.ones(int(l*N)) #List of errors
            Niter = 0 #Number of iterations done
            
            while max(err)>tol and Niter<max_iter:
                for i in range(int(l*N)):

                    F = 0
                    kt = 0
                    for j in connected[i]:
                        k = 1/np.linalg.norm(p_ref[j]-p_ref[i])
                        F += k*((p[j]-p[i])-(p_ref[j]-p_ref[i]))
                        kt += k
                    
                    if OC:
                        F *= proximity(p[i])/proximity(p_prev[i])
                    
                    p_pred[i] = p[i] + alpha*F/kt

                    if proximity(p_pred[i])>con_tol:
                        p_new[i] = p_pred[i]

                    err[i] = np.linalg.norm(p_new[i]-p[i])
                    p_prev = p.copy()
                    p = p_new.copy()
                    
                Niter += 1
                
            if verbose:
                print(l,Niter,max(err))

            if render_gif:
                result_mesh.points=p
                frame(mesh=result_mesh)
                
        # end frame of gif rendering
        if render_gif:
            frame(end=True)
        else:    
            result_mesh.points = p
            
        self.result.mesh = result_mesh
        
        return self.result    

'''
Some Stent Patterns
'''

#Helical stents: PED/Silk
def helical(size):
    
    unit_cell_lines = np.array([[[0,0],[0,0]]])
    for i in range(size-1):
        # First loop: Adds lines like (i, i) → (i+1, i+1) (primary diagonal).
        unit_cell_lines = np.append(unit_cell_lines,
                                    [[[i,i],[i+1,i+1]]],axis=0)
        
        # Second loop: Adds lines like (i, size-1-i) → (i+1, size-i-2) (anti-diagonal).
        unit_cell_lines = np.append(unit_cell_lines,
                                    [[[i,size-1-i],[i+1,size-i-2]]],axis=0)
    # # Example of size 3
    # [(0,0)-(0,0)] -> Initialization
    # [(0,0)-(1,1)], -> Primary-diagonal
    # [(0,2)-(1,1)], -> Anti-diagonal 
    # [(1,1)-(2,2)], -> Primary-diagonal
    # [(1,1)-(2,0)] -> Anti-diagonal
    return Pattern(pattern=unit_cell_lines)
                


#Tilted Rectangular
def semienterprise():
    
    pattern_lines = np.array([
                                [[0,0],[1,1]],    # Line from (0,0) to (1,1)
                                [[2,0],[1,1]],    # Line from (2,0) to (1,1)
                                [[1,1],[0,2]]     # Line from (1,1) to (0,2)
                            ]) 
    
    pattern_cap = np.array([[[0,0],[1,1]],
                            [[1,1],[0,2]]])
    
    return Pattern(pattern=pattern_lines, bot_cap=pattern_cap)

#Enterprise
def enterprise(N=1):
    
    if N != 1 and N != 2:
        return enterprise(2)
    
    elif N == 1:
        # 6 pattern lines forming a braided structure.
        pattern_lines = np.array([[[0,0],[1,1]],
                                  [[1,1],[0,2]],
                                  [[0,2],[1,3]],
                                  [[2,0],[1,1]],
                                  [[2,2],[1,3]],
                                  [[1,3],[2,4]]])
        
        bot_cap = np.array([[[0,0],[1,1]],
                                [[1,1],[0,2]]])
        
        top_cap = np.array([[[1,2],[0,3]],
                            [[0,3],[1,4]]])
    
    else:
        # 12 pattern lines with additional layers for larger stents.
        # Includes both bottom and top caps for anchoring.
        pattern_lines = np.array([[[0,0],[1,1]],
                                  [[1,1],[2,2]],
                                  [[2,2],[1,3]],
                                  [[1,3],[0,4]],
                                  [[0,4],[1,5]],
                                  [[1,5],[2,6]],
                                  [[4,0],[3,1]],
                                  [[3,1],[2,2]],
                                  [[4,4],[3,5]],
                                  [[3,5],[2,6]],
                                  [[2,6],[3,7]],
                                  [[3,7],[4,8]]])
        
        bot_cap = np.array([[[0,0],[1,1]],
                            [[1,1],[2,2]],
                            [[2,2],[1,3]],
                            [[1,3],[0,4]]])
        
        top_cap = np.array([[[2,4],[1,5]],
                            [[1,5],[0,6]],
                            [[0,6],[1,7]],
                            [[1,7],[2,8]]])
        
    return Pattern(pattern=pattern_lines, bot_cap=bot_cap, top_cap=top_cap)


#Honeycomb
def honeycomb():
    # Generates a hexagonal (honeycomb-like) pattern, known for mechanical stability.
    # Forms interconnected hexagons.
    pattern_lines = np.array([[[2,0],[0,1]],
                              [[0,1],[0,2]],
                              [[0,2],[2,3]],
                              [[2,3],[2,4]],
                              [[2,3],[4,2]],
                              [[4,1],[2,0]]])
    
    pattern_cap = np.array([[[0,1],[0,2]]])
    
    return Pattern(pattern=pattern_lines, bot_cap=pattern_cap)

'''GIF Generating Functions'''

plotter = pv.Plotter(off_screen=True)
def frame(init_mesh=None, mesh=None, init=False, end=False, ztrans = 0,fname='Deploy.gif'):
    if init:
        plotter.open_gif(fname)
        plotter.add_mesh(init_mesh,color='b',opacity=0.1)
        # plotter.show(cpos = [-1,1,0.5],auto_close=False)
    else:
        actor = plotter.add_mesh(mesh)
        plotter.write_frame()
        plotter.remove_actor(actor)
    if end:
        plotter.close()
    return 0
