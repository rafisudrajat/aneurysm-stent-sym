import numpy as np
import pyvista as pv
import splipy.curve_factory as sp
from PyStenting import rotate_layer

def points2lines(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

'''Frequently Used Boundaries'''

def cylinder_bound(R=1,height=10,hstent=10,res_ang=100,res_lon=100, 
                   origin=np.zeros(3), direction=np.array([0,0,1]),
                   get_inlet_outlet=False):
    
    #Parametes
    Nz = res_lon
    N = res_ang
    sep_angle = 2*np.pi/N
    
    #Centerline
    t = np.linspace(0,1,res_lon)
    z = height*t
    y = np.zeros(len(t))
    x = np.zeros(len(t))
    
    points = np.append(x.reshape(len(x),1),
                       y.reshape(len(y),1), axis=1)
    points = np.append(points,z.reshape(len(z),1),axis=1)
    
    if hstent != height:
        
        start = int(0.5*res_lon*(1-hstent/height))
        stop = int(0.5*res_lon*(1+hstent/height))
        
        stent_centerline = points[start:stop+1]
        
    else:
        stent_centerline = points.copy()
    
    
    #Unit layer
    circ_nodes = np.zeros((N,3)) 
    for i in range(N):
        circ_nodes[i] = R*np.array([np.sin(i*sep_angle),np.cos(i*sep_angle),0])
    circ_nodes = rotate_layer(origin,direction,circ_nodes)
    
    
    #Layer displacement vector
    dz = height*direction/(Nz-1)
    dz = np.array([dz for i in range(N)])
    
    #Generate positional nodes
    nodes = circ_nodes.copy()
    for i in range(1,Nz):
        nodes = np.append(nodes,circ_nodes-i*dz,axis=0)
    
    nodes += height*direction

    # Get inlet and outlet
    if get_inlet_outlet:
        inletPoints=nodes[:res_ang]
        outletPoints=nodes[len(nodes)-res_ang:]
        facesInletOutlet=np.array([i for i in range(res_ang)])
        facesInletOutlet=np.insert(facesInletOutlet, 0, res_ang)
        facesInletOutlet=facesInletOutlet.astype('int')
        print("inletPoints",inletPoints)
        print("facesInletOutlet",inletPoints)
        print("outletPoints",inletPoints)
        meshInlet=pv.PolyData(inletPoints,facesInletOutlet)
        meshOutlet=pv.PolyData(outletPoints,facesInletOutlet)
        dict_inlet_outlet={"inlet":meshInlet,"outlet":meshOutlet}
    
    faces = np.array([])
    for i in range(Nz-1):
        for j in range(N):
            f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
            faces = np.append(faces,f)
        
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes,faces)
    
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()

    if get_inlet_outlet:
        return mesh,stent_centerline,dict_inlet_outlet
    return mesh, stent_centerline

def conical_boundary(Rbottom,Rtop,height=10,hstent=10,
                     res_lon=100,res_ang=100,origin=np.zeros(3)):
    '''Create conical boundary mesh'''
    direction = np.array([0,0,1])
    
    #Parametes
    Nz = res_lon
    N = res_ang
    sep_angle = 2*np.pi/N
    
    #Centerline
    t = np.linspace(0,1,100)
    z = height*(1-t)
    y = np.zeros(len(t))
    x = np.zeros(len(t))
    
    points = np.append(x.reshape(len(x),1),
                       y.reshape(len(y),1), axis=1)
    points = np.append(points,z.reshape(len(z),1),axis=1)
    
    if hstent != height:
        
        start = int(50*(1-hstent/height))
        stop = int(50*(1+hstent/height))
        stent_centerline = points[start:stop+1]
        
    else:
        stent_centerline = points.copy()
    
    
    #Unit layer
    circ_nodes = np.zeros((N,3)) 
    for i in range(N):
        circ_nodes[i] = Rtop*np.array([np.sin(i*sep_angle),np.cos(i*sep_angle),0])
    circ_nodes = rotate_layer(origin,direction,circ_nodes)
    
    #Radius change
    Rf = np.linspace(1,Rbottom/Rtop,Nz)
    
    #Layer displacement vector
    dz = height*direction/(Nz-1)
    dz = np.array([dz for i in range(N)])
    
    #Generate positional nodes
    nodes = circ_nodes.copy()
    for i in range(1,Nz):
        nodes = np.append(nodes,Rf[i]*circ_nodes-i*dz,axis=0)
    
    nodes += height*direction
    
    faces = np.array([])
    for i in range(Nz-1):
        for j in range(N):
            f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
            faces = np.append(faces,f)
        
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes,faces)
    
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()
    
    return mesh, stent_centerline

def bent_tube(r,angle,h=10,hstent=10,res_ang=100,res_lon=100,get_inlet_outlet=False):
    
    #Parametes
    Nz = res_lon
    N = res_ang
    sep_angle = 2*np.pi/N
    R = h/angle
    
    #Centerline
    t = np.linspace(0,angle,5)
    y = R*(1-np.sin(t))
    z = R*np.cos(t)
    x = np.zeros(len(t))
    
    points = np.append(x.reshape(len(x),1),
                       y.reshape(len(y),1), axis=1)
    points = np.append(points,z.reshape(len(z),1),axis=1)
    
    if hstent != h:
        
        start = int(50*(1-hstent/h))
        stop = int(50*(1+hstent/h))
        
        stent_centerline = points[start:stop+1]
        
    else:
        stent_centerline = points.copy()
    
    centerline = sp.cubic_curve(points)

    t = np.linspace(centerline.start()[0],centerline.end()[0],Nz)
    spline_points = centerline.evaluate(t)
    tangents = centerline.tangent(t)
    
    #Unit layer
    circ_nodes = np.zeros((N,3)) 
    for i in range(N):
        circ_nodes[i] = r*np.array([np.sin(i*sep_angle),np.cos(i*sep_angle),0])
    
    #Place layers long centerline
    nodes = np.array([[0,0,0]])
    for i in range(0,Nz):
        layer = rotate_layer(spline_points[i],
                             tangents[i],
                             circ_nodes)
        nodes = np.append(nodes,layer,axis=0)
    nodes = nodes[1:]
    
    # Get inlet and outlet
    if get_inlet_outlet:
        inletPoints=nodes[:res_ang]
        outletPoints=nodes[len(nodes)-res_ang:]
        facesInletOutlet=np.array([i for i in range(res_ang)])
        facesInletOutlet=np.insert(facesInletOutlet, 0, res_ang)
        facesInletOutlet=facesInletOutlet.astype('int')

        meshInlet=pv.PolyData(inletPoints,facesInletOutlet)
        meshOutlet=pv.PolyData(outletPoints,facesInletOutlet)
        dict_inlet_outlet={"inlet":meshInlet,"outlet":meshOutlet}

    faces = np.array([])
    for i in range(Nz-1):
        for j in range(N):
            f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
            faces = np.append(faces,f)
        
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes,faces)
    
    if get_inlet_outlet:
        return mesh,stent_centerline,dict_inlet_outlet
    return mesh, stent_centerline

def s_curve(A,r,height,hstent=10,res_ang=100,res_lon=100):
    
    #Parametes
    Nz = res_lon
    N = res_ang
    sep_angle = 2*np.pi/N
    
    #Centerline
    t = np.linspace(0,1,Nz)
    z = height*(1-t)
    y = -A*np.sin(2*np.pi*t)
    x = np.zeros(Nz)
    
    points = np.append(x.reshape(len(x),1),
                       y.reshape(len(y),1), axis=1)
    points = np.append(points,z.reshape(len(z),1),axis=1)
    
    if hstent != height:
        
        start = int(50*(1-hstent/height))
        stop = int(50*(1+hstent/height))
        
        stent_centerline = points[start:stop+1]
        
    else:
        stent_centerline = points.copy()
    
    centerline = sp.cubic_curve(points)

    t = np.linspace(centerline.start()[0],centerline.end()[0],Nz)
    spline_points = centerline.evaluate(t)
    tangents = centerline.tangent(t)
    
    #Unit layer
    circ_nodes = np.zeros((N,3)) 
    for i in range(N):
        circ_nodes[i] = r*np.array([np.sin(i*sep_angle),np.cos(i*sep_angle),0])
    
    #Place layers long centerline
    nodes = np.array([[0,0,0]])
    for i in range(0,Nz):
        layer = rotate_layer(spline_points[i],
                             tangents[i],
                             circ_nodes)
        nodes = np.append(nodes,layer,axis=0)
    nodes = nodes[1:]
    
    faces = np.array([])
    for i in range(Nz-1):
        for j in range(N):
            f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
            faces = np.append(faces,f)
        
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes,faces)
    
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()
    
    return mesh, stent_centerline

def rugged_cylinder(R,maxVar=0.1,height=10,hstent=10, seed=1, Nsmooth=500,
                    Nsubdiv=1, res_lon=100,res_ang=100,origin=np.zeros(3)):

    direction = np.array([0,0,1])
    
    #Parametes
    Nz = res_lon
    N = res_ang
    sep_angle = 2*np.pi/N
    
    #Centerline
    t = np.linspace(0,1,100)
    z = height*(1-t)
    y = np.zeros(len(t))
    x = np.zeros(len(t))
    
    points = np.append(x.reshape(len(x),1),
                       y.reshape(len(y),1), axis=1)
    points = np.append(points,z.reshape(len(z),1),axis=1)
    
    if hstent != height:
        
        start = int(50*(1-hstent/height))
        stop = int(50*(1+hstent/height))
        
        stent_centerline = points[start:stop+1]
        
    else:
        stent_centerline = points.copy()
    
    
    #Unit layer
    circ_nodes = np.zeros((N,3)) 
    for i in range(N):
        circ_nodes[i] = R*np.array([np.sin(i*sep_angle),np.cos(i*sep_angle),0])
    circ_nodes = rotate_layer(origin,direction,circ_nodes)
    
    #Radius change
    rng = np.random.default_rng(seed)
    Rf = 1-maxVar*(1-2*rng.random(Nz))
    
    
    #Layer displacement vector
    dz = height*direction/(Nz-1)
    dz = np.array([dz for i in range(N)])
    
    #Generate positional nodes
    nodes = circ_nodes.copy()
    for i in range(1,Nz):
        nodes = np.append(nodes,Rf[i]*circ_nodes-i*dz,axis=0)
    
    nodes += height*direction
    
    faces = np.array([])
    for i in range(Nz-1):
        for j in range(N):
            f = np.array([4,i*N+j,i*N+(j+1)%N,(i+1)*N+(j+1)%N,(i+1)*N+j])
            faces = np.append(faces,f)
        
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes,faces)
    
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()
    
    mesh = mesh.subdivide(Nsubdiv, subfilter='loop')
    mesh = mesh.smooth(n_iter=Nsmooth)
    
    return mesh, stent_centerline

'''Simple Aneurysm'''

def aneu_geom(r=1, h=20, hstent=20, angle=0, aneu_rad= 1.5, aneu_pos=0.5, overlap=0.25,
              cyl_res=100, sph_res=50, extension_ratio=0, ext_res=20,get_inlet_outlet=False):
    
    dict_inlet_outlet=None
    if angle:
        if get_inlet_outlet:
            vessel, centerline_points, dict_inlet_outlet = bent_tube(r, angle, h=h, hstent=h,res_ang=cyl_res, res_lon=cyl_res,get_inlet_outlet=get_inlet_outlet)
        else:
            vessel, centerline_points = bent_tube(r, angle, h=h, hstent=h,res_ang=cyl_res, res_lon=cyl_res)
        
    else:
        if get_inlet_outlet:
            vessel, centerline_points,dict_inlet_outlet = cylinder_bound(r, h, hstent=h, res_ang=cyl_res, res_lon=cyl_res,get_inlet_outlet=get_inlet_outlet)
        else:
            vessel, centerline_points = cylinder_bound(r, h, hstent=h, res_ang=cyl_res, res_lon=cyl_res)
    
    centerline = sp.cubic_curve(centerline_points)
    t = np.linspace(centerline.start()[0], centerline.end()[0], 1000)
    centerline_points = centerline.evaluate(t)
    if hstent != h:
        
        start = int(500*(1-hstent/h))
        stop = int(500*(1+hstent/h))
        
        stent_centerline = centerline_points[start:stop+1]
        
    else:
        stent_centerline = centerline_points.copy()
    
    
    if extension_ratio:
        h = extension_ratio*2*r
    
        org = centerline_points[0]
        tg = -centerline.tangent(t[0])
        inlet, _ = cylinder_bound(r, h, res_ang=cyl_res, res_lon=ext_res,
                                  origin=org, direction=tg)

        org = centerline_points[-1]
        tg = centerline.tangent(t[-1])
        outlet, _ = cylinder_bound(r, h, res_ang=cyl_res, res_lon=ext_res,
                                   origin=org, direction=tg)
    '''
    ta = int(t[0] + aneu_pos*(t[-1]-t[0]))
    tg = centerline.tangent(ta)
    d = np.array([0,-1,tg[1]/tg[2]])
    d *= (r+aneu_rad-overlap)/np.linalg.norm(d)
    
    aneu_center = centerline.evaluate(ta) + d
    '''
    
    d = np.array([0,-np.cos(angle*aneu_pos),np.sin(angle*aneu_pos)])
    d *= (r+aneu_rad-overlap)/np.linalg.norm(d)
    aneu_center = stent_centerline[int(len(stent_centerline)*aneu_pos)] + d
    sacc = pv.Sphere(radius=aneu_rad, center=aneu_center, 
                     direction=d/np.linalg.norm(d),
                     theta_resolution=sph_res, phi_resolution=sph_res)
    
    if aneu_rad:
        geom = vessel.clip_surface(sacc, invert=False) + sacc.clip_surface(vessel, invert=bool(angle))
    else:
        geom=vessel
    
    geom = geom.clean()  
    geom = geom.triangulate()
    geom = geom.clean()
    
    
    # return geom, stent_centerline
    return {
        "geom":geom,
        "stent_centerline":stent_centerline,
        "inlet": None if dict_inlet_outlet==None else dict_inlet_outlet["inlet"],
        "outlet":None if dict_inlet_outlet==None else dict_inlet_outlet["outlet"]
    }
    
'''
Flow Extension

def extend_flow(surface, centerline_points, extension_ratio=10, inlet=True, outlet=True):
    init_points = surface.points
    xmax = max(init_points[:,0])
    ymax = max(init_points[:,1])
    zmax = max(init_points[:,2])
    size = max([abs(xmax-ymax),abs(xmax-zmax),abs(ymax-zmax)])
    
    def extend(suface,idx,reverse=False):
        origin = centerline_points[idx]
        tangent = centerline_points[idx+1]-centerline_points[idx]
        tangent *= (1-2*reverse)/np.linalg.norm(tangent)
        center = origin+tangent*size/2
        new_surface = surface-pv.Cylinder(center=center, direction=tangent,
                                          radius=size, height=size)
        tree = KDTree(new_surface.points)
        r, _ = tree.query(origin)
        new_surface = new_surface + 
'''    




    