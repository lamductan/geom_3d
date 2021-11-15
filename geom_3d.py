###############################################################################
# A very minimal library for manipulating 3D coordinate system                #
# Methods in this file work in normal cases, without proper error handling in #
# edge cases                                                                  #
# Refer to file 'test_geom_3d.py' to know how to use classes and methods in   #
# this file                                                                   #
# Author: Tan Lam                                                             #
###############################################################################

import copy
import numpy as np
import scipy
import open3d as o3d
import plotly.graph_objects as go
import sklearn

EPSILON = 1e-9

def _approx(a, b):
    if isinstance(a, Point3D) and isinstance(b, Point3D):
        return _approx(a.arr, b.arr)
    res = (np.abs(a - b) <= EPSILON)
    if isinstance(res, np.ndarray):
        return res.all()
    return res

def compute_norm(u: np.ndarray):
    if isinstance(u, Point2D) or isinstance(u, Vector2D):
        u = u.arr
    elif isinstance(u, Point3D) or isinstance(u, Vector3D):
        u = u.arr
    return np.sqrt(np.sum(u**2))


class Point2D:    
    def __init__(self, arr: np.ndarray):
        self.x, self.y = arr
        self.arr = arr

    def get_np_array(self) -> np.ndarray:
        return self.arr.copy()

    def __str__(self) -> str:
        return "Point3D({0}, {1})".format(self.x, self.y)

    def __getitem__(self, idx):
        return self.arr[idx]

    def approx(self, p2) -> bool:
        arr = p2 if not isinstance(p2, Point2D) else p2.arr
        return _approx(self.arr, arr)

    def __eq__(self, p2) -> bool:
        return self.approx(p2)

    def __add__(self, p2):
        arr = p2 if not isinstance(p2, Point2D) else p2.arr
        return Point2D(self.arr + arr)

    def __sub__(self, p2):
        arr = p2 if not isinstance(p2, Point2D) else p2.arr
        return Point2D(self.arr - arr)

    def __mul__(self, t):
        arr = t if not isinstance(t, Point2D) else t.arr
        return Point2D(self.arr*arr)

    def __truediv__(self, t):
        return Point2D(self.arr/t)


class Vector2D(Point2D):
    def norm(self):
        return compute_norm(self.arr)

    def __str__(self) -> str:
        return "Vector2D({0}, {1})".format(self.x, self.y)


class Line2D:
    def __init__(self, point: Point2D, norm_vector: Vector2D):
        self.a, self.b = norm_vector.arr
        self.norm_vector = norm_vector
        self.d = -np.dot(norm_vector.arr, point.arr)
        self.point0 = point
        self.norm_val = compute_norm(self.norm_vector)

    def __str__(self) -> str:
        return f"Plane({self.a:.3f}*x + {self.b:.3f}*y + {self.d:.3f} = 0)"

    @classmethod
    def from_coefficient(cls, coefficient: np.ndarray):
        norm_vector = Vector2D(coefficient[:2])
        d = coefficient[2]
        point_arr = np.zeros(2)
        for i in range(2):
            if not _approx(norm_vector[i], 0.0):
                point_arr[i] = -d/norm_vector[i]
                break
        point = Point2D(point_arr)
        return Line2D(point, norm_vector)

    @classmethod
    def from_two_points(cls, point1: Point2D, point2: Point2D):
        v1 = Vector2D((point1 - point2).arr)
        norm_arr = v1.arr[::-1]*np.array([-1, 1])
        norm_vector = Vector2D(norm_arr)
        return Line2D(point1, norm_vector)
    
    def __call__(self, point: Point2D):
        return np.dot(self.norm_vector.arr, point.arr) + self.d

    def passes_through(self, point: Point2D) -> bool:
        return _approx(self.__call__(point), 0.0)

    def has_same_norm_vector_with(self, line2d) -> bool:
        k = None
        for i in range(2):
            if not _approx(self.norm_vector[i], 0.0):
                k = line2d.norm_vector[i] / self.norm_vector[i]
                break
        norm_vector_1 = self.norm_vector * k
        return _approx(line2d.norm_vector, norm_vector_1)

    def is_parallel_with_line(self, line2d) -> bool:
        if not self.has_same_norm_vector_with(line2d):
            return False
        return not line2d.passes_through(self.point0)

    def is_identical(self, line2d) -> bool:
        if not self.has_same_norm_vector_with(line2d):
            return False
        return line2d.passes_through(self.point0)

    def find_intersection_with_line(self, line2d) -> Point2D:
        if self.has_same_norm_vector_with(line2d):
            return None
        a = np.array([[self.a, self.b], [line2d.a, line2d.b]])
        b = np.array([-self.d, -line2d.d])
        x, y = np.linalg.solve(a, b)
        return Point2D(np.array([x, y]))

    def get_a_point(self):
        return self.point0

    def distance_to_point(self, point2d: Point2D):
        return np.abs(self.__call__(point2d))/self.norm_val

    def distance_to_line(self, line2d):
        if not self.is_parallel_with_line(line2d):
            return 0
        return np.abs(self.d - line2d.d)/self.norm_val


class Point3D:    
    def __init__(self, arr: np.ndarray):
        self.x, self.y, self.z = arr
        self.arr = arr

    def get_np_array(self) -> np.ndarray:
        return self.arr.copy()

    def __str__(self) -> str:
        return "Point3D({0}, {1}, {2})".format(self.x, self.y, self.z)

    def __getitem__(self, idx):
        return self.arr[idx]

    def approx(self, p2) -> bool:
        arr = p2 if not isinstance(p2, Point3D) else p2.arr
        return _approx(self.arr, arr)

    def __eq__(self, p2) -> bool:
        return self.approx(p2)

    def __add__(self, p2):
        arr = p2 if not isinstance(p2, Point3D) else p2.arr
        return Point3D(self.arr + arr)

    def __sub__(self, p2):
        arr = p2 if not isinstance(p2, Point3D) else p2.arr
        return Point3D(self.arr - arr)

    def __mul__(self, t):
        arr = t if not isinstance(t, Point3D) else t.arr
        return Point3D(self.arr*arr)

    def __truediv__(self, t):
        return Point3D(self.arr/t)


class Vector3D(Point3D):
    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __str__(self) -> str:
        return "Vector3D({0}, {1}, {2})".format(self.x, self.y, self.z)


class Line3D:
    def __init__(self, point: Point3D, direction_vector: Vector3D):
        self.a, self.b, self.c = direction_vector.arr
        self.direction_vector = direction_vector
        self.x0, self.y0, self.z0 = point.x, point.y, point.z
        self.point0 = point
        self.norm_direction_vector = compute_norm(self.direction_vector)
        #assert self.has_non_zero_coefficient()

    def __str__(self) -> str:
        return "Line3D({0}, {1})".format(self.point0, self.direction_vector)

    @classmethod
    def from_two_points(cls, point1: Point3D, point2: Point3D):
        direction_vector = Vector3D((point1 - point2).arr)
        return Line3D(point1, direction_vector)

    def has_non_zero_coefficient(self) -> bool:
        for i in range(3):
            if _approx(self.direction_vector[i], 0.0): return False
        return True

    def passes_through(self, point: Point3D) -> bool:
        t = None
        for i in range(3):
            if not _approx(self.direction_vector[i], 0.0):
                t = (point.arr[i] - self.point0.arr[i])/self.direction_vector[i]
                break
        point1 = self.point0 + self.direction_vector*t
        return point.approx(point1)
        
    def has_same_direction_vector_with(self, line) -> bool:
        k = None
        for i in range(3):
            if not _approx(self.direction_vector[i], 0.0):
                k = line.direction_vector[i] / self.direction_vector[i]
                break
        direction_vector_1 = self.direction_vector * k
        return _approx(line.direction_vector, direction_vector_1)

    def is_parallel(self, line) -> bool:
        if not self.has_same_direction_vector_with(line):
            return False
        return not self.passes_through(line.point0)

    def is_identical(self, line) -> bool:
        if not self.has_same_direction_vector_with(line):
            return False
        return self.passes_through(line.point0)

    def _compute_intersect_criterion(self, line):
        return np.dot(np.cross(self.direction_vector.arr, line.direction_vector.arr), (self.point0 - line.point0).arr)

    def is_skew(self, line) -> bool:
        intersect_criterion = self._compute_intersect_criterion(line)
        return not _approx(intersect_criterion, 0.0)

    def is_intersect(self, line) -> bool:
        if self.has_same_direction_vector_with(line):
            return False
        intersect_criterion = self._compute_intersect_criterion(line)
        return _approx(intersect_criterion, 0.0)

    def find_intersection(self, line) -> Point3D:
        '''
        This function should be called after calling the function is_intersection and its return result is True
        '''
        a = np.array([[self.a, -line.a, 1], [self.b, -line.b, 1], [self.c, -line.c, 1]])
        b = (line.point0 - self.point0).arr
        res = np.linalg.solve(a, b)
        t = res[0]
        return self.point0 + self.direction_vector*t

    def get_a_point(self):
        return self.point0

    def distance_to_point(self, point3d: Point3D):
        v = self.point0 - point3d
        cross = np.cross(v.arr, self.direction_vector.arr)
        return compute_norm(cross)/self.norm_direction_vector

    def distance_to_line(self, line3d):
        if self.is_parallel(line3d):
            return self.distance_to_point(line3d.get_a_point())
        if self.is_skew(line3d):
            u = np.cross(self.direction_vector.arr, line3d.direction_vector.arr)
            m = self.get_a_point() - line3d.get_a_point()
            u_m = np.abs(np.sum(u*m.arr))
            return u_m / compute_norm(u)
        return 0.0


class Segment3D(Line3D):
    def __init__(self, point1, point2):
        direction_vector = Vector3D((point1 - point2).arr)
        super().__init__(point1, direction_vector)
        self.endpoint1 = point1
        self.endpoint2 = point2

    @classmethod
    def is_between_two_points(cls, point1: Point3D, point2: Point3D, point3: Point3D):
        '''
        Check if point3 is between point1 and point2, given these 3 points are collinear 
        '''
        v1 = Vector3D((point1 - point3).arr)
        v2 = Vector3D((point3 - point2).arr)
        k = None
        for i in range(3):
            if not _approx(v2[i], 0.0):
                k = v1[i]/v2[i]
                break
        return _approx(v1, v2*k) and k > 0

    def contains(self, point: Point3D) -> bool:
        res = super().passes_through(point)
        return res and self.is_between_two_points(self.endpoint1, self.endpoint2, point)

    def is_identical(self, segment) -> bool:
        res = super().is_identical(segment)
        if not res:
            return False
        if (self.endpoint1, self.endpoint2) == (segment.endpoint1, self.endpoint2):
            return True
        if (self.endpoint1, self.endpoint2) == (segment.endpoint2, self.endpoint1):
            return True
        return False

    def is_intersect_with_line(self, line) -> bool:
        res = super().is_intersect(line)
        if not res:
            return False
        intersect_point = super().find_intersection(line)
        return self.contains(intersect_point)

    def is_intersect_with_segment(self, segment) -> bool:
        res = super().is_intersect(segment)
        if not res:
            return False
        intersect_point = super().find_intersection(segment)
        return self.contains(intersect_point) and segment.contains(intersect_point)

    def find_intersection(self, two_d_object) -> Point3D:
        '''
        This function should be called after calling the function is_intersection/is_intersect_with_segment 
        and its return result is True
        '''
        return super().find_intersection(two_d_object)

    def get_a_point(self):
        return self.endpoint1

    
class Plane:
    def __init__(self, point: Point3D, norm_vector: Vector3D):
        self.a, self.b, self.c = norm_vector.arr
        self.norm_vector = norm_vector
        self.d = -np.dot(norm_vector.arr, point.arr)
        self.point0 = point
        self.norm_val = compute_norm(self.norm_vector)

    def __str__(self) -> str:
        return f"Plane({self.a:.3f}*x + {self.b:.3f}*y + {self.c:.3f}*z + {self.d:.3f} = 0)"

    @classmethod
    def from_coefficient(cls, coefficient: np.ndarray):
        norm_vector = Vector3D(coefficient[:3])
        d = coefficient[3]
        point_arr = np.zeros(3)
        for i in range(3):
            if not _approx(norm_vector[i], 0.0):
                point_arr[i] = -d/norm_vector[i]
                break
        point = Point3D(point_arr)
        return Plane(point, norm_vector)

    @classmethod
    def from_three_points(cls, point1: Point3D, point2: Point3D, point3: Point3D):
        line1 = Line3D.from_two_points(point1, point2)
        if line1.passes_through(point3):
            return None
        v1 = Vector3D((point1 - point2).arr)
        v2 = Vector3D((point1 - point3).arr)
        norm_vector = Vector3D(np.cross(v1.arr, v2.arr))
        return Plane(point1, norm_vector)
    
    def __call__(self, point: Point3D):
        return np.dot(self.norm_vector.arr, point.arr) + self.d

    def passes_through(self, point: Point3D) -> bool:
        return _approx(self.__call__(point), 0.0)

    def has_same_norm_vector_with(self, plane) -> bool:
        k = None
        for i in range(3):
            if not _approx(self.norm_vector[i], 0.0):
                k = plane.norm_vector[i] / self.norm_vector[i]
                break
        norm_vector_1 = self.norm_vector * k
        return _approx(plane.norm_vector, norm_vector_1)

    def is_parallel_with_plane(self, plane) -> bool:
        if not self.has_same_norm_vector_with(plane):
            return False
        return not plane.passes_through(self.point0)

    def is_identical(self, plane) -> bool:
        if not self.has_same_norm_vector_with(plane):
            return False
        return plane.passes_through(self.point0)

    def find_intersection_with_plane(self, plane) -> Line3D:
        if self.has_same_norm_vector_with(plane):
            return None
        direction_vector = Vector3D(np.cross(self.norm_vector.arr, plane.norm_vector.arr))
        a = np.array([[self.a, self.b], [plane.a, plane.b]])
        b = np.array([-self.d, -plane.d])
        x, y = np.linalg.solve(a, b)
        point = Point3D(np.array([x, y, 0]))
        return Line3D(point, direction_vector)

    def is_parallel_with_line(self, line: Line3D):
        product = np.dot(self.norm_vector.arr, line.direction_vector.arr)
        return _approx(product, 0.0) and not self.passes_through(line.point0)

    def contains_line(self, line: Line3D):
        product = np.dot(self.norm_vector.arr, line.direction_vector.arr)
        return _approx(product, 0.0) and self.passes_through(line.point0)

    def is_intersect_with_line(self, line: Line3D) -> bool:
        return not self.is_parallel_with_line(line)

    def find_intersection_with_line(self, line: Line3D) -> Point3D:
        '''
        This function should be called after calling the function is_intersection_with_line 
        and its return result is True
        '''
        product = np.dot(self.norm_vector.arr, line.direction_vector.arr)
        t = -(np.dot(self.norm_vector.arr, line.point0.arr) + self.d)/product
        return line.point0 + line.direction_vector*t

    def is_intersect_with_segment(self, segment: Segment3D) -> bool:
        if not self.is_intersect_with_line(segment):
            return False
        possible_intersect_point = self.find_intersection_with_line(segment)
        if not segment.contains(possible_intersect_point):
            return False
        return True

    def find_intersection_with_segment(self, segment: Segment3D) -> Point3D:
        '''
        This function should be called after calling the function is_intersection_with_segment 
        and its return result is True
        '''
        return self.find_intersection_with_line(segment)

    def get_a_point(self):
        return self.point0

    def distance_to_point(self, point3d: Point3D):
        return np.abs(self.__call__(point3d))/self.norm_val

    def distance_to_line(self, line3d: Line3D):
        if not self.is_parallel_with_line(line3d):
            return 0
        return self.distance_to_point(line3d.get_a_point())

    def distance_to_plane(self, plane):
        if not self.is_parallel_with_plane(plane):
            return 0
        return np.abs(self.d - plane.d)/self.norm_val

#TODO: find intersection between cuboic and line, segment, plane
class Cuboic:
    def __init__(self, cube_dict, endplate_ratio=0.2):
        self.cube_dict = copy.deepcopy(cube_dict)
        self.endplate_ratio = endplate_ratio
        self.epsilon = 1e-9
        
        x, y, z = cube_dict['position']['x'], cube_dict['position']['y'], cube_dict['position']['z']
        rx, ry, rz = cube_dict['rotation']['x'], cube_dict['rotation']['y'], cube_dict['rotation']['z']
        dx, dy, dz = cube_dict['dimensions']['x'], cube_dict['dimensions']['y'], cube_dict['dimensions']['z']
        
        self.x, self.y, self.z = x, y, z
        self.center = np.asarray([x, y, z])
        self.rx, self.ry, self.rz = rx, ry, rz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.half_dx, self.half_dy, self.half_dz = dx/2, dy/2, dz/2
        self.half_d = self.half_dx, self.half_dy, self.half_dz
        
        self.rotation_matrix = np.multiply(o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz)), -1)
        self.inverse_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        
        self.pA_orig = (x - self.half_dx, y - self.half_dy, z - self.half_dz)
        self.pB_orig = (x - self.half_dx, y + self.half_dy, z - self.half_dz)
        self.pC_orig = (x + self.half_dx, y + self.half_dy, z - self.half_dz)
        self.pD_orig = (x + self.half_dx, y - self.half_dy, z - self.half_dz)
        self.pE_orig = (x - self.half_dx, y - self.half_dy, z + self.half_dz)
        self.pF_orig = (x - self.half_dx, y + self.half_dy, z + self.half_dz)
        self.pG_orig = (x + self.half_dx, y + self.half_dy, z + self.half_dz)
        self.pH_orig = (x + self.half_dx, y - self.half_dy, z + self.half_dz)
        self.all_points_orig = np.asarray([
            self.pA_orig, self.pB_orig, self.pC_orig, self.pD_orig,
            self.pE_orig, self.pF_orig, self.pG_orig, self.pH_orig])
        self.xs_orig, self.ys_orig, self.zs_orig = \
            self.all_points_orig[:,0], self.all_points_orig[:,1], self.all_points_orig[:2]
        
        self.all_points = self.rotate(self.all_points_orig, self.rotation_matrix, self.center)
        self.xs, self.ys, self.zs = self.all_points[:,0], self.all_points[:,1], self.all_points[:,2]

        self.pointA = Point3D(self.all_points[0])
        self.pointB = Point3D(self.all_points[1])
        self.pointC = Point3D(self.all_points[2])
        self.pointD = Point3D(self.all_points[3])
        self.pointE = Point3D(self.all_points[4])
        self.pointF = Point3D(self.all_points[5])
        self.pointG = Point3D(self.all_points[6])
        self.pointH = Point3D(self.all_points[7])
        self.points = [
            self.pointA, self.pointB, self.pointC, self.pointD,
            self.pointE, self.pointF, self.pointG, self.pointH
        ]

        self.segments = [
            Segment3D(self.pointA, self.pointB), Segment3D(self.pointB, self.pointC),
            Segment3D(self.pointC, self.pointD), Segment3D(self.pointD, self.pointA),
            Segment3D(self.pointA, self.pointE), Segment3D(self.pointB, self.pointF),
            Segment3D(self.pointD, self.pointH), Segment3D(self.pointC, self.pointG),
            Segment3D(self.pointE, self.pointF), Segment3D(self.pointF, self.pointG),
            Segment3D(self.pointG, self.pointH), Segment3D(self.pointH, self.pointE)
        ]

        self.planes = [
            Plane.from_three_points(self.pointA, self.pointB, self.pointC), #[ABCD]
            Plane.from_three_points(self.pointA, self.pointE, self.pointB), #[AEFB]
            Plane.from_three_points(self.pointA, self.pointE, self.pointD), #[AEHD]
            Plane.from_three_points(self.pointC, self.pointD, self.pointH), #[CDHG]
            Plane.from_three_points(self.pointB, self.pointC, self.pointF), #[BCGF]
            Plane.from_three_points(self.pointE, self.pointF, self.pointG), #[EFGH]
        ]
        
        self.top_endplate_dict, self.bottom_endplate_dict = None, None
        self.top_endplate_cuboic, self.bottom_endplate_cuboic = None, None
        
    def rotate(cls, points, rotation_matrix, rotation_center):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.rotate(rotation_matrix, center=rotation_center)
        return np.asarray(pcl.points)
    
    def visualize_with_point_clouds(self, point_clouds, color='blue'):
        plotly_x, plotly_y, plotly_z = point_clouds[:,0], point_clouds[:,1], point_clouds[:,2]
        fig = go.Figure(data=[
            go.Scatter3d(x=plotly_x, y=plotly_y, z=plotly_z, mode='markers', marker=dict(size=2)),
            go.Mesh3d(
                # 8 vertices of a cube
                x=self.xs, y=self.ys, z=self.zs,
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.6,
                color=color
            )                    
        ])
        return fig
    
    def visualize_with_point_clouds_left_right(
            self, left_points, right_points, left_color='blue', right_color='red'):
        left_x, left_y, left_z = left_points[:,0], left_points[:,1], left_points[:,2]
        right_x, right_y, right_z = right_points[:,0], right_points[:,1], right_points[:,2]
        fig = go.Figure(data=[
            go.Scatter3d(x=left_x, y=left_y, z=left_z, mode='markers', marker=dict(size=2, color=left_color)),
            go.Scatter3d(x=right_x, y=right_y, z=right_z, mode='markers', marker=dict(size=2, color=right_color)),
        ])
        return fig
    
    def visualize_with_point_clouds_1(
            self, point_clouds, inside_color='yellow', outside_color='blue', cuboic_color='red'):
        inside_cube_points = []
        outside_cube_points = []
        for point in point_clouds:
            if self.contains(point): inside_cube_points.append(point)
            else: outside_cube_points.append(point)
        inside_cube_points = np.asarray(inside_cube_points)
        outside_cube_points = np.asarray(outside_cube_points)
        inside_x, inside_y, inside_z = inside_cube_points[:,0], inside_cube_points[:,1], inside_cube_points[:,2]
        outside_x, outside_y, outside_z =\
            outside_cube_points[:,0], outside_cube_points[:,1], outside_cube_points[:,2]
        fig = go.Figure(data=[
            go.Scatter3d(
                x=inside_x, y=inside_y, z=inside_z, mode='markers', marker=dict(size=2, color=inside_color)),
            go.Scatter3d(
                x=outside_x, y=outside_y, z=outside_z, mode='markers', marker=dict(
                size=2, color=outside_color, opacity=0.5)),
            go.Mesh3d(
                # 8 vertices of a cube
                x=self.xs, y=self.ys, z=self.zs,
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.6,
                color=cuboic_color
            )                    
        ])
        return fig
    
    def visualize_endplate(self, endplate_cuboic, fig):
        fig.add_mesh3d(
            x=endplate_cuboic.xs, y=endplate_cuboic.ys, z=endplate_cuboic.zs,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.8,
            color='gray'
        )
        return fig
    
    
    def visualize_endplates(self, top_endplate_cuboic, bottom_endplate_cuboic, fig):
        fig = self.visualize_endplate(top_endplate_cuboic, fig)
        fig = self.visualize_endplate(bottom_endplate_cuboic, fig)
        return fig
    
    def contains(self, point):
        point = np.asarray([point])
        transform_point = self.rotate(point, self.inverse_rotation_matrix, self.center)[0]
        def is_inside_one_d(px, center_x, half_d, epsilon):
            return px >= center_x - half_d - epsilon and px <= center_x + half_d + epsilon
        
        for i in range(3):
            if not is_inside_one_d(transform_point[i], self.center[i], self.half_d[i], self.epsilon):
                return False
        return True
    
    def get_endplate_cuboics(self):
        top_endplate_dict = copy.deepcopy(self.cube_dict)
        bottom_endplate_dict = copy.deepcopy(self.cube_dict)
        top_endplate_dict['dimensions']['x'] = self.endplate_ratio*self.dx
        bottom_endplate_dict['dimensions']['x'] = self.endplate_ratio*self.dx

        top_x = self.x + self.dx/2 - self.dx*self.endplate_ratio/2
        bottom_x = self.x - self.dx/2 + self.dx*self.endplate_ratio/2
        points = np.asarray([
            [top_x, self.y, self.z],
            [bottom_x, self.y, self.z]
        ])
        points = self.rotate(points, self.rotation_matrix, self.center)
        
        (
            top_endplate_dict['position']['x'], 
            top_endplate_dict['position']['y'], 
            top_endplate_dict['position']['z']
        ) = points[0]
        
        (
            bottom_endplate_dict['position']['x'], 
            bottom_endplate_dict['position']['y'], 
            bottom_endplate_dict['position']['z']
        ) = points[1]
        
        self.top_endplate_dict = top_endplate_dict
        self.bottom_endplate_dict = bottom_endplate_dict
        self.top_endplate_cuboic = Cuboic(self.top_endplate_dict)
        self.bottom_endplate_cuboic = Cuboic(self.bottom_endplate_dict)
        return self.top_endplate_cuboic, self.bottom_endplate_cuboic


def sort_points_convex(points: np.ndarray) -> np.ndarray:
    pca = sklearn.decomposition.PCA(n_components=2)
    points_2d = pca.fit_transform(points)
    hull = scipy.spatial.ConvexHull(points_2d)
    print(hull.vertices)
    return points[hull.vertices]