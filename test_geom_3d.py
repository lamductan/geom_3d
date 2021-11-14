###############################################################################
# Usage: python test_geom_3d.py                                               #               
###############################################################################

import json
import unittest

import numpy as np
import open3d as o3d

from geom_3d import Point3D, Vector3D, Line3D, Segment3D, Plane, Cuboic
from geom_3d import _approx, sort_points_convex


class TestPoint3D(unittest.TestCase):
    def test_str(self):
        x, y, z = 1, 2, 3
        point = Point3D(np.array([x, y, z]))
        self.assertTrue(True)

    def test_add_1(self):
        point1 = Point3D(np.array([1, 2, 3]))
        point2 = Point3D(np.array([4, 5, 6]))
        point3 = Point3D(np.array([1+4, 2+5, 3+6]))
        self.assertEqual(point1 + point2, point3)

    def test_add_2(self):
        point1 = Point3D(np.array([1, 2, 3]))
        arr2 = np.array([4, 5, 6])
        point3 = Point3D(np.array([1+4, 2+5, 3+6]))
        self.assertEqual(point1 + arr2, point3)

    def test_sub_1(self):
        point1 = Point3D(np.array([1, 2, 3]))
        point2 = Point3D(np.array([4, 5, 6]))
        point3 = Point3D(np.array([1-4, 2-5, 3-6]))
        self.assertEqual(point1 - point2, point3)

    def test_sub_2(self):
        point1 = Point3D(np.array([1, 2, 3]))
        arr2 = np.array([4, 5, 6])
        point3 = Point3D(np.array([1-4, 2-5, 3-6]))
        self.assertEqual(point1 - arr2, point3)

    def test_mul_1(self):
        point1 = Point3D(np.array([1, 2, 3]))
        point2 = Point3D(np.array([4, 5, 6]))
        point3 = Point3D(np.array([1*4, 2*5, 3*6]))
        self.assertEqual(point1 * point2, point3)

    def test_mul_2(self):
        point1 = Point3D(np.array([1, 2, 3]))
        arr2 = np.array([4, 5, 6])
        point3 = Point3D(np.array([1*4, 2*5, 3*6]))
        self.assertEqual(point1 * arr2, point3)


class TestLine3D(unittest.TestCase):
    def test_init(self):
        point = Point3D(np.array([1, 2, 3]))
        direction_vector = Vector3D(np.array([4, 5, 6]))
        line = Line3D(point, direction_vector)
        self.assertEqual(point, line.point0)
        self.assertTrue(_approx(direction_vector, line.direction_vector))

    def test_has_non_zero_coefficient(self):
        point = Point3D(np.array([1, 2, 3]))
        direction_vector = Vector3D(np.array([0, 0, 0]))
        line = Line3D(point, direction_vector)
        self.assertFalse(line.has_non_zero_coefficient())
        direction_vector = Vector3D(np.array([1, 2, 3]))
        line = Line3D(point, direction_vector)
        self.assertTrue(line.has_non_zero_coefficient())

    def test_passes_through_1(self):
        line = Line3D(
            point=Point3D(np.array([-3, -2, 6])), 
            direction_vector=Vector3D(np.array([1, 3/2, 2]))
        )
        point1 = Point3D(np.array([3, 7, 18]))
        point2 = Point3D(np.array([3, 7, 19]))
        self.assertTrue(line.passes_through(point1))
        self.assertFalse(line.passes_through(point2))

    def test_passes_through_2(self):
        line = Line3D(
            point=Point3D(np.array([-3, -2, 6])), 
            direction_vector=Vector3D(np.array([0, 3/2, 2]))
        )
        point1 = Point3D(np.array([-3, 7, 18]))
        point2 = Point3D(np.array([3, 7, 18]))
        self.assertTrue(line.passes_through(point1))
        self.assertFalse(line.passes_through(point2))

    def test_parallel_and_identical(self):
        line1 = Line3D(
            point=Point3D(np.array([-1, -3, 2])), 
            direction_vector=Vector3D(np.array([3, 2, -1]))
        )
        line2 = Line3D(
            point=Point3D(np.array([26, 21, 0])), 
            direction_vector=Vector3D(np.array([-3, -2, 1]))
        )
        line3 = Line3D(
            point=Point3D(np.array([26, 21, 0])), 
            direction_vector=Vector3D(np.array([-4, -2, 1]))
        )
        self.assertTrue(line1.is_parallel(line2) and line2.is_parallel(line1))
        self.assertFalse(line1.is_parallel(line1))
        self.assertTrue(not line1.is_parallel(line3) and not line1.is_identical(line3))

    def test_is_skew(self):
        line1 = Line3D(
            point=Point3D(np.array([0, 5, 14])), 
            direction_vector=Vector3D(np.array([0, -2, -3]))
        )
        line2 = Line3D(
            point=Point3D(np.array([9, 1, -1])), 
            direction_vector=Vector3D(np.array([-4, 1, 5]))
        )
        self.assertTrue(line1.is_skew(line2) and line2.is_skew(line1))

    def test_intersect(self):
        line1 = Line3D(
            point=Point3D(np.array([-3, -2, 6])), 
            direction_vector=Vector3D(np.array([2, 3, 4]))
        )
        line2 = Line3D(
            point=Point3D(np.array([5, -1, 20])), 
            direction_vector=Vector3D(np.array([1, -4, 1]))
        )
        self.assertTrue(line1.is_intersect(line2) and line2.is_intersect(line1))
        intersect_point_1 = line1.find_intersection(line2)
        intersect_point_2 = line2.find_intersection(line1)
        self.assertEqual(intersect_point_1, intersect_point_2)
        self.assertTrue(line1.passes_through(intersect_point_2) and line2.passes_through(intersect_point_1))

    def test_distance_to_point(self):
        line = Line3D(
            point=Point3D(np.array([0, 1, -1])), 
            direction_vector=Vector3D(np.array([1, 2, 1]))
        )
        point = Point3D(np.array([1, 1, 1]))
        distance = line.distance_to_point(point)
        self.assertAlmostEqual(distance, np.sqrt(14)/2)

    def test_distance_to_skew_line(self):
        line1 = Line3D(
            point=Point3D(np.array([0, 1, 6])), 
            direction_vector=Vector3D(np.array([1, 2, 3]))
        )
        line2 = Line3D(
            point=Point3D(np.array([1, -2, 3])), 
            direction_vector=Vector3D(np.array([1, 1, -1]))
        )
        distance = line1.distance_to_line(line2)
        self.assertAlmostEqual(distance, np.sqrt(42)/3)


class TestSegment3D(unittest.TestCase):
    def test_contains(self):
        point1 = Point3D(np.array([2, -4, 3]))
        point2 = Point3D(np.array([2, 2, 7]))
        point3 = (point1 + point2)/2
        segment = Segment3D(point1, point2)
        self.assertTrue(segment.contains(point3))

    def test_intersect_with_line(self):
        segment = Segment3D(
            point1=Point3D(np.array([0, 0, 0])),
            point2=Point3D(np.array([3, 0, 6]))
        )
        line = Line3D.from_two_points(
            point1=Point3D(np.array([-1, -2, -3])),
            point2=Point3D(np.array([3, 2, 7]))
        )
        self.assertTrue(segment.is_intersect_with_line(line))
        intersect_point = segment.find_intersection(line)
        point = Point3D(np.array([1, 0, 2]))
        self.assertEqual(point, intersect_point)

    def test_intersect_with_segment(self):
        segment1 = Segment3D(
            point1=Point3D(np.array([0, 0, 0])),
            point2=Point3D(np.array([3, 0, 6]))
        )
        segment2 = Segment3D(
            point1=Point3D(np.array([-1, -2, -3])),
            point2=Point3D(np.array([3, 2, 7]))
        )
        point = Point3D(np.array([1, 0, 2]))
        self.assertTrue(segment1.contains(point))
        self.assertTrue(segment2.contains(point))

        self.assertTrue(segment1.is_intersect_with_segment(segment2) and segment2.is_intersect_with_segment(segment1))
        intersect_point1 = segment1.find_intersection(segment2)
        intersect_point2 = segment2.find_intersection(segment1)
        self.assertTrue(point == intersect_point1 and point == intersect_point2)


class TestPlane(unittest.TestCase):
    def test_init(self):
        point = Point3D(np.array([1, 2, 3]))
        norm_vector = Vector3D(np.array([4, 5, 6]))
        plane = Plane(point, norm_vector)
        self.assertEqual(point, plane.point0)
        self.assertTrue(_approx(norm_vector, plane.norm_vector))

    def test_from_three_points(self):
        point1 = Point3D(np.array([1, 2, 3]))
        point2 = Point3D(np.array([2, 5, 9]))
        point3 = Point3D(np.array([-1, -3, -10]))
        plane = Plane.from_three_points(point1, point2, point3)
        self.assertTrue(plane.passes_through(point1) and plane.passes_through(point2) and plane.passes_through(point3))

    def test_passes_through(self):
        plane = Plane.from_coefficient(np.array([1, -1, -1, -4]))
        point1 = Point3D(np.array([2, -1, -1]))
        point2 = Point3D(np.array([1, -2, 0]))
        self.assertTrue(plane.passes_through(point1))
        self.assertFalse(plane.passes_through(point2))

    def test_has_same_norm_vector_with(self):
        plane1 = Plane.from_coefficient(np.array([1, -1, -1, -4]))
        plane2 = Plane(
            point=Point3D(np.array([1, 2, 3])),
            norm_vector=Vector3D(np.array([2.1, -2.1, -2.1]))
        )
        self.assertTrue(plane1.has_same_norm_vector_with(plane2))

    def test_find_intersection_with_plane(self):
        plane1 = Plane.from_coefficient(np.array([1, -3, 1, 0]))
        plane2 = Plane.from_coefficient(np.array([1, 1, -1, 4]))
        line1 = plane1.find_intersection_with_plane(plane2)
        line2 = plane2.find_intersection_with_plane(plane1)
        line3 = Line3D(
            point=Point3D(np.array([-2, 0, 2])),
            direction_vector=Vector3D(np.array([1, 1, 2]))
        )
        self.assertTrue(line1.is_identical(line2) and line2.is_identical(line3))

    def test_is_parallel_and_identical_with_plane(self):
        plane1 = Plane.from_coefficient(np.array([1, 1, 1, -1]))
        plane2 = Plane.from_coefficient(np.array([2, 2, 2, 3]))
        self.assertTrue(plane1.is_parallel_with_plane(plane2))
        self.assertFalse(plane1.is_identical(plane2))
        plane3 = Plane.from_coefficient(np.array([1, -2, 1, -3]))
        plane4 = Plane.from_coefficient(np.array([2, -4, 2, -6]))
        self.assertFalse(plane3.is_parallel_with_plane(plane4))
        self.assertTrue(plane3.is_identical(plane4))
        plane5 = Plane.from_coefficient(np.array([1, 2, -1, 5]))
        plane6 = Plane.from_coefficient(np.array([2, 3, -7, 4]))
        self.assertFalse(plane5.is_parallel_with_plane(plane6))
        self.assertFalse(plane5.is_identical(plane6))

    def test_parallel_with_line(self):
        line1 = Line3D(
            point=Point3D(np.array([2, 0, 1])),
            direction_vector=Vector3D(np.array([1, 2, -1]))
        )
        line2 = Line3D(
            point=Point3D(np.array([0, 0, 1])),
            direction_vector=Vector3D(np.array([1, 1, -1]))
        )
        plane1 = Plane.from_coefficient(np.array([1, 0, 1, 1]))
        plane2 = Plane.from_coefficient(np.array([1, 1, 0, 1]))
        self.assertTrue(plane1.is_parallel_with_line(line1) and plane1.is_parallel_with_line(line2))
        self.assertFalse(plane2.is_parallel_with_line(line1) and plane2.is_parallel_with_line(line2))

    def test_contains_line(self):
        plane = Plane.from_coefficient(np.array([1, -1, -1, -4]))
        point1 = Point3D(np.array([2, -1, -1]))
        point2 = Point3D(np.array([1, -3, 0]))
        line1 = Line3D.from_two_points(point1, point2)
        line2 = Line3D.from_two_points(point1, point2 - point2)
        self.assertTrue(plane.contains_line(line1))
        self.assertFalse(plane.contains_line(line2))

    def test_find_intersection_with_line(self):
        line = Line3D(
            point=Point3D(np.array([2, 0, -1])),
            direction_vector=Vector3D(np.array([-3, 1, 2]))
        )
        plane = Plane.from_coefficient(np.array([1, 2, -3, 2]))
        self.assertTrue(plane.is_intersect_with_line(line))
        point1 = Point3D(np.array([-1, 1, 1]))
        point2 = plane.find_intersection_with_line(line)
        self.assertEqual(point1, point2)

    def test_find_intersection_with_segment(self):
        segment = Segment3D(
            point1=Point3D(np.array([4, 0, 1])),
            point2=Point3D(np.array([-2, 2, 3]))
        )
        plane = Plane.from_coefficient(np.array([3, -1, -1, 0]))
        self.assertTrue(plane.is_intersect_with_segment(segment))
        point1 = (segment.endpoint1 + segment.endpoint2)/2
        point2 = plane.find_intersection_with_segment(segment)
        self.assertEqual(point1, point2)

    def test_distance_to_point(self):
        plane = Plane.from_coefficient(np.array([3, 4, 2, 4]))
        point = Point3D(np.array([1, -2, 3]))
        distance = plane.distance_to_point(point)
        self.assertAlmostEqual(distance, 5.0/np.sqrt(29))

    def test_distance_to_line(self):
        plane = Plane.from_coefficient(np.array([2, -2, -1, 1]))
        line = Line3D(
            point=Point3D(np.array([1, -2, 1])),
            direction_vector=Vector3D(np.array([2, 1, 2]))
        )
        distance = plane.distance_to_line(line)
        self.assertAlmostEqual(distance, 2)

    def test_distance_to_plane(self):
        plane1 = Plane.from_coefficient(np.array([1, 2, 2, -10]))
        plane2 = Plane.from_coefficient(np.array([1, 2, 2, -3]))
        distance = plane1.distance_to_plane(plane2)
        self.assertAlmostEqual(distance, 7.0/3)


class TestCuboic(unittest.TestCase):
    def setUp(self):
        self.supervisely_pointcloud_filepath = '/G/KAI/landmark_detection/draft/pedicle_contours/KAI_supervisely/point_clouds_pcd_v1_draft/pointcloud/03_verse208_seg.pcd'
        self.supervisely_ann_filepath = '/G/KAI/landmark_detection/draft/pedicle_contours/KAI_supervisely/point_clouds_pcd_v1_draft/ann/03_verse208_seg.pcd.json'
        self.ps = o3d.io.read_point_cloud(self.supervisely_pointcloud_filepath)
        self.points = np.asarray(self.ps.points)
        self.ann = json.load(open(self.supervisely_ann_filepath))
        self.cube_dict = self.ann['figures'][0]['geometry']
        self.cuboic = Cuboic(self.cube_dict)

    def test_convex_hull(self):
        points = np.array([
            self.cuboic.pointA.arr,
            self.cuboic.pointC.arr,
            self.cuboic.pointB.arr,
            self.cuboic.pointD.arr
        ])
        print("pointA:, ", self.cuboic.pointA.arr)
        print("pointC:, ", self.cuboic.pointC.arr)
        print("pointB:, ", self.cuboic.pointB.arr)
        print("pointD:, ", self.cuboic.pointD.arr)
        convex_points = sort_points_convex(points)
        print("convex_points")
        print(convex_points)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPoint3D('test_add_1'))
    suite.addTest(TestPoint3D('test_add_2'))
    suite.addTest(TestPoint3D('test_sub'))
    #suite.addTest(TestLine3D('test_init'))
    return suite

def run_test_classes():
    # Run only the tests in the specified classes
    test_classes_to_run = [TestCuboic]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)


if __name__ == '__main__':
    # To test specific test cases, add those test cases to test suite in 'suite()' function 
    # and uncomment the following two lines
    #runner = unittest.TextTestRunner()
    #runner.run(suite())

    # To test specific test classes, add those test classes into test_classes_to_run in 'run_test_classes()' function
    # and uncomment the following line
    run_test_classes()
    
    #To test all test cases, uncomment the following line:
    unittest.main()