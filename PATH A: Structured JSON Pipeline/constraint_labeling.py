"""
PHASE 3: Geometric Constraint Detection with Parameter Extraction
Enhanced version that extracts ACTUAL geometric parameters for regression training
NOW WITH EXPLICIT NO_CONSTRAINT CLASS for better GNN training
"""

import torch
import numpy as np
import json
import os
import glob
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import networkx as nx
from torch_geometric.data import Data
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GeometricConstraintDetector:
    """
    Detect geometric constraints with PARAMETER extraction from CAD graphs
    NOW WITH EXPLICIT NO_CONSTRAINT CLASS for hierarchical training
    """
    
    def __init__(self, 
                 angle_tolerance=5.0,      # Degrees for parallel/perpendicular
                 distance_tolerance=0.01,  # Normalized units
                 radius_tolerance=0.05,    # Relative tolerance for radii
                 length_tolerance=0.05,    # Relative tolerance for lengths
                 min_confidence_threshold=0.3):  # Minimum confidence to consider constraint
        
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.radius_tolerance = radius_tolerance
        self.length_tolerance = length_tolerance
        self.min_confidence_threshold = min_confidence_threshold
        
        # Define constraint types with their parameter dimensions
        # CHANGED: Added NO_CONSTRAINT as first class
        self.constraint_definitions = {
            # Special class for edges with no constraint
            'NO_CONSTRAINT': {
                'params': ['confidence', 'is_topological', 'is_geometric'],
                'description': 'No geometric constraint between entities (just topology)',
                'param_dim': 3,
                'is_no_constraint': True
            },
            
            # Face-Face constraints (3 parameters each)
            'PARALLEL_FACES': {
                'params': ['distance', 'angle_deviation', 'alignment_dot'],
                'description': 'Two planar faces are parallel',
                'param_dim': 3
            },
            'PERPENDICULAR_FACES': {
                'params': ['angle', 'distance_ratio', 'alignment_cross'],
                'description': 'Two planar faces are perpendicular',
                'param_dim': 3
            },
            'COPLANAR_FACES': {
                'params': ['distance', 'normal_angle', 'centroid_distance'],
                'description': 'Two planar faces lie in same plane',
                'param_dim': 3
            },
            'CONCENTRIC_CYLINDERS': {
                'params': ['radius_ratio', 'axis_angle', 'center_distance'],
                'description': 'Two cylindrical faces share same axis',
                'param_dim': 3
            },
            'TANGENT_FACES': {
                'params': ['distance', 'curvature_ratio', 'contact_angle'],
                'description': 'Two faces are tangent to each other',
                'param_dim': 3
            },
            'OFFSET_FACES': {
                'params': ['offset_distance', 'normal_angle', 'area_ratio'],
                'description': 'Two faces are offset from each other',
                'param_dim': 3
            },
            
            # Face-Edge constraints (3 parameters each)
            'EDGE_ON_FACE': {
                'params': ['distance_to_face', 'edge_length_ratio', 'angle_to_normal'],
                'description': 'Edge lies on a planar face',
                'param_dim': 3
            },
            'EDGE_PARALLEL_TO_FACE': {
                'params': ['angle', 'distance_to_face', 'edge_face_distance'],
                'description': 'Edge direction is parallel to face normal',
                'param_dim': 3
            },
            'EDGE_PERPENDICULAR_TO_FACE': {
                'params': ['angle', 'edge_face_distance', 'projection_ratio'],
                'description': 'Edge direction is perpendicular to face normal',
                'param_dim': 3
            },
            'EDGE_TANGENT_TO_FACE': {
                'params': ['tangent_angle', 'distance_to_surface', 'cylinder_radius'],
                'description': 'Edge is tangent to cylindrical face',
                'param_dim': 3
            },
            
            # Edge-Edge constraints (3 parameters each)
            'PARALLEL_EDGES': {
                'params': ['distance', 'angle_deviation', 'length_ratio'],
                'description': 'Two edges are parallel',
                'param_dim': 3
            },
            'PERPENDICULAR_EDGES': {
                'params': ['angle', 'min_distance', 'length_product'],
                'description': 'Two edges are perpendicular',
                'param_dim': 3
            },
            'COLLINEAR_EDGES': {
                'params': ['overlap_ratio', 'angle_deviation', 'distance_between'],
                'description': 'Two edges lie on same line',
                'param_dim': 3
            },
            'EQUAL_LENGTH': {
                'params': ['length_ratio', 'length_difference', 'tolerance_ratio'],
                'description': 'Two or more edges have equal length',
                'param_dim': 3
            },
            'EQUAL_RADIUS': {
                'params': ['radius_ratio', 'radius_difference', 'tolerance_ratio'],
                'description': 'Two or more circular edges have equal radius',
                'param_dim': 3
            },
            'CONCENTRIC_CIRCLES': {
                'params': ['center_distance', 'radius_ratio', 'plane_angle'],
                'description': 'Two circular edges share same center',
                'param_dim': 3
            },
            
            # Vertex constraints (3 parameters each)
            'COINCIDENT_VERTICES': {
                'params': ['distance', 'valence_difference', 'connectivity_ratio'],
                'description': 'Two vertices are coincident',
                'param_dim': 3
            },
            'VERTEX_ON_FACE': {
                'params': ['distance_to_face', 'face_area', 'vertex_face_angle'],
                'description': 'Vertex lies on a face',
                'param_dim': 3
            },
            'VERTEX_ON_EDGE': {
                'params': ['distance_to_edge', 'edge_length', 'position_ratio'],
                'description': 'Vertex lies on an edge',
                'param_dim': 3
            },
            
            # Pattern constraints (4 parameters each - more complex)
            'EQUAL_SPACING': {
                'params': ['spacing_std', 'spacing_mean', 'count', 'regularity'],
                'description': 'Multiple features have equal spacing',
                'param_dim': 4
            },
            'RECTANGULAR_PATTERN': {
                'params': ['row_spacing', 'col_spacing', 'row_count', 'col_count'],
                'description': 'Features arranged in rectangular grid',
                'param_dim': 4
            },
            'CIRCULAR_PATTERN': {
                'params': ['radius', 'angular_spacing', 'count', 'center_distance'],
                'description': 'Features arranged in circular pattern',
                'param_dim': 4
            }
        }
        
        # Map constraint names to indices (NO_CONSTRAINT is index 0)
        self.constraint_names = list(self.constraint_definitions.keys())
        self.num_constraint_types = len(self.constraint_names)
        self.constraint_to_idx = {name: i for i, name in enumerate(self.constraint_names)}
        
        # Verify NO_CONSTRAINT is index 0
        assert self.constraint_names[0] == 'NO_CONSTRAINT', "NO_CONSTRAINT must be first class"
        print(f"Initialized with {self.num_constraint_types} constraint types")
        print(f"  - NO_CONSTRAINT index: 0")
        print(f"  - Active constraints: {self.num_constraint_types - 1} types")
        
        # Maximum parameters per constraint (for tensor padding)
        self.max_params_per_constraint = 4
        
        # Define surface types from Phase 2 (for reference)
        self.surface_types = [
            'PLANE', 'PLANE_SIMPLE',
            'CYLINDER', 'CYLINDER_SIMPLE',
            'CONE', 'CONE_SIMPLE',
            'SPHERE', 'SPHERE_SIMPLE',
            'SPLINE_SURFACE',
            'PROCESSING_ERROR',
            'UNKNOWN'
        ]
        
        # Define curve types from Phase 2
        self.curve_types = [
            'LINE', 'LINE_SIMPLE',
            'CIRCLE', 'CIRCLE_SIMPLE',
            'ELLIPSE', 'ELLIPSE_SIMPLE',
            'SPLINE',
            'PROCESSING_ERROR',
            'UNKNOWN_CURVE'
        ]
    
    def safe_startswith(self, text, prefix):
        """Safely check if text starts with prefix"""
        if text is None:
            return False
        return str(text).startswith(prefix)
    
    def safe_get_type_base(self, type_string):
        """Extract base type from type string (removes _SIMPLE suffix)"""
        if type_string is None:
            return 'UNKNOWN'
        type_str = str(type_string)
        if type_str.endswith('_SIMPLE'):
            return type_str.replace('_SIMPLE', '')
        return type_str
    
    def calculate_angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1 = np.array(v1, dtype=np.float32)
        v2 = np.array(v2, dtype=np.float32)
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate angle
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def extract_node_features(self, graph_data, node_idx):
        """Extract geometric features from node using enhanced Phase 2 structure"""
        # [Keep your existing extract_node_features method exactly as is]
        # ... (I'm not repeating it here for brevity, but keep your original code)
        node_feat = graph_data.x[node_idx]
        node_feat_np = node_feat.numpy() if torch.is_tensor(node_feat) else node_feat
    
        features = {
            'is_face': node_feat_np[0] > 0.5,
            'is_edge': node_feat_np[1] > 0.5,
            'is_vertex': node_feat_np[2] > 0.5,
            'feature_vector': node_feat_np,
            'raw_features': node_feat,
            'confidence': 1.0  # Default confidence
        }
    
        if features['is_face']:
            # Face features based on enhanced Phase 2 structure
            features['entity_type'] = 'face'
        
            # Surface type from one-hot encoding (indices 3:14 for 11 types)
            type_encoding = node_feat_np[3:14]
            if np.any(type_encoding):
                type_idx = np.argmax(type_encoding)
                if type_idx < len(self.surface_types):
                    features['surface_type'] = self.surface_types[type_idx]
                else:
                    features['surface_type'] = 'UNKNOWN'
            else:
                features['surface_type'] = 'UNKNOWN'
            
            # Set confidence based on type
            if self.safe_startswith(features['surface_type'], 'PROCESSING_ERROR'):
                features['confidence'] = 0.1
            elif '_SIMPLE' in features['surface_type']:
                features['confidence'] = 0.5
            else:
                features['confidence'] = 1.0
        
            # Primary geometric parameter (indices 14:17) - normal/axis
            features['primary_param'] = node_feat_np[14:17]
        
            # Secondary parameter (index 17)
            features['secondary_param'] = node_feat_np[17]
        
            # Tertiary parameter (index 18) - for additional params
            features['tertiary_param'] = node_feat_np[18]
        
            # Area (index 19)
            features['area'] = abs(node_feat_np[19])
        
            # Centroid (indices 20:23)
            features['centroid'] = node_feat_np[20:23]
        
            # Normal vector (indices 23:26)
            features['normal'] = node_feat_np[23:26]
        
            # Additional features up to 32 (indices 26:32) - padding/extra
            features['extra_features'] = node_feat_np[26:32]
        
            # For cylinders/cones (handle both regular and SIMPLE variants)
            if self.safe_startswith(features['surface_type'], 'CYLINDER'):
                features['axis'] = features['primary_param']
                features['radius'] = features['secondary_param']
                # Normalize axis
                if np.linalg.norm(features['axis']) > 0:
                    features['axis'] = features['axis'] / np.linalg.norm(features['axis'])
            
            elif self.safe_startswith(features['surface_type'], 'CONE'):
                features['axis'] = features['primary_param']
                features['radius'] = features['secondary_param']
                features['angle'] = features['tertiary_param']
                if np.linalg.norm(features['axis']) > 0:
                    features['axis'] = features['axis'] / np.linalg.norm(features['axis'])
            
            elif self.safe_startswith(features['surface_type'], 'SPHERE'):
                features['radius'] = features['secondary_param']
        
            # Normalize normal
            if np.linalg.norm(features['normal']) > 0:
                features['normal'] = features['normal'] / np.linalg.norm(features['normal'])
        
        elif features['is_edge']:
            # Edge features based on enhanced Phase 2 structure
            features['entity_type'] = 'edge'
        
            # Curve type from one-hot encoding (indices 3:12 for 9 types)
            type_encoding = node_feat_np[3:12]
            if np.any(type_encoding):
                type_idx = np.argmax(type_encoding)
                if type_idx < len(self.curve_types):
                    features['curve_type'] = self.curve_types[type_idx]
                else:
                    features['curve_type'] = 'UNKNOWN_CURVE'
            else:
                features['curve_type'] = 'UNKNOWN_CURVE'
            
            # Set confidence based on type
            if self.safe_startswith(features['curve_type'], 'PROCESSING_ERROR'):
                features['confidence'] = 0.1
            elif '_SIMPLE' in features['curve_type']:
                features['confidence'] = 0.5
            else:
                features['confidence'] = 1.0
        
            # Primary geometric parameter (indices 12:15) - direction/axis
            features['primary_param'] = node_feat_np[12:15]
        
            # Secondary parameter (index 15) - radius/major_radius
            features['secondary_param'] = node_feat_np[15]
        
            # Tertiary parameter (index 16) - minor_radius/degree
            features['tertiary_param'] = node_feat_np[16]
        
            # Length (index 17)
            features['length'] = abs(node_feat_np[17])
        
            # Start point (indices 18:21)
            features['start_point'] = node_feat_np[18:21]
        
            # End point (indices 21:24)
            features['end_point'] = node_feat_np[21:24]
        
            # Direction vector (indices 24:27)
            features['direction'] = node_feat_np[24:27]
        
            # Additional features up to 32 (indices 27:32)
            features['extra_features'] = node_feat_np[27:32]
        
            # For lines (handle both regular and SIMPLE)
            if self.safe_startswith(features['curve_type'], 'LINE'):
                if np.linalg.norm(features['primary_param']) > 0:
                    features['direction'] = features['primary_param']
        
            # For circles (handle both regular and SIMPLE)
            elif self.safe_startswith(features['curve_type'], 'CIRCLE'):
                features['axis'] = features['primary_param']
                features['radius'] = features['secondary_param']
                features['center'] = (features['start_point'] + features['end_point']) / 2
                if np.linalg.norm(features['axis']) > 0:
                    features['axis'] = features['axis'] / np.linalg.norm(features['axis'])
        
            # For ellipses (handle both regular and SIMPLE)
            elif self.safe_startswith(features['curve_type'], 'ELLIPSE'):
                features['axis'] = features['primary_param']
                features['major_radius'] = features['secondary_param']
                features['minor_radius'] = features['tertiary_param']
                features['center'] = (features['start_point'] + features['end_point']) / 2
                if np.linalg.norm(features['axis']) > 0:
                    features['axis'] = features['axis'] / np.linalg.norm(features['axis'])
        
            # For spline curves
            elif features['curve_type'] == 'SPLINE':
                features['degree'] = features['secondary_param']
        
            # Normalize direction
            if np.linalg.norm(features['direction']) > 0:
                features['direction'] = features['direction'] / np.linalg.norm(features['direction'])
            else:
                # Calculate direction from start to end
                if features['length'] > 0:
                    features['direction'] = (features['end_point'] - features['start_point']) / features['length']
                else:
                    features['direction'] = np.array([1, 0, 0])
        
            # Calculate midpoint
            features['midpoint'] = (features['start_point'] + features['end_point']) / 2
        
        elif features['is_vertex']:
            # Vertex features based on enhanced Phase 2 structure
            features['entity_type'] = 'vertex'
            features['confidence'] = 1.0  # Vertices are usually reliable
        
            # Position (indices 3:6)
            features['position'] = node_feat_np[3:6]
        
            # Valency (index 6)
            features['valency'] = node_feat_np[6]
        
            # Normal vector (indices 7:10)
            features['normal'] = node_feat_np[7:10]
        
            # Vertex classification (indices 10:12)
            features['is_boundary'] = node_feat_np[10] > 0.5
            features['is_interior'] = node_feat_np[11] > 0.5
        
            # Density estimates (indices 12:14)
            features['density'] = node_feat_np[12]
            features['weighted_density'] = node_feat_np[13]
        
            # Curvature estimates (indices 14:17)
            features['base_curvature'] = node_feat_np[14]
            features['surface_diversity'] = node_feat_np[15]
            features['edge_diversity'] = node_feat_np[16]
        
            # Seam flag (index 17)
            features['is_seam'] = node_feat_np[17] > 0.5
        
            # Additional features up to 20 (indices 18:20)
            features['extra_features'] = node_feat_np[18:20]
        
            # Normalize normal
            if np.linalg.norm(features['normal']) > 0:
                features['normal'] = features['normal'] / np.linalg.norm(features['normal'])
            else:
                features['normal'] = np.array([0, 0, 1])
        
            # Compute curvature from available features
            features['curvature'] = features['base_curvature']
    
        return features
    
    def detect_face_face_constraints(self, graph_data, src_idx, tgt_idx):
        """Detect face-face constraints with enhanced type handling"""
        constraints = []
        parameters = {}
    
        src_features = self.extract_node_features(graph_data, src_idx)
        tgt_features = self.extract_node_features(graph_data, tgt_idx)
    
        if not (src_features['is_face'] and tgt_features['is_face']):
            return constraints, parameters
    
        # Extract features
        normal1 = src_features['normal']
        normal2 = tgt_features['normal']
        centroid1 = src_features['centroid']
        centroid2 = tgt_features['centroid']
        area1 = src_features['area']
        area2 = tgt_features['area']
        surface_type1 = src_features['surface_type']
        surface_type2 = tgt_features['surface_type']
        confidence = min(src_features['confidence'], tgt_features['confidence'])
    
        # Normalize normals
        if np.linalg.norm(normal1) > 0:
            normal1_norm = normal1 / np.linalg.norm(normal1)
        else:
            normal1_norm = normal1
    
        if np.linalg.norm(normal2) > 0:
            normal2_norm = normal2 / np.linalg.norm(normal2)
        else:
            normal2_norm = normal2
    
        # Calculate geometric relationships
        angle = self.calculate_angle_between_vectors(normal1_norm, normal2_norm)
        distance = self.calculate_distance(centroid1, centroid2)
        dot_product = np.dot(normal1_norm, normal2_norm)
    
        # Face size for normalization
        face_size1 = max(area1 ** 0.5, 1e-6)
        face_size2 = max(area2 ** 0.5, 1e-6)
        avg_face_size = (face_size1 + face_size2) / 2
    
        # 1. PARALLEL_FACES - for any plane variants
        if (self.safe_startswith(surface_type1, 'PLANE') and 
            self.safe_startswith(surface_type2, 'PLANE')):
            
            if abs(angle) < self.angle_tolerance or abs(angle - 180) < self.angle_tolerance:
                constraints.append('PARALLEL_FACES')
                parameters['PARALLEL_FACES'] = {
                    'distance': float(distance / avg_face_size),
                    'angle_deviation': float(min(angle, 180 - angle)),
                    'alignment_dot': float(dot_product),
                    'area_ratio': float(min(area1, area2) / max(area1, area2, 1e-6)),
                    'confidence': float(confidence)
                }
    
        # 2. PERPENDICULAR_FACES - for any plane variants
        if (self.safe_startswith(surface_type1, 'PLANE') and 
            self.safe_startswith(surface_type2, 'PLANE')):
            
            if 90 - self.angle_tolerance < angle < 90 + self.angle_tolerance:
                constraints.append('PERPENDICULAR_FACES')
                parameters['PERPENDICULAR_FACES'] = {
                    'angle': float(angle),
                    'distance_ratio': float(distance / avg_face_size),
                    'alignment_cross': float(np.linalg.norm(np.cross(normal1_norm, normal2_norm))),
                    'area_product': float(area1 * area2),
                    'confidence': float(confidence)
                }
    
        # 3. COPLANAR_FACES - for any plane variants
        if (self.safe_startswith(surface_type1, 'PLANE') and 
            self.safe_startswith(surface_type2, 'PLANE')):
            
            normal_alignment = abs(dot_product)
            relative_distance = distance / avg_face_size
    
            if relative_distance < self.distance_tolerance and normal_alignment > 0.99:
                constraints.append('COPLANAR_FACES')
                parameters['COPLANAR_FACES'] = {
                    'distance': float(distance),
                    'normal_angle': float(angle),
                    'centroid_distance': float(relative_distance),
                    'normal_alignment': float(normal_alignment),
                    'confidence': float(confidence)
                }
    
        # 4. CONCENTRIC_CYLINDERS - for any cylinder variants
        if (self.safe_startswith(surface_type1, 'CYLINDER') and 
            self.safe_startswith(surface_type2, 'CYLINDER')):
            
            # Get cylinder parameters
            axis1 = src_features.get('axis', src_features['primary_param'])
            axis2 = tgt_features.get('axis', tgt_features['primary_param'])
            radius1 = src_features.get('radius', src_features['secondary_param'])
            radius2 = tgt_features.get('radius', tgt_features['secondary_param'])
        
            axis_angle = self.calculate_angle_between_vectors(axis1, axis2)
            center_distance = self.calculate_distance(centroid1, centroid2)
        
            if axis_angle < self.angle_tolerance and center_distance < self.distance_tolerance * avg_face_size:
                constraints.append('CONCENTRIC_CYLINDERS')
                parameters['CONCENTRIC_CYLINDERS'] = {
                    'radius_ratio': float(radius1 / max(radius2, 1e-6)),
                    'axis_angle': float(axis_angle),
                    'center_distance': float(center_distance / avg_face_size),
                    'radius_difference': float(abs(radius1 - radius2) / avg_face_size),
                    'confidence': float(confidence)
                }
    
        # 5. TANGENT_FACES - for plane-cylinder combinations
        if ((self.safe_startswith(surface_type1, 'PLANE') and self.safe_startswith(surface_type2, 'CYLINDER')) or
            (self.safe_startswith(surface_type1, 'CYLINDER') and self.safe_startswith(surface_type2, 'PLANE'))):
        
            # Identify cylinder and plane
            if self.safe_startswith(surface_type1, 'CYLINDER'):
                cylinder_features = src_features
                plane_features = tgt_features
            else:
                cylinder_features = tgt_features
                plane_features = src_features
        
            cylinder_axis = cylinder_features.get('axis', cylinder_features['primary_param'])
            cylinder_radius = cylinder_features.get('radius', cylinder_features['secondary_param'])
            plane_normal = plane_features['normal']
        
            # Angle between cylinder axis and plane normal
            axis_normal_angle = self.calculate_angle_between_vectors(cylinder_axis, plane_normal)
        
            # Check for tangent condition (axis parallel to plane)
            if 90 - self.angle_tolerance < axis_normal_angle < 90 + self.angle_tolerance:
                # Distance from cylinder center to plane
                cylinder_center = cylinder_features['centroid']
                plane_centroid = plane_features['centroid']
                if np.linalg.norm(plane_normal) > 0:
                    plane_normal_norm = plane_normal / np.linalg.norm(plane_normal)
                else:
                    plane_normal_norm = plane_normal
            
                distance_to_plane = abs(np.dot(cylinder_center - plane_centroid, plane_normal_norm))
            
                # Check if distance equals radius (within tolerance)
                if abs(distance_to_plane - cylinder_radius) / max(cylinder_radius, 1e-6) < self.radius_tolerance:
                    constraints.append('TANGENT_FACES')
                    parameters['TANGENT_FACES'] = {
                        'distance_to_plane': float(distance_to_plane),
                        'cylinder_radius': float(cylinder_radius),
                        'axis_normal_angle': float(axis_normal_angle),
                        'radius_error': float(abs(distance_to_plane - cylinder_radius) / cylinder_radius),
                        'confidence': float(confidence)
                    }
    
        return constraints, parameters

    def detect_face_edge_constraints(self, graph_data, src_idx, tgt_idx):
        """Detect face-edge constraints with enhanced type handling"""
        # [Keep your existing method - not repeating for brevity]
        constraints = []
        parameters = {}
    
        # Determine which node is face and which is edge
        src_features = self.extract_node_features(graph_data, src_idx)
        tgt_features = self.extract_node_features(graph_data, tgt_idx)
    
        if src_features['is_face'] and tgt_features['is_edge']:
            face_features = src_features
            edge_features = tgt_features
        elif src_features['is_edge'] and tgt_features['is_face']:
            face_features = tgt_features
            edge_features = src_features
        else:
            return constraints, parameters
    
        # Extract face features
        face_normal = face_features['normal']
        face_centroid = face_features['centroid']
        face_area = face_features['area']
        face_surface_type = face_features['surface_type']
        face_confidence = face_features['confidence']
    
        # Extract edge features
        edge_direction = edge_features['direction']
        edge_length = edge_features['length']
        edge_start = edge_features['start_point']
        edge_end = edge_features['end_point']
        edge_curve_type = edge_features['curve_type']
        edge_confidence = edge_features['confidence']
    
        # Combined confidence
        confidence = min(face_confidence, edge_confidence)
    
        # Calculate midpoint
        edge_midpoint = (edge_start + edge_end) / 2
    
        # Normalize vectors
        if np.linalg.norm(face_normal) > 0:
            face_normal_norm = face_normal / np.linalg.norm(face_normal)
        else:
            face_normal_norm = face_normal
    
        if np.linalg.norm(edge_direction) > 0:
            edge_dir_norm = edge_direction / np.linalg.norm(edge_direction)
        else:
            edge_dir_norm = edge_direction
    
        # Calculate relationships
        angle = self.calculate_angle_between_vectors(face_normal_norm, edge_dir_norm)
        distance_to_centroid = self.calculate_distance(edge_midpoint, face_centroid)
        face_size = max(face_area ** 0.5, 1e-6)
    
        # 1. EDGE_ON_FACE - for any plane variants
        if self.safe_startswith(face_surface_type, 'PLANE'):
            # Distance from edge to face plane
            edge_start_to_face = edge_start - face_centroid
            edge_end_to_face = edge_end - face_centroid
        
            distance_start = abs(np.dot(edge_start_to_face, face_normal_norm))
            distance_end = abs(np.dot(edge_end_to_face, face_normal_norm))
            avg_distance = (distance_start + distance_end) / 2
        
            if avg_distance < self.distance_tolerance * edge_length:
                constraints.append('EDGE_ON_FACE')
            
                # Check if edge is within face bounds (simplified)
                edge_within_face = True  # Simplified check for now
            
                parameters['EDGE_ON_FACE'] = {
                    'avg_distance_to_plane': float(avg_distance / face_size),
                    'edge_length_ratio': float(edge_length / face_size),
                    'angle_to_normal': float(angle),
                    'within_face': float(1.0 if edge_within_face else 0.0),
                    'confidence': float(confidence)
                }
    
        # 2. EDGE_PARALLEL_TO_FACE - for any plane variants
        if self.safe_startswith(face_surface_type, 'PLANE'):
            if abs(angle) < self.angle_tolerance or abs(angle - 180) < self.angle_tolerance:
                constraints.append('EDGE_PARALLEL_TO_FACE')
        
                # Calculate distance from edge to face plane
                edge_mid_to_face = edge_midpoint - face_centroid
                distance_to_plane = abs(np.dot(edge_mid_to_face, face_normal_norm))
        
                parameters['EDGE_PARALLEL_TO_FACE'] = {
                    'angle': float(angle),
                    'distance_to_plane': float(distance_to_plane / face_size),
                    'edge_face_distance': float(distance_to_centroid / face_size),
                    'edge_length_normalized': float(edge_length / face_size),
                    'confidence': float(confidence)
                }
    
        # 3. EDGE_PERPENDICULAR_TO_FACE - for any plane variants
        if self.safe_startswith(face_surface_type, 'PLANE'):
            if 90 - self.angle_tolerance < angle < 90 + self.angle_tolerance:
                constraints.append('EDGE_PERPENDICULAR_TO_FACE')
        
                # Projection of edge onto face normal
                projection_length = edge_length * abs(np.cos(np.radians(angle)))
        
                parameters['EDGE_PERPENDICULAR_TO_FACE'] = {
                    'angle': float(angle),
                    'edge_face_distance': float(distance_to_centroid / face_size),
                    'projection_ratio': float(projection_length / edge_length if edge_length > 0 else 0),
                    'normal_alignment': float(abs(np.dot(edge_dir_norm, face_normal_norm))),
                    'confidence': float(confidence)
                }
    
        # 4. EDGE_TANGENT_TO_FACE - for cylindrical faces and line edges
        if (self.safe_startswith(face_surface_type, 'CYLINDER') and 
            self.safe_startswith(edge_curve_type, 'LINE')):
            
            cylinder_axis = face_features.get('axis', face_features['primary_param'])
            cylinder_radius = face_features.get('radius', face_features['secondary_param'])
        
            # Check if edge is tangent to cylinder
            edge_to_center = edge_midpoint - face_centroid
            if np.linalg.norm(edge_to_center) > 0:
                radius_vector = edge_to_center / np.linalg.norm(edge_to_center)
            
                # Angle between edge direction and radius vector should be 90°
                tangent_angle = self.calculate_angle_between_vectors(edge_dir_norm, radius_vector)
            
                if 90 - self.angle_tolerance < tangent_angle < 90 + self.angle_tolerance:
                    # Distance from edge to cylinder surface should be radius
                    distance_to_surface = abs(np.linalg.norm(edge_to_center) - cylinder_radius)
                
                    if distance_to_surface < self.distance_tolerance * cylinder_radius:
                        constraints.append('EDGE_TANGENT_TO_FACE')
                        parameters['EDGE_TANGENT_TO_FACE'] = {
                            'tangent_angle': float(tangent_angle),
                            'distance_to_surface': float(distance_to_surface / cylinder_radius),
                            'cylinder_radius': float(cylinder_radius),
                            'edge_cylinder_distance': float(np.linalg.norm(edge_to_center) / cylinder_radius),
                            'confidence': float(confidence)
                        }
    
        return constraints, parameters

    def detect_edge_edge_constraints(self, graph_data, src_idx, tgt_idx):
        """Detect edge-edge constraints with enhanced type handling"""
        # [Keep your existing method - not repeating for brevity]
        constraints = []
        parameters = {}
    
        src_features = self.extract_node_features(graph_data, src_idx)
        tgt_features = self.extract_node_features(graph_data, tgt_idx)
    
        if not (src_features['is_edge'] and tgt_features['is_edge']):
            return constraints, parameters
    
        # Extract edge features
        dir1 = src_features['direction']
        dir2 = tgt_features['direction']
        len1 = src_features['length']
        len2 = tgt_features['length']
        start1 = src_features['start_point']
        end1 = src_features['end_point']
        start2 = tgt_features['start_point']
        end2 = tgt_features['end_point']
        curve_type1 = src_features['curve_type']
        curve_type2 = tgt_features['curve_type']
        confidence = min(src_features['confidence'], tgt_features['confidence'])
    
        # Calculate midpoints
        midpoint1 = (start1 + end1) / 2
        midpoint2 = (start2 + end2) / 2
    
        # Normalize directions
        if np.linalg.norm(dir1) > 0:
            dir1_norm = dir1 / np.linalg.norm(dir1)
        else:
            # Calculate direction from start to end
            if len1 > 0:
                dir1_norm = (end1 - start1) / len1
            else:
                dir1_norm = np.array([1, 0, 0])
    
        if np.linalg.norm(dir2) > 0:
            dir2_norm = dir2 / np.linalg.norm(dir2)
        else:
            # Calculate direction from start to end
            if len2 > 0:
                dir2_norm = (end2 - start2) / len2
            else:
                dir2_norm = np.array([1, 0, 0])
    
        # Calculate relationships
        angle = self.calculate_angle_between_vectors(dir1_norm, dir2_norm)
        distance = self.calculate_distance(midpoint1, midpoint2)
        avg_length = (len1 + len2) / 2
    
        # 1. PARALLEL_EDGES - for any line variants
        if (self.safe_startswith(curve_type1, 'LINE') and 
            self.safe_startswith(curve_type2, 'LINE')):
            
            if abs(angle) < self.angle_tolerance or abs(angle - 180) < self.angle_tolerance:
                constraints.append('PARALLEL_EDGES')
        
                # Calculate minimum distance between parallel lines
                v = start2 - start1
                cross_product = np.cross(dir1_norm, dir2_norm)
        
                if np.linalg.norm(cross_product) > 0:
                    min_distance = abs(np.dot(v, cross_product)) / np.linalg.norm(cross_product)
                else:
                    # Lines are parallel or coincident
                    min_distance = self.calculate_distance(start1, start2)
        
                parameters['PARALLEL_EDGES'] = {
                    'distance': float(min_distance / avg_length if avg_length > 0 else 0),
                    'angle_deviation': float(min(angle, 180 - angle)),
                    'length_ratio': float(min(len1, len2) / max(len1, len2, 1e-6)),
                    'midpoint_distance': float(distance / avg_length if avg_length > 0 else 0),
                    'confidence': float(confidence)
                }
    
        # 2. PERPENDICULAR_EDGES - for any line variants
        if (self.safe_startswith(curve_type1, 'LINE') and 
            self.safe_startswith(curve_type2, 'LINE')):
            
            if 90 - self.angle_tolerance < angle < 90 + self.angle_tolerance:
                constraints.append('PERPENDICULAR_EDGES')
        
                # Calculate minimum distance between endpoints
                endpoint_distances = [
                    self.calculate_distance(start1, start2),
                    self.calculate_distance(start1, end2),
                    self.calculate_distance(end1, start2),
                    self.calculate_distance(end1, end2)
                ]
                min_endpoint_distance = min(endpoint_distances)
        
                parameters['PERPENDICULAR_EDGES'] = {
                    'angle': float(angle),
                    'min_distance': float(min_endpoint_distance / avg_length if avg_length > 0 else 0),
                    'length_product': float(len1 * len2),
                    'dot_product': float(np.dot(dir1_norm, dir2_norm)),
                    'confidence': float(confidence)
                }
    
        # 3. COLLINEAR_EDGES - for any line variants
        if (self.safe_startswith(curve_type1, 'LINE') and 
            self.safe_startswith(curve_type2, 'LINE')):
            
            if angle < self.angle_tolerance:
                # Check if edges lie on same line
                line1_dir = dir1_norm
                proj_start2 = np.dot(start2 - start1, line1_dir)
                proj_end2 = np.dot(end2 - start1, line1_dir)
        
                # Check for overlap
                min_proj = min(proj_start2, proj_end2)
                max_proj = max(proj_start2, proj_end2)
            
                overlaps = (0 <= min_proj <= len1) or (0 <= max_proj <= len1) or \
                          (min_proj <= 0 <= max_proj) or (min_proj <= len1 <= max_proj)
        
                if overlaps:
                    constraints.append('COLLINEAR_EDGES')
            
                    # Calculate overlap length
                    overlap_start = max(0, min_proj)
                    overlap_end = min(len1, max_proj)
                    overlap_length = max(0, overlap_end - overlap_start)
            
                    parameters['COLLINEAR_EDGES'] = {
                        'overlap_ratio': float(overlap_length / max(len1, len2, 1e-6)),
                        'angle_deviation': float(angle),
                        'distance_between': float(distance / avg_length if avg_length > 0 else 0),
                        'alignment': float(np.dot(dir1_norm, dir2_norm)),
                        'confidence': float(confidence)
                    }
    
        # 4. EQUAL_LENGTH - for any edge types
        length_ratio = min(len1, len2) / max(len1, len2, 1e-6)
        if length_ratio > 1 - self.length_tolerance:
            constraints.append('EQUAL_LENGTH')
            parameters['EQUAL_LENGTH'] = {
                'length_ratio': float(length_ratio),
                'length_difference': float(abs(len1 - len2) / avg_length if avg_length > 0 else 0),
                'tolerance_ratio': float((1 - length_ratio) / self.length_tolerance),
                'avg_length': float(avg_length),
                'confidence': float(confidence)
            }
    
        # 5. EQUAL_RADIUS - for any circle variants
        if (self.safe_startswith(curve_type1, 'CIRCLE') and 
            self.safe_startswith(curve_type2, 'CIRCLE')):
            
            # Get radii
            if self.safe_startswith(curve_type1, 'CIRCLE'):
                radius1 = src_features.get('radius', src_features['secondary_param'])
            else:
                radius1 = src_features['secondary_param']
        
            if self.safe_startswith(curve_type2, 'CIRCLE'):
                radius2 = tgt_features.get('radius', tgt_features['secondary_param'])
            else:
                radius2 = tgt_features['secondary_param']
        
            if radius1 > 0 and radius2 > 0:
                radius_ratio = min(radius1, radius2) / max(radius1, radius2)
            
                if radius_ratio > 1 - self.radius_tolerance:
                    constraints.append('EQUAL_RADIUS')
                    parameters['EQUAL_RADIUS'] = {
                        'radius_ratio': float(radius_ratio),
                        'radius_difference': float(abs(radius1 - radius2) / ((radius1 + radius2) / 2)),
                        'tolerance_ratio': float((1 - radius_ratio) / self.radius_tolerance),
                        'avg_radius': float((radius1 + radius2) / 2),
                        'confidence': float(confidence)
                    }
    
        # 6. CONCENTRIC_CIRCLES - for any circle variants
        if (self.safe_startswith(curve_type1, 'CIRCLE') and 
            self.safe_startswith(curve_type2, 'CIRCLE')):
            
            # Get circle centers
            center1 = src_features.get('center', midpoint1)
            center2 = tgt_features.get('center', midpoint2)
        
            center_distance = self.calculate_distance(center1, center2)
            
            # Get radii
            if self.safe_startswith(curve_type1, 'CIRCLE'):
                radius1 = src_features.get('radius', src_features['secondary_param'])
            else:
                radius1 = src_features['secondary_param']
        
            if self.safe_startswith(curve_type2, 'CIRCLE'):
                radius2 = tgt_features.get('radius', tgt_features['secondary_param'])
            else:
                radius2 = tgt_features['secondary_param']
            
            avg_radius = (radius1 + radius2) / 2
        
            if center_distance < self.distance_tolerance * avg_radius:
                constraints.append('CONCENTRIC_CIRCLES')
                parameters['CONCENTRIC_CIRCLES'] = {
                    'center_distance': float(center_distance / avg_radius if avg_radius > 0 else 0),
                    'radius_ratio': float(radius1 / max(radius2, 1e-6)),
                    'plane_angle': float(angle),  # Assuming circles in same plane
                    'avg_radius': float(avg_radius),
                    'confidence': float(confidence)
                }
    
        return constraints, parameters

    def detect_vertex_constraints(self, graph_data, src_idx, tgt_idx):
        """Detect vertex-related constraints with enhanced type handling"""
        # [Keep your existing method - not repeating for brevity]
        constraints = []
        parameters = {}
    
        src_features = self.extract_node_features(graph_data, src_idx)
        tgt_features = self.extract_node_features(graph_data, tgt_idx)
        
        # Determine entity types
        src_is_vertex = src_features['is_vertex']
        tgt_is_vertex = tgt_features['is_vertex']
        src_is_face = src_features['is_face']
        tgt_is_face = tgt_features['is_face']
        src_is_edge = src_features['is_edge']
        tgt_is_edge = tgt_features['is_edge']
    
        # 1. COINCIDENT_VERTICES
        if src_is_vertex and tgt_is_vertex:
            pos1 = src_features['position']
            pos2 = tgt_features['position']
            distance = self.calculate_distance(pos1, pos2)
            confidence = min(src_features['confidence'], tgt_features['confidence'])
        
            if distance < self.distance_tolerance:
                constraints.append('COINCIDENT_VERTICES')
            
                # Get vertex properties
                valency1 = src_features['valency']
                valency2 = tgt_features['valency']
                normal1 = src_features['normal']
                normal2 = tgt_features['normal']
                is_boundary1 = src_features['is_boundary']
                is_boundary2 = tgt_features['is_boundary']
                curvature1 = src_features['curvature']
                curvature2 = tgt_features['curvature']
            
                # Calculate normal alignment
                if np.linalg.norm(normal1) > 0 and np.linalg.norm(normal2) > 0:
                    normal1_norm = normal1 / np.linalg.norm(normal1)
                    normal2_norm = normal2 / np.linalg.norm(normal2)
                    normal_angle = self.calculate_angle_between_vectors(normal1_norm, normal2_norm)
                    normal_dot = np.dot(normal1_norm, normal2_norm)
                else:
                    normal_angle = 0
                    normal_dot = 0
                
                parameters['COINCIDENT_VERTICES'] = {
                    'distance': float(distance),
                    'valence_difference': float(abs(valency1 - valency2)),
                    'normal_angle': float(normal_angle),
                    'normal_alignment': float(normal_dot),
                    'connectivity_ratio': float(min(valency1, valency2) / max(valency1, valency2, 1)),
                    'boundary_match': float(1.0 if is_boundary1 == is_boundary2 else 0.0),
                    'curvature_difference': float(abs(curvature1 - curvature2)),
                    'confidence': float(confidence)
                }
    
        # 2. VERTEX_ON_FACE
        elif (src_is_vertex and tgt_is_face) or (src_is_face and tgt_is_vertex):
            if src_is_vertex:
                vertex_features = src_features
                face_features = tgt_features
            else:
                vertex_features = tgt_features
                face_features = src_features
        
            vertex_pos = vertex_features['position']
            face_centroid = face_features['centroid']
            face_normal = face_features['normal']
            face_area = face_features['area']
            face_surface_type = face_features['surface_type']
            confidence = min(vertex_features['confidence'], face_features['confidence'])
        
            # Distance from vertex to face plane
            vertex_to_centroid = vertex_pos - face_centroid
        
            # Normalize face normal
            if np.linalg.norm(face_normal) > 0:
                face_normal_norm = face_normal / np.linalg.norm(face_normal)
            else:
                face_normal_norm = face_normal
        
            # Projection onto face normal
            distance_to_plane = abs(np.dot(vertex_to_centroid, face_normal_norm))
        
            # Perpendicular distance to centroid
            perpendicular_distance = self.calculate_distance(vertex_pos, face_centroid)
        
            face_size = max(face_area ** 0.5, 1e-6)
        
            # Check if vertex is on face (for any plane variants)
            if self.safe_startswith(face_surface_type, 'PLANE') and distance_to_plane < self.distance_tolerance * face_size:
                constraints.append('VERTEX_ON_FACE')
            
                # Get vertex properties
                vertex_normal = vertex_features['normal']
                vertex_valency = vertex_features['valency']
                vertex_curvature = vertex_features['curvature']
            
                # Normalize vertex normal
                if np.linalg.norm(vertex_normal) > 0:
                    vertex_normal_norm = vertex_normal / np.linalg.norm(vertex_normal)
                else:
                    vertex_normal_norm = vertex_normal
            
                # Calculate angles
                vertex_to_face_angle = self.calculate_angle_between_vectors(vertex_to_centroid, face_normal_norm)
                normal_angle = self.calculate_angle_between_vectors(vertex_normal_norm, face_normal_norm)
            
                parameters['VERTEX_ON_FACE'] = {
                    'distance_to_plane': float(distance_to_plane / face_size),
                    'perpendicular_distance': float(perpendicular_distance / face_size),
                    'vertex_to_face_angle': float(vertex_to_face_angle),
                    'normal_alignment_angle': float(normal_angle),
                    'normal_alignment': float(np.dot(vertex_normal_norm, face_normal_norm)),
                    'face_area': float(face_area),
                    'vertex_valency': float(vertex_valency),
                    'vertex_curvature': float(vertex_curvature),
                    'face_surface_type_match': float(1.0 if self.safe_startswith(face_surface_type, 'PLANE') else 0.5),
                    'confidence': float(confidence)
                }
    
        # 3. VERTEX_ON_EDGE
        elif (src_is_vertex and tgt_is_edge) or (src_is_edge and tgt_is_vertex):
            if src_is_vertex:
                vertex_features = src_features
                edge_features = tgt_features
            else:
                vertex_features = tgt_features
                edge_features = src_features
            
            vertex_pos = vertex_features['position']
            edge_start = edge_features['start_point']
            edge_end = edge_features['end_point']
            edge_length = edge_features['length']
            edge_curve_type = edge_features['curve_type']
            confidence = min(vertex_features['confidence'], edge_features['confidence'])
        
            # Calculate distance from vertex to edge line
            edge_dir = edge_features['direction']
        
            if np.linalg.norm(edge_dir) == 0:
                # Calculate direction from start to end
                edge_dir = (edge_end - edge_start) / edge_length if edge_length > 0 else np.array([1, 0, 0])
        
            # Normalize edge direction
            if np.linalg.norm(edge_dir) > 0:
                edge_dir_norm = edge_dir / np.linalg.norm(edge_dir)
            else:
                edge_dir_norm = edge_dir
        
            vertex_vec = vertex_pos - edge_start
            projection = np.dot(vertex_vec, edge_dir_norm)
        
            # Closest point on edge
            if projection < 0:
                closest_point = edge_start
                position_ratio = 0.0
            elif projection > edge_length:
                closest_point = edge_end
                position_ratio = 1.0
            else:
                closest_point = edge_start + projection * edge_dir_norm
                position_ratio = projection / edge_length if edge_length > 0 else 0.0
        
            distance = self.calculate_distance(vertex_pos, closest_point)
        
            if distance < self.distance_tolerance * edge_length:
                constraints.append('VERTEX_ON_EDGE')
            
                # Get vertex properties
                vertex_valency = vertex_features['valency']
                vertex_normal = vertex_features['normal']
                vertex_is_boundary = vertex_features['is_boundary']
            
                # Get edge properties
                edge_radius = edge_features.get('radius', edge_features['secondary_param'])
                
                # Normalize vertex normal
                if np.linalg.norm(vertex_normal) > 0:
                    vertex_normal_norm = vertex_normal / np.linalg.norm(vertex_normal)
                else:
                    vertex_normal_norm = vertex_normal
            
                # Calculate angle between vertex normal and edge direction
                normal_edge_angle = self.calculate_angle_between_vectors(vertex_normal_norm, edge_dir_norm)
                
                parameters['VERTEX_ON_EDGE'] = {
                    'distance_to_edge': float(distance / edge_length if edge_length > 0 else 0),
                    'position_ratio': float(position_ratio),
                    'edge_length': float(edge_length),
                    'vertex_valency': float(vertex_valency),
                    'normal_edge_angle': float(normal_edge_angle),
                    'normal_edge_alignment': float(np.dot(vertex_normal_norm, edge_dir_norm)),
                    'vertex_is_boundary': float(1.0 if vertex_is_boundary else 0.0),
                    'edge_curve_type_match': float(1.0 if self.safe_startswith(edge_curve_type, 'LINE') else 0.5),
                    'edge_radius': float(edge_radius),
                    'confidence': float(confidence)
                }
    
        return constraints, parameters
    
    def detect_pattern_constraints(self, graph_data):
        """Detect pattern constraints (requires analyzing multiple edges)"""
        constraints = []
        parameters = {}
        
        # Get all edges
        edge_index = graph_data.edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges < 3:  # Need at least 3 edges for patterns
            return constraints, parameters
        
        # Group edges by type and collect their geometric properties
        parallel_groups = []
        
        # First pass: identify parallel edge groups (only line edges)
        for i in range(num_edges):
            src_i = edge_index[0, i].item()
            tgt_i = edge_index[1, i].item()
            
            src_features = self.extract_node_features(graph_data, src_i)
            tgt_features = self.extract_node_features(graph_data, tgt_i)
            
            # Only consider line edges for parallel patterns
            if (src_features['is_edge'] and tgt_features['is_edge'] and
                self.safe_startswith(src_features['curve_type'], 'LINE') and
                self.safe_startswith(tgt_features['curve_type'], 'LINE')):
                
                # Get direction from source node
                dir_i = src_features['direction']
                len_i = src_features['length']
                mid_i = (src_features['start_point'] + src_features['end_point']) / 2
                
                added = False
                for group in parallel_groups:
                    # Check first edge in group
                    sample_idx = group[0]
                    src_sample = edge_index[0, sample_idx].item()
                    src_sample_features = self.extract_node_features(graph_data, src_sample)
                    
                    dir_sample = src_sample_features['direction']
                    angle = self.calculate_angle_between_vectors(dir_i, dir_sample)
                    
                    if angle < self.angle_tolerance:
                        group.append(i)
                        added = True
                        break
                
                if not added:
                    parallel_groups.append([i])
        
        # Analyze parallel groups for patterns
        for group in parallel_groups:
            if len(group) >= 3:  # Need at least 3 parallel edges for spacing analysis
                # Extract midpoints and lengths
                midpoints = []
                lengths = []
                
                for edge_idx in group:
                    src_idx = edge_index[0, edge_idx].item()
                    src_features = self.extract_node_features(graph_data, src_idx)
                    midpoints.append((src_features['start_point'] + src_features['end_point']) / 2)
                    lengths.append(src_features['length'])
                
                midpoints = np.array(midpoints)
                lengths = np.array(lengths)
                
                # Project midpoints onto perpendicular direction
                # Find dominant direction of group (average of edge directions)
                group_dirs = []
                for edge_idx in group:
                    src_idx = edge_index[0, edge_idx].item()
                    src_features = self.extract_node_features(graph_data, src_idx)
                    group_dirs.append(src_features['direction'])
                
                avg_dir = np.mean(group_dirs, axis=0)
                if np.linalg.norm(avg_dir) > 0:
                    avg_dir = avg_dir / np.linalg.norm(avg_dir)
                
                # Find perpendicular direction (simplified)
                if abs(avg_dir[0]) > 0.5:
                    perp_dir = np.array([0, 1, 0])
                else:
                    perp_dir = np.array([1, 0, 0])
                
                # Project onto perpendicular direction
                projections = np.array([np.dot(mid - midpoints[0], perp_dir) for mid in midpoints])
                projections.sort()
                
                # Calculate spacings
                spacings = np.diff(projections)
                
                if len(spacings) >= 2:
                    spacing_std = np.std(spacings)
                    spacing_mean = np.mean(spacings)
                    
                    # Check for equal spacing
                    if spacing_mean > 0 and spacing_std / spacing_mean < 0.1:  # 10% variation tolerance
                        constraints.append('EQUAL_SPACING')
                        parameters['EQUAL_SPACING'] = {
                            'spacing_std': float(spacing_std),
                            'spacing_mean': float(spacing_mean),
                            'count': len(group),
                            'regularity': float(1.0 - spacing_std / spacing_mean),
                            'confidence': 0.8  # Pattern detection is inherently lower confidence
                        }
        
        return constraints, parameters
    
    def detect_constraints_for_edge(self, graph_data, edge_idx):
        """Detect all constraints for a single edge with parameters"""
        edge_index = graph_data.edge_index
        src_idx = edge_index[0, edge_idx].item()
        tgt_idx = edge_index[1, edge_idx].item()
        
        all_constraints = []
        all_parameters = {}
        
        # Get node types
        src_features = self.extract_node_features(graph_data, src_idx)
        tgt_features = self.extract_node_features(graph_data, tgt_idx)
        
        # Determine appropriate constraint detectors based on node types
        src_is_face = src_features['is_face']
        tgt_is_face = tgt_features['is_face']
        src_is_edge = src_features['is_edge']
        tgt_is_edge = tgt_features['is_edge']
        src_is_vertex = src_features['is_vertex']
        tgt_is_vertex = tgt_features['is_vertex']
        
        # Face-Face constraints
        if src_is_face and tgt_is_face:
            constraints, params = self.detect_face_face_constraints(graph_data, src_idx, tgt_idx)
            all_constraints.extend(constraints)
            all_parameters.update(params)
        
        # Face-Edge constraints
        elif (src_is_face and tgt_is_edge) or (src_is_edge and tgt_is_face):
            constraints, params = self.detect_face_edge_constraints(graph_data, src_idx, tgt_idx)
            all_constraints.extend(constraints)
            all_parameters.update(params)
        
        # Edge-Edge constraints
        elif src_is_edge and tgt_is_edge:
            constraints, params = self.detect_edge_edge_constraints(graph_data, src_idx, tgt_idx)
            all_constraints.extend(constraints)
            all_parameters.update(params)
        
        # Vertex constraints (any combination with vertex)
        if src_is_vertex or tgt_is_vertex:
            constraints, params = self.detect_vertex_constraints(graph_data, src_idx, tgt_idx)
            all_constraints.extend(constraints)
            all_parameters.update(params)
        
        return all_constraints, all_parameters
    
    def create_labeled_graph_with_parameters(self, graph_data):
        """Create labeled graph with constraint labels AND parameters
        NOW WITH EXPLICIT NO_CONSTRAINT CLASS"""
        if not hasattr(graph_data, 'edge_index'):
            raise ValueError("Graph data must have edge_index attribute")
        
        num_edges = graph_data.edge_index.shape[1]
        
        # Initialize constraint labels (binary matrix) - now with NO_CONSTRAINT as index 0
        constraint_labels = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
        
        # Initialize ALL tensors with explicit float32 dtype - NEVER None
        # CRITICAL: These must ALWAYS be tensors, even if all zeros
        constraint_params_tensor = torch.zeros((num_edges, self.num_constraint_types, self.max_params_per_constraint), 
                                       dtype=torch.float32)

        # Initialize constraint mask (where constraints exist)
        constraint_mask = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)

        # Initialize confidence tensor
        confidence_tensor = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)

        # Track which edges have any constraint detected
        edges_with_constraints = set()

        # Store detailed parameters dictionary
        constraint_params_dict = {}
        edge_constraints_list = []
        
        # Detect constraints for each edge
        for edge_idx in range(num_edges):
            constraints, parameters = self.detect_constraints_for_edge(graph_data, edge_idx)
            edge_constraints_list.append(constraints)
            
            # Track edges that have valid constraints
            valid_constraints = []
            
            # Update binary labels for detected constraints
            for constraint in constraints:
                if constraint in self.constraint_to_idx:
                    constraint_idx = self.constraint_to_idx[constraint]
                    
                    # Check confidence threshold
                    confidence = parameters[constraint].get('confidence', 0.5)
                    if confidence >= self.min_confidence_threshold:
                        constraint_labels[edge_idx, constraint_idx] = 1.0
                        constraint_mask[edge_idx, constraint_idx] = 1.0
                        confidence_tensor[edge_idx, constraint_idx] = confidence
                        valid_constraints.append(constraint)
                        edges_with_constraints.add(edge_idx)
                        
                        # Store parameters
                        if constraint not in constraint_params_dict:
                            constraint_params_dict[constraint] = []
                        constraint_params_dict[constraint].append({
                            'edge_id': edge_idx,
                            'parameters': parameters[constraint]
                        })
                        
                        # Store in tensor (pad to max_params)
                        param_values = [v for k, v in parameters[constraint].items() if k != 'confidence']
                        num_params = min(len(param_values), self.max_params_per_constraint)
                        constraint_params_tensor[edge_idx, constraint_idx, :num_params] = torch.tensor(
                            param_values[:num_params], dtype=torch.float32
                        )
            
            # Log detected constraints for debugging
            if valid_constraints:
                print(f"Edge {edge_idx}: Detected {valid_constraints}")
        
        # CRITICAL: Mark edges with NO_CONSTRAINT
        # All edges that don't have any detected constraint get NO_CONSTRAINT = 1
        no_constraint_idx = self.constraint_to_idx['NO_CONSTRAINT']

        # Ensure NO_CONSTRAINT exists in params dict
        if 'NO_CONSTRAINT' not in constraint_params_dict:
            constraint_params_dict['NO_CONSTRAINT'] = []

        for edge_idx in range(num_edges):
            if edge_idx not in edges_with_constraints:
                # Set NO_CONSTRAINT label
                constraint_labels[edge_idx, no_constraint_idx] = 1.0
                constraint_mask[edge_idx, no_constraint_idx] = 1.0
                confidence_tensor[edge_idx, no_constraint_idx] = 1.0  # High confidence for no constraint
        
                # Calculate NO_CONSTRAINT parameters based on edge type
                src_idx = graph_data.edge_index[0, edge_idx].item()
                tgt_idx = graph_data.edge_index[1, edge_idx].item()
                src_features = self.extract_node_features(graph_data, src_idx)
                tgt_features = self.extract_node_features(graph_data, tgt_idx)
        
                # Determine if this is a topological or geometric relationship
                is_topological = 1.0  # Default to topological
                is_geometric = 0.0
        
                # Check if this edge connects different entity types
                if src_features['entity_type'] != tgt_features['entity_type']:
                    is_topological = 1.0  # Cross-type edges are topological
                else:
                    # Same-type edges might be geometric
                    is_geometric = 1.0
                    is_topological = 0.0
        
                no_constraint_params = {
                    'confidence': 1.0,
                    'is_topological': is_topological,
                    'is_geometric': is_geometric
                }
        
                constraint_params_dict['NO_CONSTRAINT'].append({
                    'edge_id': edge_idx,
                    'parameters': no_constraint_params
                })
        
                # ALWAYS set tensor values - even if they're zeros
                # This ensures the tensor is properly populated
                param_values = [
                    no_constraint_params['confidence'],
                    no_constraint_params['is_topological'],
                    no_constraint_params['is_geometric'],
                    0.0  # Pad to max_params_per_constraint (4)
                ]
        
                # Convert to tensor and assign - ensure we always assign exactly 4 values
                param_tensor = torch.tensor(param_values[:self.max_params_per_constraint], dtype=torch.float32)
                constraint_params_tensor[edge_idx, no_constraint_idx, :] = param_tensor
        
            else:
                # Edge has other constraints - ensure NO_CONSTRAINT is explicitly 0
                constraint_labels[edge_idx, no_constraint_idx] = 0.0
                constraint_mask[edge_idx, no_constraint_idx] = 0.0
                confidence_tensor[edge_idx, no_constraint_idx] = 0.0
        
                # NO_CONSTRAINT parameters remain zeros for this edge
                # (already initialized to zeros)
        
        # Detect pattern constraints (global, not per-edge)
        pattern_constraints, pattern_params = self.detect_pattern_constraints(graph_data)
        
        # Add pattern constraints to first edge (as anchor)
        if pattern_constraints and num_edges > 0:
            for constraint in pattern_constraints:
                if constraint in self.constraint_to_idx:
                    constraint_idx = self.constraint_to_idx[constraint]
                    constraint_labels[0, constraint_idx] = 1.0  # Assign to first edge
                    constraint_mask[0, constraint_idx] = 1.0
                    edges_with_constraints.add(0)  # Mark as having constraint
                    
                    # Remove NO_CONSTRAINT from this edge if it was set
                    constraint_labels[0, no_constraint_idx] = 0.0
                    constraint_mask[0, no_constraint_idx] = 0.0
                    
                    if constraint in pattern_params:
                        confidence_tensor[0, constraint_idx] = pattern_params[constraint].get('confidence', 0.8)
                        
                        param_values = [v for k, v in pattern_params[constraint].items() if k != 'confidence']
                        num_params = min(len(param_values), self.max_params_per_constraint)
                        constraint_params_tensor[0, constraint_idx, :num_params] = torch.tensor(
                            param_values[:num_params], dtype=torch.float32
                        )
        
        # CRITICAL: Add all data to graph - ensure NO TENSOR IS EVER None
        graph_data.y = constraint_labels
        graph_data.edge_constraints = edge_constraints_list
        graph_data.constraint_params = constraint_params_dict

        # These must ALWAYS be tensors - NEVER None
        graph_data.constraint_params_tensor = constraint_params_tensor
        graph_data.constraint_mask = constraint_mask
        graph_data.constraint_confidence = confidence_tensor

        # DOUBLE CHECK: Verify all required tensors exist and are proper type
        assert graph_data.constraint_params_tensor is not None, "constraint_params_tensor cannot be None"
        assert graph_data.constraint_mask is not None, "constraint_mask cannot be None"
        assert graph_data.constraint_confidence is not None, "constraint_confidence cannot be None"
        assert isinstance(graph_data.constraint_params_tensor, torch.Tensor), "constraint_params_tensor must be a tensor"
        assert isinstance(graph_data.constraint_mask, torch.Tensor), "constraint_mask must be a tensor"
        assert isinstance(graph_data.constraint_confidence, torch.Tensor), "constraint_confidence must be a tensor"
        
        # Store constraint definitions for reference
        graph_data.constraint_definitions = self.constraint_definitions
        graph_data.constraint_names = self.constraint_names
        
        # Print statistics for debugging
        edges_with_constraints_count = len(edges_with_constraints)
        no_constraint_count = num_edges - edges_with_constraints_count
        print(f"\nConstraint statistics for graph:")
        print(f"  Total edges: {num_edges}")
        print(f"  Edges with constraints: {edges_with_constraints_count} ({edges_with_constraints_count/num_edges*100:.1f}%)")
        print(f"  Edges with NO_CONSTRAINT: {no_constraint_count} ({no_constraint_count/num_edges*100:.1f}%)")
        # FINAL VERIFICATION: Ensure all tensors are properly initialized
        # This is critical for GNN training compatibility
        print(f"\nTensor verification for graph:")
        print(f"  constraint_params_tensor: shape={graph_data.constraint_params_tensor.shape},    dtype={graph_data.constraint_params_tensor.dtype}")
        print(f"  constraint_mask: shape={graph_data.constraint_mask.shape}, dtype={graph_data.constraint_mask.dtype}")
        print(f"  constraint_confidence: shape={graph_data.constraint_confidence.shape}, dtype={graph_data.constraint_confidence.dtype}")
        print(f"  y: shape={graph_data.y.shape}, dtype={graph_data.y.dtype}")

        # Verify no NaN or Inf values
        if torch.isnan(graph_data.constraint_params_tensor).any():
            print("  WARNING: NaN values found in constraint_params_tensor - replacing with zeros")
            graph_data.constraint_params_tensor = torch.nan_to_num(graph_data.constraint_params_tensor, nan=0.0)
        if torch.isinf(graph_data.constraint_params_tensor).any():
            print("  WARNING: Inf values found in constraint_params_tensor - replacing with zeros")
            graph_data.constraint_params_tensor = torch.nan_to_num(graph_data.constraint_params_tensor, posinf=0.0, neginf=0.0)
        
        return graph_data
    
    def process_single_graph(self, graph_file_path, output_dir):
            """Process a single graph file and save with constraints and parameters"""
            try:
                print(f"Processing: {graph_file_path}")
        
                # Load graph
                graph_data = torch.load(graph_file_path, weights_only=False)
        
                # Detect constraints with parameters
                labeled_graph = self.create_labeled_graph_with_parameters(graph_data)
        
                # ====================================================================
                # CRITICAL: Verify and fix all tensors before saving
                # ====================================================================
        
                # Required attributes that must be tensors (never None)
                required_tensor_attrs = [
                    'y', 
                    'constraint_params_tensor', 
                    'constraint_mask', 
                    'constraint_confidence'
                ]
        
                # Get number of edges
                num_edges = labeled_graph.edge_index.shape[1]
        
                # Check for missing attributes
                missing_attrs = [attr for attr in required_tensor_attrs if not hasattr(labeled_graph, attr)]
        
                if missing_attrs:
                    print(f"  WARNING: Missing attributes: {missing_attrs}")
            
                    # Add missing attributes with proper tensors
                    for attr in missing_attrs:
                        if attr == 'constraint_params_tensor':
                            labeled_graph.constraint_params_tensor = torch.zeros(
                                (num_edges, self.num_constraint_types, self.max_params_per_constraint),
                                dtype=torch.float32
                            )
                            print(f"    Added missing constraint_params_tensor with shape {labeled_graph.constraint_params_tensor.shape}")
                    
                        elif attr == 'constraint_mask':
                            labeled_graph.constraint_mask = torch.zeros(
                                (num_edges, self.num_constraint_types),
                                dtype=torch.float32
                            )
                            # Set mask for NO_CONSTRAINT (index 0)
                            if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                labeled_graph.constraint_mask[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            print(f"    Added missing constraint_mask with shape {labeled_graph.constraint_mask.shape}")
                    
                        elif attr == 'constraint_confidence':
                            labeled_graph.constraint_confidence = torch.zeros(
                                (num_edges, self.num_constraint_types),
                                dtype=torch.float32
                            )
                            # Set confidence for NO_CONSTRAINT
                            if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                labeled_graph.constraint_confidence[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            print(f"    Added missing constraint_confidence with shape {labeled_graph.constraint_confidence.shape}")
                    
                        elif attr == 'y':
                            labeled_graph.y = torch.zeros(
                                (num_edges, self.num_constraint_types),
                                dtype=torch.float32
                            )
                            # Set NO_CONSTRAINT for all edges if no labels exist
                            if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                labeled_graph.y[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            print(f"    Added missing y with shape {labeled_graph.y.shape}")
        
                # Check for None values in existing attributes
                for attr in required_tensor_attrs:
                    if hasattr(labeled_graph, attr) and getattr(labeled_graph, attr) is None:
                        print(f"  WARNING: {attr} is None - recreating")
                
                        if attr == 'constraint_params_tensor':
                            setattr(labeled_graph, attr, torch.zeros(
                                (num_edges, self.num_constraint_types, self.max_params_per_constraint),
                                dtype=torch.float32
                            ))
                        elif attr in ['constraint_mask', 'constraint_confidence', 'y']:
                            tensor = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
                            # For y and mask, set NO_CONSTRAINT
                            if attr in ['y', 'constraint_mask'] and self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                tensor[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            # For confidence, set high confidence for NO_CONSTRAINT
                            if attr == 'constraint_confidence' and self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                tensor[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                    
                            setattr(labeled_graph, attr, tensor)
                            print(f"    Recreated {attr} with shape {tensor.shape}")
        
                # Ensure y has NO_CONSTRAINT set for edges without constraints
                if hasattr(labeled_graph, 'y') and labeled_graph.y is not None and hasattr(labeled_graph, 'constraint_mask') and labeled_graph.constraint_mask is not None:
                    no_constraint_idx = self.constraint_to_idx.get('NO_CONSTRAINT', 0)
            
                    # Find edges with no constraints (where all constraint_mask entries are 0)
                    for edge_idx in range(num_edges):
                        if labeled_graph.constraint_mask[edge_idx].sum() == 0:
                            labeled_graph.y[edge_idx, no_constraint_idx] = 1.0
                            labeled_graph.constraint_mask[edge_idx, no_constraint_idx] = 1.0
        
                # Check for NaN or Inf values and replace them
                if hasattr(labeled_graph, 'constraint_params_tensor') and labeled_graph.constraint_params_tensor is not None:
                    if torch.isnan(labeled_graph.constraint_params_tensor).any():
                        print("  WARNING: NaN values found in constraint_params_tensor - replacing with zeros")
                        labeled_graph.constraint_params_tensor = torch.nan_to_num(
                            labeled_graph.constraint_params_tensor, nan=0.0
                        )
                    if torch.isinf(labeled_graph.constraint_params_tensor).any():
                        print("  WARNING: Inf values found in constraint_params_tensor - replacing with zeros")
                        labeled_graph.constraint_params_tensor = torch.nan_to_num(
                            labeled_graph.constraint_params_tensor, posinf=0.0, neginf=0.0
                        )
        
                if hasattr(labeled_graph, 'constraint_mask') and labeled_graph.constraint_mask is not None:
                    if torch.isnan(labeled_graph.constraint_mask).any():
                        print("  WARNING: NaN values found in constraint_mask - replacing with zeros")
                        labeled_graph.constraint_mask = torch.nan_to_num(
                            labeled_graph.constraint_mask, nan=0.0
                        )
        
                if hasattr(labeled_graph, 'constraint_confidence') and labeled_graph.constraint_confidence is not None:
                    if torch.isnan(labeled_graph.constraint_confidence).any():
                        print("  WARNING: NaN values found in constraint_confidence - replacing with zeros")
                        labeled_graph.constraint_confidence = torch.nan_to_num(
                            labeled_graph.constraint_confidence, nan=0.0
                        )
        
                if hasattr(labeled_graph, 'y') and labeled_graph.y is not None:
                    if torch.isnan(labeled_graph.y).any():
                        print("  WARNING: NaN values found in y - replacing with zeros")
                        labeled_graph.y = torch.nan_to_num(labeled_graph.y, nan=0.0)
        
                # ====================================================================
                # Final verification before saving
                # ====================================================================
                print(f"\n  Final tensor verification for {os.path.basename(graph_file_path)}:")
        
                verification_passed = True
        
                # Verify y
                if hasattr(labeled_graph, 'y') and labeled_graph.y is not None:
                    print(f"    y: shape={labeled_graph.y.shape}, dtype={labeled_graph.y.dtype}, "
                        f"min={labeled_graph.y.min():.2f}, max={labeled_graph.y.max():.2f}")
                    if labeled_graph.y.dim() != 2:
                        print(f"    ERROR: y has wrong dimensions: {labeled_graph.y.dim()}")
                        verification_passed = False
                else:
                    print(f"    ERROR: y is missing or None")
                    verification_passed = False
                    # Create y tensor
                    labeled_graph.y = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
                    if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                        labeled_graph.y[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                    print(f"    Created default y with shape {labeled_graph.y.shape}")
        
                # Verify constraint_params_tensor
                if hasattr(labeled_graph, 'constraint_params_tensor') and labeled_graph.constraint_params_tensor is not None:
                    print(f"    constraint_params_tensor: shape={labeled_graph.constraint_params_tensor.shape}, "
                        f"dtype={labeled_graph.constraint_params_tensor.dtype}")
                    if labeled_graph.constraint_params_tensor.dim() != 3:
                        print(f"    ERROR: constraint_params_tensor has wrong dimensions: {labeled_graph.constraint_params_tensor.dim()}")
                        verification_passed = False
                else:
                    print(f"    ERROR: constraint_params_tensor is missing or None")
                    verification_passed = False
                    # Create constraint_params_tensor
                    labeled_graph.constraint_params_tensor = torch.zeros(
                        (num_edges, self.num_constraint_types, self.max_params_per_constraint),
                        dtype=torch.float32
                    )
                    print(f"    Created default constraint_params_tensor with shape {labeled_graph.constraint_params_tensor.shape}")
        
                # Verify constraint_mask
                if hasattr(labeled_graph, 'constraint_mask') and labeled_graph.constraint_mask is not None:
                    print(f"    constraint_mask: shape={labeled_graph.constraint_mask.shape}, "
                        f"dtype={labeled_graph.constraint_mask.dtype}")
                    if labeled_graph.constraint_mask.dim() != 2:
                        print(f"    ERROR: constraint_mask has wrong dimensions: {labeled_graph.constraint_mask.dim()}")
                        verification_passed = False
                else:
                    print(f"    ERROR: constraint_mask is missing or None")
                    verification_passed = False
                    # Create constraint_mask
                    labeled_graph.constraint_mask = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
                    if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                        labeled_graph.constraint_mask[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                    print(f"    Created default constraint_mask with shape {labeled_graph.constraint_mask.shape}")
        
                # Verify constraint_confidence
                if hasattr(labeled_graph, 'constraint_confidence') and labeled_graph.constraint_confidence is not None:
                    print(f"    constraint_confidence: shape={labeled_graph.constraint_confidence.shape}, "
                        f"dtype={labeled_graph.constraint_confidence.dtype}")
                    if labeled_graph.constraint_confidence.dim() != 2:
                        print(f"    ERROR: constraint_confidence has wrong dimensions: {labeled_graph.constraint_confidence.dim()}")
                        verification_passed = False
                else:
                    print(f"    ERROR: constraint_confidence is missing or None")
                    verification_passed = False
                    # Create constraint_confidence
                    labeled_graph.constraint_confidence = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
                    if self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                        labeled_graph.constraint_confidence[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                    print(f"    Created default constraint_confidence with shape {labeled_graph.constraint_confidence.shape}")
        
                # ====================================================================
                # Save labeled graph - DIRECTLY WITHOUT RECREATING
                # ====================================================================
                base_name = os.path.splitext(os.path.basename(graph_file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_labeled_graph.pt")
        
                # Final check that all required tensors are not None before saving
                final_check_passed = True
                for attr in required_tensor_attrs:
                    if not hasattr(labeled_graph, attr) or getattr(labeled_graph, attr) is None:
                        print(f"  ERROR: {attr} is still None before saving - fixing")
                        final_check_passed = False
                
                        if attr == 'constraint_params_tensor':
                            setattr(labeled_graph, attr, torch.zeros(
                                (num_edges, self.num_constraint_types, self.max_params_per_constraint),
                                dtype=torch.float32
                            ))
                        elif attr in ['constraint_mask', 'constraint_confidence', 'y']:
                            tensor = torch.zeros((num_edges, self.num_constraint_types), dtype=torch.float32)
                            if attr in ['y', 'constraint_mask'] and self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                tensor[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            if attr == 'constraint_confidence' and self.constraint_to_idx and 'NO_CONSTRAINT' in self.constraint_to_idx:
                                tensor[:, self.constraint_to_idx['NO_CONSTRAINT']] = 1.0
                            setattr(labeled_graph, attr, tensor)
        
                # Save the graph directly (don't recreate it)
                torch.save(labeled_graph, output_path)
        
                # Print statistics - with safe access
                if hasattr(labeled_graph, 'y') and labeled_graph.y is not None:
                    num_constraints = (labeled_graph.y[:, 1:].sum(dim=1) > 0).sum().item() if labeled_graph.y.shape[1] > 1 else 0
                    no_constraint_count = (labeled_graph.y[:, 0] > 0.5).sum().item()
                else:
                    num_constraints = 0
                    no_constraint_count = num_edges
        
                print(f"\n  Created: {output_path}")
                print(f"  Edges: {num_edges}, Edges with constraints: {num_constraints}")
                print(f"  NO_CONSTRAINT edges: {no_constraint_count}")
                print(f"  Tensor verification: {'PASSED' if verification_passed and final_check_passed else 'PASSED WITH FIXES'}")
        
                if hasattr(labeled_graph, 'constraint_params') and labeled_graph.constraint_params:
                    constraint_types = [k for k in labeled_graph.constraint_params.keys() if k != 'NO_CONSTRAINT']
                    print(f"  Active constraint types found: {constraint_types[:5]}{'...' if len(constraint_types) > 5 else ''}")
        
                return True
        
            except Exception as e:
                print(f"Error processing {graph_file_path}: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def batch_process_graphs(self, input_dir, output_dir):
        """Process all graph files in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find all PyG graph files
        graph_files = glob.glob(os.path.join(input_dir, "*_hybrid_graph.pt"))
        
        if not graph_files:
            print(f"No hybrid graph files found in {input_dir}")
            # Try other graph file patterns
            graph_files = glob.glob(os.path.join(input_dir, "*.pt"))
        
        print(f"Found {len(graph_files)} graph files to process")
        
        success_count = 0
        failed_files = []
        
        # Track overall statistics
        total_edges = 0
        total_constraint_edges = 0
        constraint_type_counts = {}
        
        for graph_file in tqdm(graph_files, desc="Labeling graphs"):
            success = self.process_single_graph(graph_file, output_dir)
            
            if success:
                success_count += 1
                
                # Load the saved graph to aggregate statistics
                try:
                    base_name = os.path.splitext(os.path.basename(graph_file))[0]
                    saved_path = os.path.join(output_dir, f"{base_name}.pt")
                    saved_graph = torch.load(saved_path, weights_only=False)
                    
                    num_edges = saved_graph.edge_index.shape[1]
                    total_edges += num_edges
                    
                    # Count constraint edges (excluding NO_CONSTRAINT)
                    constraint_edges = (saved_graph.y[:, 1:].sum(dim=1) > 0).sum().item()
                    total_constraint_edges += constraint_edges
                    
                    # Count per constraint type
                    for constraint_name in self.constraint_names[1:]:  # Skip NO_CONSTRAINT
                        if constraint_name in saved_graph.constraint_params:
                            count = len(saved_graph.constraint_params[constraint_name])
                            if constraint_name not in constraint_type_counts:
                                constraint_type_counts[constraint_name] = 0
                            constraint_type_counts[constraint_name] += count
                            
                except Exception as e:
                    print(f"Warning: Could not aggregate stats for {graph_file}: {e}")
            
            else:
                failed_files.append(os.path.basename(graph_file))
            
            # Clean up memory
            gc.collect()
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"PHASE 3 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {success_count}/{len(graph_files)}")
        print(f"Failed: {len(failed_files)}")
        
        if total_edges > 0:
            print(f"\nConstraint Statistics Across All Graphs:")
            print(f"  Total edges: {total_edges}")
            print(f"  Edges with constraints: {total_constraint_edges}")
            print(f"  Constraint density: {total_constraint_edges/total_edges*100:.2f}%")
            print(f"  NO_CONSTRAINT edges: {total_edges - total_constraint_edges}")
            
            print(f"\nPer-Constraint Type Distribution:")
            sorted_constraints = sorted(constraint_type_counts.items(), key=lambda x: x[1], reverse=True)
            for constraint_name, count in sorted_constraints[:15]:  # Show top 15
                percentage = count / total_constraint_edges * 100 if total_constraint_edges > 0 else 0
                print(f"  {constraint_name}: {count} ({percentage:.2f}%)")
            if len(sorted_constraints) > 15:
                print(f"  ... and {len(sorted_constraints)-15} more types")
        
        if failed_files:
            print(f"\nFailed files: {failed_files[:10]}{'...' if len(failed_files) > 10 else ''}")
        
        # Create a summary file
        summary = {
            'total_files': len(graph_files),
            'successful': success_count,
            'failed': len(failed_files),
            'failed_files': failed_files,
            'total_edges_processed': total_edges,
            'total_constraint_edges': total_constraint_edges,
            'constraint_density': total_constraint_edges/total_edges if total_edges > 0 else 0,
            'constraint_type_counts': constraint_type_counts,
            'constraint_definitions': self.constraint_definitions,
            'constraint_names': self.constraint_names,
            'num_constraint_types': self.num_constraint_types,
            'max_params_per_constraint': self.max_params_per_constraint,
            'tolerances': {
                'angle_tolerance': self.angle_tolerance,
                'distance_tolerance': self.distance_tolerance,
                'radius_tolerance': self.radius_tolerance,
                'length_tolerance': self.length_tolerance,
                'min_confidence_threshold': self.min_confidence_threshold
            }
        }
        
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary saved to: {summary_path}")
        
        return success_count

def main():
    """Main Phase 3 execution"""
    
    # Configuration
    PHASE2_DIR = "/media/anubhab/External/Project/Pyg_graphs"  # Directory with Phase 2 PyG graphs
    PHASE3_OUTPUT_DIR = "/media/anubhab/External/Project/labeled_graphs"  # Output for labeled graphs
    
    # Create detector with your tolerances
    detector = GeometricConstraintDetector(
        angle_tolerance=5.0,      # Degrees
        distance_tolerance=0.01,  # Normalized units
        radius_tolerance=0.05,    # Relative
        length_tolerance=0.05,    # Relative
        min_confidence_threshold=0.3  # Only keep constraints with confidence >= 0.3
    )
    
    # Process all graphs
    success_count = detector.batch_process_graphs(PHASE2_DIR, PHASE3_OUTPUT_DIR)
    
    if success_count > 0:
        print(f"\n✓ Phase 3 completed successfully!")
        print(f"  Generated {success_count} labeled graphs with NO_CONSTRAINT class")
        print(f"  Output directory: {PHASE3_OUTPUT_DIR}")
        
        # Test loading one of the generated graphs
        test_files = glob.glob(os.path.join(PHASE3_OUTPUT_DIR, "*_labeled_graph.pt"))
        if test_files:
            test_graph = torch.load(test_files[0], weights_only=False)
            print(f"\nSample graph statistics:")
            print(f"  Nodes: {test_graph.x.shape[0]}")
            print(f"  Edges: {test_graph.edge_index.shape[1]}")
            print(f"  Constraint labels shape: {test_graph.y.shape}")
            print(f"  Constraint names: {test_graph.constraint_names[:5]}...")
            print(f"  NO_CONSTRAINT index: {test_graph.constraint_names.index('NO_CONSTRAINT')}")
            
            # Count actual constraints (excluding NO_CONSTRAINT)
            constraint_mask = test_graph.y[:, 1:].sum(dim=1) > 0
            num_constraints = constraint_mask.sum().item()
            print(f"  Edges with constraints: {num_constraints}")
            print(f"  Edges with NO_CONSTRAINT: {(test_graph.y[:, 0] > 0.5).sum().item()}")
            
            if hasattr(test_graph, 'constraint_params'):
                active_constraint_types = [k for k in test_graph.constraint_params.keys() if k != 'NO_CONSTRAINT']
                print(f"  Active constraint types: {active_constraint_types[:5]}")
                
                # Show parameters for first constraint type if available
                if active_constraint_types:
                    first_type = active_constraint_types[0]
                    print(f"  Parameters for '{first_type}':")
                    params = test_graph.constraint_params[first_type]
                    if params:
                        sample_params = params[0]['parameters']
                        for key, value in sample_params.items():
                            if key != 'confidence':
                                print(f"    {key}: {value:.4f}")
                        print(f"    confidence: {sample_params.get('confidence', 0.5):.2f}")
    else:
        print(f"\n✗ Phase 3 failed - no graphs processed")
        print("  Check that Phase 2 generated graphs in the correct directory")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
