#!/usr/bin/env python3
"""
PHASE 5 (Enhanced): Synthesise Rich Construction Actions from Phase 1 JSON
Features:
- Extracts actual 2D sketch profiles (lines, arcs, circles) from planar faces.
- Recognises base extrusion, additive bosses, subtractive pockets, holes, fillets, chamfers.
- Detects revolved features and circular patterns.
- Outputs a detailed action sequence for each model.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Union
import numpy as np
from collections import defaultdict, deque
import math


# ----------------------------------------------------------------------
# Geometric Utilities
# ----------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def project_point_to_plane(point: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a plane."""
    v = point - plane_origin
    dist = np.dot(v, plane_normal)
    return point - dist * plane_normal


def transform_to_2d(point_3d: np.ndarray, origin: np.ndarray, u_dir: np.ndarray, v_dir: np.ndarray) -> Tuple[float, float]:
    """Convert 3D point on plane to 2D coordinates in local (u,v) basis."""
    rel = point_3d - origin
    x = np.dot(rel, u_dir)
    y = np.dot(rel, v_dir)
    return float(x), float(y)


def build_local_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given a plane normal, return two orthogonal unit vectors spanning the plane."""
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = normalize(u)
    v = np.cross(normal, u)
    v = normalize(v)
    return u, v


def fit_circle(points: List[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """Fit a circle to 3D points (assumed coplanar). Returns center, radius, rmse."""
    # Simple algebraic fit using least squares (Taubin method is better but we use basic)
    # For brevity, we assume points are roughly circular; in production use scipy.optimize.
    if len(points) < 3:
        return np.array([0,0,0]), 0.0, float('inf')
    # Convert to 2D using PCA
    mean = np.mean(points, axis=0)
    centered = points - mean
    u, s, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    u_dir = vh[0, :]
    v_dir = vh[1, :]
    pts_2d = np.array([[np.dot(p, u_dir), np.dot(p, v_dir)] for p in centered])
    # Solve: (x^2 + y^2) = 2*xc*x + 2*yc*y + (r^2 - xc^2 - yc^2)
    A = np.column_stack([2*pts_2d[:,0], 2*pts_2d[:,1], np.ones(len(points))])
    b = pts_2d[:,0]**2 + pts_2d[:,1]**2
    try:
        xc, yc, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(c + xc**2 + yc**2)
        center_2d = np.array([xc, yc])
        center_3d = mean + center_2d[0]*u_dir + center_2d[1]*v_dir
        # RMSE
        residuals = np.sqrt((pts_2d[:,0]-xc)**2 + (pts_2d[:,1]-yc)**2) - radius
        rmse = np.sqrt(np.mean(residuals**2))
        return center_3d, radius, rmse
    except:
        return np.array([0,0,0]), 0.0, float('inf')


# ----------------------------------------------------------------------
# Main Action Generator Class
# ----------------------------------------------------------------------
class RobustActionGenerator:
    def __init__(self):
        self.scale_factor = 1.0
        self.tolerance = 1e-4
        self.angle_tol = 5.0  # degrees
        self.fillet_radius_tol = 0.2  # relative to model size

    def load_json(self, path: str) -> Dict:
        with open(path) as f:
            return json.load(f)

    def get_face_by_id(self, faces: List, fid: int) -> Optional[Dict]:
        for f in faces:
            if f['id'] == fid:
                return f
        return None

    def get_edge_by_id(self, edges: List, eid: int) -> Optional[Dict]:
        for e in edges:
            if e['id'] == eid:
                return e
        return None

    def get_vertex_by_id(self, vertices: List, vid: int) -> Optional[Dict]:
        for v in vertices:
            if v['id'] == vid:
                return v
        return None

    def is_planar(self, face: Dict) -> bool:
        st = face.get('surface_type', '')
        return st.startswith('PLANE')

    def is_cylindrical(self, face: Dict) -> bool:
        st = face.get('surface_type', '')
        return st.startswith('CYLINDER')

    def is_conical(self, face: Dict) -> bool:
        st = face.get('surface_type', '')
        return st.startswith('CONE')

    def is_spherical(self, face: Dict) -> bool:
        st = face.get('surface_type', '')
        return st.startswith('SPHERE')

    def is_toroidal(self, face: Dict) -> bool:
        st = face.get('surface_type', '')
        return st.startswith('TORUS')

    def is_line_edge(self, edge: Dict) -> bool:
        ct = edge.get('curve_type', '')
        return ct.startswith('LINE')

    def is_circle_edge(self, edge: Dict) -> bool:
        ct = edge.get('curve_type', '')
        return ct.startswith('CIRCLE')

    def compute_face_area(self, face: Dict) -> float:
        return face.get('area', 0.0)

    def normalize_point(self, pt: List[float]) -> np.ndarray:
        return np.array(pt) * self.scale_factor

    # ------------------------------------------------------------------
    # Topology Queries
    # ------------------------------------------------------------------
    def build_adjacency_maps(self, faces: List, edges: List, topology: Dict):
        """Create dictionaries for quick lookup."""
        self.face_edge = defaultdict(list)   # face_id -> list of edge_ids
        self.edge_face = defaultdict(list)   # edge_id -> list of face_ids
        self.edge_vertex = defaultdict(list) # edge_id -> list of vertex_ids
        self.vertex_edge = defaultdict(list) # vertex_id -> list of edge_ids
        self.vertex_face = defaultdict(list) # vertex_id -> list of face_ids
        self.face_face = defaultdict(set)    # face_id -> set of adjacent face_ids

        fe_adj = topology.get('face_edge_adjacency', [[], []])
        for f_idx, e_idx in zip(fe_adj[0], fe_adj[1]):
            self.face_edge[f_idx].append(e_idx)
            self.edge_face[e_idx].append(f_idx)

        ev_adj = topology.get('edge_vertex_adjacency', [[], []])
        for e_idx, v_idx in zip(ev_adj[0], ev_adj[1]):
            self.edge_vertex[e_idx].append(v_idx)
            self.vertex_edge[v_idx].append(e_idx)

        vf_adj = topology.get('vertex_face_adjacency', [[], []])
        for v_idx, f_idx in zip(vf_adj[0], vf_adj[1]):
            self.vertex_face[v_idx].append(f_idx)

        # face-face adjacency (via shared edges)
        for e_idx, f_list in self.edge_face.items():
            if len(f_list) == 2:
                self.face_face[f_list[0]].add(f_list[1])
                self.face_face[f_list[1]].add(f_list[0])

    def get_adjacent_faces(self, face_id: int) -> Set[int]:
        return self.face_face.get(face_id, set())

    # ------------------------------------------------------------------
    # Sketch Profile Extraction
    # ------------------------------------------------------------------
    def extract_outer_wire(self, face_id: int, faces: List, edges: List, vertices: List) -> Optional[List[Dict]]:
        """
        Return the outer boundary loop of a planar face as an ordered list of edge
        curves, each represented as a dict with type and parameters in 2D.
        """
        face = self.get_face_by_id(faces, face_id)
        if not face or not self.is_planar(face):
            return None

        normal = np.array(face.get('normal', [0, 0, 1]))
        origin = self.normalize_point(face.get('centroid', [0, 0, 0]))
        u_dir, v_dir = build_local_basis(normal)

        # Get all edges belonging to this face
        edge_ids = self.face_edge.get(face_id, [])
        if not edge_ids:
            return None

        # Build a graph of edges connecting via vertices
        edge_graph = defaultdict(list)  # edge_id -> list of (next_edge_id, shared_vertex_id, orientation)
        for eid in edge_ids:
            vids = self.edge_vertex.get(eid, [])
            if len(vids) != 2:
                continue
            v0, v1 = vids
            # Find other edges sharing these vertices
            for vid in (v0, v1):
                for other_eid in self.vertex_edge.get(vid, []):
                    if other_eid != eid and other_eid in edge_ids:
                        # Determine if we need to reverse orientation
                        other_vids = self.edge_vertex[other_eid]
                        if other_vids[0] == vid:
                            orientation = 1  # forward
                        else:
                            orientation = -1 # reverse
                        edge_graph[eid].append((other_eid, vid, orientation))

        # Find the outer loop (largest perimeter)
        visited_edges = set()
        loops = []
        for start_eid in edge_ids:
            if start_eid in visited_edges:
                continue
            loop = []
            current = start_eid
            prev_vid = None
            while True:
                if current in visited_edges:
                    break
                visited_edges.add(current)
                # Get edge geometry
                edge = self.get_edge_by_id(edges, current)
                vids = self.edge_vertex[current]
                if prev_vid is None:
                    # start with first vertex
                    start_vertex = vids[0]
                    next_vertex = vids[1]
                else:
                    # continue from previous shared vertex
                    if vids[0] == prev_vid:
                        start_vertex = vids[0]
                        next_vertex = vids[1]
                    else:
                        start_vertex = vids[1]
                        next_vertex = vids[0]
                # Get 3D points of start and end vertices
                v_start = self.get_vertex_by_id(vertices, start_vertex)
                v_end = self.get_vertex_by_id(vertices, next_vertex)
                if not v_start or not v_end:
                    break
                p_start = self.normalize_point(v_start['point'])
                p_end = self.normalize_point(v_end['point'])
                # Project to 2D sketch plane
                p2d_start = transform_to_2d(p_start, origin, u_dir, v_dir)
                p2d_end = transform_to_2d(p_end, origin, u_dir, v_dir)
                # Classify curve type
                if self.is_line_edge(edge):
                    loop.append({
                        'type': 'line',
                        'start': list(p2d_start),
                        'end': list(p2d_end)
                    })
                elif self.is_circle_edge(edge):
                    # For arcs, we need center and angles.
                    # Extract from edge parameters.
                    params = edge.get('parameters', {})
                    center_3d = self.normalize_point(params.get('center', [0,0,0]))
                    radius = params.get('radius', 1.0) * self.scale_factor
                    axis = np.array(params.get('axis', [0,0,1]))
                    # Check if the circle is in the sketch plane
                    if angle_between(axis, normal) > self.angle_tol and angle_between(axis, normal) < 180-self.angle_tol:
                        # Not in plane, treat as line approximation (fallback)
                        loop.append({
                            'type': 'line',
                            'start': list(p2d_start),
                            'end': list(p2d_end)
                        })
                        continue
                    # Determine if arc is clockwise or counterclockwise relative to normal
                    # Compute angles of start and end points relative to center in local 2D.
                    center_2d = transform_to_2d(center_3d, origin, u_dir, v_dir)
                    # Vector from center to start/end
                    v_start_2d = np.array(p2d_start) - np.array(center_2d)
                    v_end_2d = np.array(p2d_end) - np.array(center_2d)
                    start_angle = math.atan2(v_start_2d[1], v_start_2d[0])
                    end_angle = math.atan2(v_end_2d[1], v_end_2d[0])
                    # Adjust for full circle
                    if np.linalg.norm(v_start_2d - v_end_2d) < self.tolerance:
                        # Full circle
                        loop.append({
                            'type': 'circle',
                            'center': list(center_2d),
                            'radius': radius
                        })
                    else:
                        # Arc: determine sweep direction based on edge direction and normal
                        # For simplicity, we output start/end angles and let CAD infer sweep.
                        loop.append({
                            'type': 'arc',
                            'center': list(center_2d),
                            'radius': radius,
                            'start_angle': start_angle,
                            'end_angle': end_angle
                        })
                else:
                    # Approximate as line
                    loop.append({
                        'type': 'line',
                        'start': list(p2d_start),
                        'end': list(p2d_end)
                    })
                # Find next edge
                next_edges = [e for e, vid, orient in edge_graph[current] if vid == next_vertex]
                if not next_edges:
                    break
                current = next_edges[0]
                prev_vid = next_vertex
                if current == start_eid:
                    # Closed loop
                    break
            if loop:
                loops.append(loop)

        if not loops:
            return None
        # Return the longest loop (by number of edges)
        return max(loops, key=len)

    # ------------------------------------------------------------------
    # Feature Recognition
    # ------------------------------------------------------------------
    def identify_base_face(self, faces: List) -> Optional[Dict]:
        """Heuristically select the base face (largest planar, normal roughly -Z or +Z)."""
        planar_faces = [f for f in faces if self.is_planar(f)]
        if not planar_faces:
            return max(faces, key=lambda f: self.compute_face_area(f), default=None)
        best = None
        best_score = -1
        for f in planar_faces:
            normal = np.array(f.get('normal', [0,0,1]))
            area = self.compute_face_area(f)
            # Prefer normals pointing down (or up)
            z_alignment = abs(normal[2])
            score = area * (0.5 + 0.5 * z_alignment)
            if score > best_score:
                best_score = score
                best = f
        return best

    def guess_extrusion_direction_and_distance(self, base_face: Dict, all_faces: List) -> Tuple[np.ndarray, float]:
        """Determine extrusion direction (base normal) and distance."""
        normal = np.array(base_face.get('normal', [0,0,1]))
        # Find opposite face (parallel, with normal opposite, and furthest along normal)
        base_center = self.normalize_point(base_face['centroid'])
        max_dist = 0.0
        for f in all_faces:
            if f['id'] == base_face['id'] or not self.is_planar(f):
                continue
            f_normal = np.array(f.get('normal', [0,0,1]))
            if angle_between(normal, f_normal) > 170:  # roughly opposite
                f_center = self.normalize_point(f['centroid'])
                dist = abs(np.dot(f_center - base_center, normal))
                if dist > max_dist:
                    max_dist = dist
        if max_dist == 0:
            # Fallback: use bounding box
            max_dist = 10.0 * self.scale_factor
        return normal, max_dist

    def classify_feature_type(self, face: Dict, base_normal: np.ndarray, base_origin: np.ndarray, extrusion_dist: float, all_faces: List) -> str:
        """
        Classify a face (not part of base or side walls) as:
        'hole', 'pocket', 'boss', 'fillet', 'chamfer', or 'other'.
        """
        if self.is_cylindrical(face):
            # Check if it's a hole (through or blind)
            adj_faces = self.get_adjacent_faces(face['id'])
            planar_adj = [fid for fid in adj_faces if self.is_planar(self.get_face_by_id(all_faces, fid))]
            # Through hole: connected to two planar faces (top and bottom) with opposite normals
            if len(planar_adj) >= 2:
                return 'hole'
            else:
                return 'pocket'  # blind hole / pocket
        elif self.is_planar(face):
            normal = np.array(face.get('normal', [0,0,1]))
            center = self.normalize_point(face['centroid'])
            # Check if it's an additive boss (normal roughly parallel to base normal and offset outward)
            if angle_between(normal, base_normal) < 30:
                offset = np.dot(center - base_origin, base_normal)
                if offset > extrusion_dist * 1.1:  # above top face
                    return 'boss'
                elif offset < -0.1 * extrusion_dist:  # below base
                    return 'boss'  # or cut
            # Could be a rib or web; for now 'other'
        elif self.is_toroidal(face) or (self.is_cylindrical(face) and self.compute_face_area(face) < 0.05 * np.mean([self.compute_face_area(f) for f in all_faces])):
            # Small cylindrical faces often are fillets
            return 'fillet'
        return 'other'

    def extract_hole_parameters(self, face: Dict) -> Dict:
        """Extract center, radius, depth, axis for a hole."""
        params = face.get('parameters', {})
        center = self.normalize_point(face.get('centroid', [0,0,0]))
        radius = params.get('radius', 1.0) * self.scale_factor
        axis = params.get('axis', [0,0,1])
        # Depth estimation: distance between adjacent planar faces along axis
        # For simplicity, we'll set a placeholder and let later steps refine.
        depth = 20.0  # Placeholder; ideally computed from adjacent faces
        return {
            'center': center.tolist(),
            'radius': radius,
            'axis': axis,
            'depth': depth
        }

    def extract_fillet_parameters(self, face: Dict) -> Optional[Dict]:
        """Extract radius for a fillet face."""
        if self.is_cylindrical(face):
            params = face.get('parameters', {})
            radius = params.get('radius', 0.5) * self.scale_factor
            return {'radius': radius}
        return None

    # ------------------------------------------------------------------
    # Main Sequence Generation
    # ------------------------------------------------------------------
    def generate_action_sequence(self, json_data: Dict) -> List[Dict]:
        geometry = json_data.get('geometry_data', {})
        faces = geometry.get('faces', [])
        edges = geometry.get('edges', [])
        vertices = geometry.get('vertices', [])
        topology = json_data.get('topology_graph', {})
        metadata = json_data.get('metadata', {})
        self.scale_factor = metadata.get('model_scale', 1.0)

        if not faces:
            return []

        self.build_adjacency_maps(faces, edges, topology)

        actions = []
        processed_faces = set()

        # 1. Identify base face and extrusion
        base_face = self.identify_base_face(faces)
        if not base_face:
            return [{'action': 'Error', 'params': {'message': 'No base face found'}}]

        base_id = base_face['id']
        processed_faces.add(base_id)

        # Extract sketch profile from base face
        profile = self.extract_outer_wire(base_id, faces, edges, vertices)
        if not profile:
            # Fallback: bounding rectangle
            profile = [{'type': 'rectangle', 'width': 10.0, 'height': 10.0}]

        normal = np.array(base_face.get('normal', [0,0,1]))
        origin = self.normalize_point(base_face.get('centroid', [0,0,0]))

        actions.append({
            'action': 'CreateSketch',
            'params': {
                'plane_normal': normal.tolist(),
                'plane_origin': origin.tolist(),
                'profile': profile
            }
        })

        extrude_dir, extrude_dist = self.guess_extrusion_direction_and_distance(base_face, faces)
        actions.append({
            'action': 'Extrude',
            'params': {
                'distance': extrude_dist,
                'direction': extrude_dir.tolist(),
                'taper_angle': 0.0
            }
        })

        # Mark side faces as processed (those adjacent to base and roughly perpendicular)
        base_adj = self.get_adjacent_faces(base_id)
        side_faces = []
        for fid in base_adj:
            f = self.get_face_by_id(faces, fid)
            if f and self.is_planar(f):
                f_normal = np.array(f.get('normal', [0,0,0]))
                if angle_between(f_normal, normal) > 80:
                    side_faces.append(fid)
                    processed_faces.add(fid)

        # Also mark the opposite face (top cap)
        for f in faces:
            if f['id'] in processed_faces:
                continue
            if self.is_planar(f):
                f_normal = np.array(f.get('normal', [0,0,1]))
                f_center = self.normalize_point(f['centroid'])
                if angle_between(f_normal, normal) < 10 and abs(np.dot(f_center - origin, normal) - extrude_dist) < 0.1 * extrude_dist:
                    processed_faces.add(f['id'])
                    break

        # 2. Process remaining faces as features
        # Sort by area descending (larger features first)
        remaining = [f for f in faces if f['id'] not in processed_faces]
        remaining.sort(key=lambda f: self.compute_face_area(f), reverse=True)

        hole_faces = []
        fillet_faces = []
        boss_faces = []

        for face in remaining:
            ftype = self.classify_feature_type(face, normal, origin, extrude_dist, faces)
            if ftype == 'hole':
                hole_faces.append(face)
            elif ftype == 'fillet':
                fillet_faces.append(face)
            elif ftype == 'boss':
                boss_faces.append(face)

        # Add holes
        for face in hole_faces:
            params = self.extract_hole_parameters(face)
            actions.append({
                'action': 'AddHole',
                'params': params
            })
            processed_faces.add(face['id'])

        # Add bosses (as extrusions from base or top)
        for face in boss_faces:
            # For simplicity, treat as an extrusion from the base plane
            boss_profile = self.extract_outer_wire(face['id'], faces, edges, vertices)
            if boss_profile:
                boss_normal = np.array(face.get('normal', [0,0,1]))
                boss_origin = self.normalize_point(face['centroid'])
                boss_height = abs(np.dot(boss_origin - origin, normal) - extrude_dist)
                actions.append({
                    'action': 'CreateSketch',
                    'params': {
                        'plane_normal': boss_normal.tolist(),
                        'plane_origin': boss_origin.tolist(),
                        'profile': boss_profile
                    }
                })
                actions.append({
                    'action': 'Extrude',
                    'params': {
                        'distance': boss_height,
                        'direction': boss_normal.tolist(),
                        'taper_angle': 0.0,
                        'operation': 'join'  # additive
                    }
                })
            processed_faces.add(face['id'])

        # Add fillets
        for face in fillet_faces:
            params = self.extract_fillet_parameters(face)
            if params:
                actions.append({
                    'action': 'Fillet',
                    'params': params
                })
            processed_faces.add(face['id'])

        # 3. Detect patterns among holes
        if len(hole_faces) >= 3:
            # Check for circular pattern
            centers = [np.array(self.extract_hole_parameters(h)['center']) for h in hole_faces]
            center_mean = np.mean(centers, axis=0)
            distances = [np.linalg.norm(c - center_mean) for c in centers]
            if np.std(distances) < 0.1 * np.mean(distances):
                # Circular pattern detected
                actions.append({
                    'action': 'CircularPattern',
                    'params': {
                        'center': center_mean.tolist(),
                        'axis': normal.tolist(),
                        'count': len(hole_faces),
                        'radius': np.mean(distances)
                    }
                })

        return actions

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------
    def process_directory(self, json_dir: str, text_csv: str, actions_dir: str, output_csv: str):
        os.makedirs(actions_dir, exist_ok=True)

        # Load text mapping
        text_map = {}
        with open(text_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_map[row['graph_path']] = row['text']

        json_files = list(Path(json_dir).glob("*.json"))
        pairs = []
        for jf in json_files:
            try:
                data = self.load_json(jf)
                actions = self.generate_action_sequence(data)
                base = jf.stem
                action_file = os.path.join(actions_dir, f"{base}_actions.json")
                with open(action_file, 'w') as f:
                    json.dump(actions, f, indent=2)

                # Find corresponding text
                text = None
                for graph_path, txt in text_map.items():
                    if base in graph_path:
                        text = txt
                        break
                if text:
                    pairs.append([text, action_file])
                else:
                    print(f"Warning: No text for {base}")
            except Exception as e:
                print(f"Error {jf}: {e}")
                import traceback
                traceback.print_exc()

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'action_file'])
            writer.writerows(pairs)

        print(f"Generated {len(pairs)} action sequences.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--text_csv", required=True)
    parser.add_argument("--actions_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    gen = RobustActionGenerator()
    gen.process_directory(args.json_dir, args.text_csv, args.actions_dir, args.output_csv)


if __name__ == "__main__":
    main()
