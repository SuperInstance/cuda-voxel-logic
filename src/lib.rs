/*!
# cuda-voxel-logic

Voxel-based spatial reasoning for agents.

The world is voxels. Not pixels, not polygons — cubes. Every position in
3D space is either occupied or empty, known or unknown, dangerous or safe.

This is how an agent builds an internal model of physical space:
- Occupancy grid (known/unknown/occupied/free)
- A* pathfinding through 3D space
- Raycasting for line-of-sight
- Constructive solid geometry for object composition

The agent doesn't see the world. It *voxelizes* the world.
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;

/// Voxel state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoxelState {
    Unknown,
    Free,
    Occupied,
    Dangerous,
    Goal,
}

/// 3D position
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Pos {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Pos { x, y, z } }
    pub fn neighbors6(&self) -> Vec<Pos> {
        vec![Pos::new(self.x+1,self.y,self.z), Pos::new(self.x-1,self.y,self.z),
             Pos::new(self.x,self.y+1,self.z), Pos::new(self.x,self.y-1,self.z),
             Pos::new(self.x,self.y,self.z+1), Pos::new(self.x,self.y,self.z-1)]
    }
    pub fn neighbors26(&self) -> Vec<Pos> {
        let mut out = vec![];
        for dx in -1..=1 { for dy in -1..=1 { for dz in -1..=1 {
            if dx == 0 && dy == 0 && dz == 0 { continue; }
            out.push(Pos::new(self.x+dx, self.y+dy, self.z+dz));
        }}}
        out
    }
    pub fn distance_to(&self, other: &Pos) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        let dz = (self.z - other.z) as f64;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }
    pub fn manhattan_to(&self, other: &Pos) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
}

/// The voxel grid — agent's internal model of space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub voxels: Vec<Vec<Vec<VoxelState>>>,
    pub confidence: Vec<Vec<Vec<f64>>>, // how sure about each voxel
}

impl VoxelGrid {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let voxels = vec![vec![vec![VoxelState::Unknown; depth]; height]; width];
        let confidence = vec![vec![vec![0.0; depth]; height]; width];
        VoxelGrid { width, height, depth, voxels, confidence }
    }

    pub fn get(&self, pos: &Pos) -> Option<VoxelState> {
        if pos.x < 0 || pos.y < 0 || pos.z < 0 { return None; }
        let x = pos.x as usize; let y = pos.y as usize; let z = pos.z as usize;
        if x >= self.width || y >= self.height || z >= self.depth { return None; }
        Some(self.voxels[x][y][z])
    }

    pub fn set(&mut self, pos: &Pos, state: VoxelState, conf: f64) {
        if pos.x < 0 || pos.y < 0 || pos.z < 0 { return; }
        let x = pos.x as usize; let y = pos.y as usize; let z = pos.z as usize;
        if x >= self.width || y >= self.height || z >= self.depth { return; }
        self.voxels[x][y][z] = state;
        self.confidence[x][y][z] = conf.clamp(0.0, 1.0);
    }

    pub fn is_traversable(&self, pos: &Pos) -> bool {
        self.get(pos).map_or(false, |s| s == VoxelState::Free || s == VoxelState::Goal || s == VoxelState::Unknown)
    }

    /// Fill a box region
    pub fn fill_box(&mut self, min: &Pos, max: &Pos, state: VoxelState) {
        for x in min.x..=max.x { for y in min.y..=max.y { for z in min.z..=max.z {
            self.set(&Pos::new(x, y, z), state, 1.0);
        }}}
    }

    /// Count voxels by state
    pub fn count(&self, state: VoxelState) -> usize {
        let mut count = 0;
        for x in &self.voxels { for y in x { for z in y { if *z == state { count += 1; } } } }
        count
    }

    /// Raycast from origin in direction, return first hit
    pub fn raycast(&self, origin: &Pos, dir: &Pos, max_steps: usize) -> Option<Pos> {
        let mut pos = *origin;
        for _ in 0..max_steps {
            match self.get(&pos) {
                Some(VoxelState::Occupied) | Some(VoxelState::Dangerous) => return Some(pos),
                Some(VoxelState::Unknown) => return Some(pos), // hit unknown wall
                _ => {}
            }
            pos = Pos::new(pos.x + dir.x, pos.y + dir.y, pos.z + dir.z);
        }
        None
    }

    /// Line of sight check
    pub fn has_los(&self, a: &Pos, b: &Pos) -> bool {
        let steps = a.manhattan_to(b).max(1) as usize;
        let dx = (b.x - a.x).signum();
        let dy = (b.y - a.y).signum();
        let dz = (b.z - a.z).signum();
        let mut pos = *a;
        for _ in 0..steps {
            pos = Pos::new(pos.x + dx, pos.y + dy, pos.z + dz);
            if pos == *b { return true; }
            if !self.is_traversable(&pos) { return false; }
        }
        true
    }
}

/// A* pathfinding through voxel grid
pub fn find_path(grid: &VoxelGrid, start: &Pos, goal: &Pos) -> Option<Vec<Pos>> {
    if !grid.is_traversable(start) || !grid.is_traversable(goal) { return None; }

    #[derive(Clone, Eq, PartialEq)]
    struct Node { pos: Pos, g: i32, f: i32 }
    impl Ord for Node { fn cmp(&self, other: &Self) -> other.f.cmp(&self.f) }
    impl PartialOrd for Node { fn partial_cmp(&self, other: &Self) -> Some(Ordering) { Some(self.cmp(other)) } }

    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<Pos, Pos> = HashMap::new();
    let mut g_score: HashMap<Pos, i32> = HashMap::new();

    g_score.insert(*start, 0);
    open.push(Node { pos: *start, g: 0, f: start.manhattan_to(goal) });

    let mut visited: HashSet<Pos> = HashSet::new();

    while let Some(current) = open.pop() {
        if current.pos == *goal {
            let mut path = vec![*goal];
            let mut p = *goal;
            while let Some(&prev) = came_from.get(&p) {
                path.push(prev);
                p = prev;
            }
            path.reverse();
            return Some(path);
        }

        if visited.contains(&current.pos) { continue; }
        visited.insert(current.pos);

        for neighbor in current.pos.neighbors6() {
            if visited.contains(&neighbor) || !grid.is_traversable(&neighbor) { continue; }
            let tentative_g = current.g + 1;
            let prev_g = *g_score.get(&neighbor).unwrap_or(&i32::MAX);
            if tentative_g < prev_g {
                g_score.insert(neighbor, tentative_g);
                came_from.insert(neighbor, current.pos);
                open.push(Node { pos: neighbor, g: tentative_g, f: tentative_g + neighbor.manhattan_to(goal) });
            }
        }
    }

    None
}

/// CSG operation types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CsgOp {
    Union,
    Intersection,
    Difference,
}

/// Apply CSG operation to grid
pub fn csg_apply(grid: &mut VoxelGrid, region_a: (&Pos, &Pos), region_b: (&Pos, &Pos), op: CsgOp, state: VoxelState) {
    let min = Pos::new(region_a.0.x.min(region_b.0.x), region_a.0.y.min(region_b.0.y), region_a.0.z.min(region_b.0.z));
    let max = Pos::new(region_a.1.x.max(region_b.1.x), region_a.1.y.max(region_b.1.y), region_a.1.z.max(region_b.1.z));

    for x in min.x..=max.x { for y in min.y..=max.y { for z in min.z..=max.z {
        let pos = Pos::new(x, y, z);
        let in_a = x >= region_a.0.x && x <= region_a.1.x && y >= region_a.0.y && y <= region_a.1.y && z >= region_a.0.z && z <= region_a.1.z;
        let in_b = x >= region_b.0.x && x <= region_b.1.x && y >= region_b.0.y && y <= region_b.1.y && z >= region_b.0.z && z <= region_b.1.z;
        let result = match op {
            CsgOp::Union => in_a || in_b,
            CsgOp::Intersection => in_a && in_b,
            CsgOp::Difference => in_a && !in_b,
        };
        if result { grid.set(&pos, state, 1.0); }
    }}}
}

/// Spatial query — find all voxels of a given state within radius
pub fn query_radius(grid: &VoxelGrid, center: &Pos, radius: i32, state: VoxelState) -> Vec<Pos> {
    let mut results = vec![];
    for dx in -radius..=radius { for dy in -radius..=radius { for dz in -radius..=radius {
        if dx*dx + dy*dy + dz*dz > radius*radius { continue; }
        let pos = Pos::new(center.x + dx, center.y + dy, center.z + dz);
        if grid.get(&pos) == Some(state) { results.push(pos); }
    }}}
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let g = VoxelGrid::new(10, 10, 10);
        assert_eq!(g.get(&Pos::new(0,0,0)), Some(VoxelState::Unknown));
        assert_eq!(g.get(&Pos::new(-1,0,0)), None);
    }

    #[test]
    fn test_set_get() {
        let mut g = VoxelGrid::new(10, 10, 10);
        g.set(&Pos::new(5, 5, 5), VoxelState::Occupied, 0.9);
        assert_eq!(g.get(&Pos::new(5, 5, 5)), Some(VoxelState::Occupied));
    }

    #[test]
    fn test_fill_box() {
        let mut g = VoxelGrid::new(20, 20, 20);
        g.fill_box(&Pos::new(0,0,0), &Pos::new(4,4,4), VoxelState::Occupied);
        assert_eq!(g.count(VoxelState::Occupied), 125); // 5^3
    }

    #[test]
    fn test_traversable() {
        let mut g = VoxelGrid::new(10, 10, 10);
        g.set(&Pos::new(5,5,5), VoxelState::Occupied, 1.0);
        assert!(!g.is_traversable(&Pos::new(5,5,5)));
        assert!(g.is_traversable(&Pos::new(0,0,0)));
    }

    #[test]
    fn test_pathfinding() {
        let mut g = VoxelGrid::new(20, 1, 20);
        // Clear path
        for x in 0..20 { g.set(&Pos::new(x, 0, 0), VoxelState::Free, 1.0); }
        let path = find_path(&g, &Pos::new(0,0,0), &Pos::new(19,0,0));
        assert!(path.is_some());
        assert!(path.unwrap().len() > 1);
    }

    #[test]
    fn test_pathfinding_blocked() {
        let mut g = VoxelGrid::new(10, 1, 10);
        for x in 0..10 { g.set(&Pos::new(x, 0, 0), VoxelState::Free, 1.0); }
        g.set(&Pos::new(5, 0, 0), VoxelState::Occupied, 1.0); // wall
        let path = find_path(&g, &Pos::new(0,0,0), &Pos::new(9,0,0));
        assert!(path.is_none()); // can't go around in 1D
    }

    #[test]
    fn test_raycast() {
        let mut g = VoxelGrid::new(20, 1, 1);
        for x in 0..20 { g.set(&Pos::new(x, 0, 0), VoxelState::Free, 1.0); }
        g.set(&Pos::new(10, 0, 0), VoxelState::Occupied, 1.0);
        let hit = g.raycast(&Pos::new(0,0,0), &Pos::new(1,0,0), 20);
        assert_eq!(hit, Some(Pos::new(10, 0, 0)));
    }

    #[test]
    fn test_line_of_sight() {
        let mut g = VoxelGrid::new(20, 1, 1);
        for x in 0..20 { g.set(&Pos::new(x, 0, 0), VoxelState::Free, 1.0); }
        assert!(g.has_los(&Pos::new(0,0,0), &Pos::new(19,0,0)));
        g.set(&Pos::new(10, 0, 0), VoxelState::Occupied, 1.0);
        assert!(!g.has_los(&Pos::new(0,0,0), &Pos::new(19,0,0)));
    }

    #[test]
    fn test_csg_union() {
        let mut g = VoxelGrid::new(20, 20, 20);
        csg_apply(&mut g, (&Pos::new(0,0,0), &Pos::new(4,4,4)), (&Pos::new(3,3,3), &Pos::new(7,7,7)), CsgOp::Union, VoxelState::Occupied);
        assert!(g.get(&Pos::new(6,6,6)) == Some(VoxelState::Occupied));
    }

    #[test]
    fn test_csg_intersection() {
        let mut g = VoxelGrid::new(20, 20, 20);
        csg_apply(&mut g, (&Pos::new(0,0,0), &Pos::new(4,4,4)), (&Pos::new(3,3,3), &Pos::new(7,7,7)), CsgOp::Intersection, VoxelState::Occupied);
        assert_eq!(g.get(&Pos::new(4,4,4)), Some(VoxelState::Occupied)); // in both
        assert_ne!(g.get(&Pos::new(6,6,6)), Some(VoxelState::Occupied)); // only in B
    }

    #[test]
    fn test_csg_difference() {
        let mut g = VoxelGrid::new(20, 20, 20);
        csg_apply(&mut g, (&Pos::new(0,0,0), &Pos::new(4,4,4)), (&Pos::new(2,2,2), &Pos::new(6,6,6)), CsgOp::Difference, VoxelState::Free);
        assert_eq!(g.get(&Pos::new(1,1,1)), Some(VoxelState::Free)); // in A only
        assert_ne!(g.get(&Pos::new(3,3,3)), Some(VoxelState::Free)); // in both, subtracted
    }

    #[test]
    fn test_query_radius() {
        let mut g = VoxelGrid::new(20, 20, 20);
        g.set(&Pos::new(5,5,5), VoxelState::Goal, 1.0);
        let results = query_radius(&g, &Pos::new(5,5,5), 3, VoxelState::Goal);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_confidence() {
        let mut g = VoxelGrid::new(10, 10, 10);
        g.set(&Pos::new(5,5,5), VoxelState::Occupied, 0.8);
        assert!((g.confidence[5][5][5] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_distance() {
        let a = Pos::new(0, 0, 0);
        let b = Pos::new(3, 4, 0);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_neighbors() {
        let p = Pos::new(5, 5, 5);
        assert_eq!(p.neighbors6().len(), 6);
        assert_eq!(p.neighbors26().len(), 26);
    }
}
