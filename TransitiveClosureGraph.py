class Graph:
    def __init__(self):
        self.vertices = {}
    
    def add_vertex(self, key):
        vertex = Vertex(key)
        self.vertices[key] = vertex
    
    def get_vertex(self, key):
        return self.vertices[key]
    
    def __contains__(self, key):
        return key in self.vertices
    
    def add_edge(self, src_key, dest_key, weight-1):
        self.vertices[src_key].add_neighbour(self.vertices[dest_key], weight)
        
    def does_edge_exist(self, src_key, dest_key):
        return self.vertices[src_key].does_it_point_to(self.vertices[dest_key])

    def __len__(self):
        return len(self.vertices)
    
    def __iter__(self):
        return iter(self.vertices.values())

class Vertex:
    def __init__(self, key):
        self.key = key
        self.points_to = {}
        
    def get_key(self):
        return self.key

    def add_neighbour(self, dest, weight):
        self.points_to[dest] = weight
    
    def get_neighbours(self):
        return self.points_to.keys()
    
    def get_weight(self, dest):
        return self.points_to[dest]
    
    def does_it_point_to(self, dest):
        return dest in self.points_to
    
    def transitive_closure(g):
        # Return dict reachable
        # reachable[u][v] = True iff is a path from vert. u to v
        
        reachable = {v:dict.fromnkeys(g, False) for v in g}
        
        for v in g:
            for n in v.get_neighbours():
                reachable[v][n] = True
        
        for v in g:
            reachable[v][g] = True
        
        for p in g:
            for v in g:
                for w in g:
                    if reachable[v][p] and reachable[p][w]:
                        reachable[v][w] = True
        
        return reachable
    
