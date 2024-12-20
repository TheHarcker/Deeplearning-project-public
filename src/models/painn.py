import torch
import torch.nn as nn


def create_edge_data(
    atom_positions: torch.FloatTensor, 
    graph_indexes: torch.LongTensor, 
    cutoff_dist: float
) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Create an edge indices matrix for each edge (i,j) with a correspoding edge 
    diff r_ij = r_i - r_j and edge norm ||r_ij||

    Args:
        atom_positions: torch.FloatTensor of size [num_nodes, 3] with
            euclidean coordinates of each node / atom.
        graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
            index each node belongs to.
        cutoff_dist: float specifiyng the cutoff distance for two nodes to 
            have an edge between each other

    Returns:
        edges_indices: torch.LongTensor of size [num_edges, 2] with edge indecies
        edges_diff: torch.FloatTensor of size [num_edges, 3] with node position 
            difference r_ij = r_i - r_j = (x,y,z) for the two nodes in an edge
        edges_norm: torch.FloatTensor of size [num_edges] with the norm of the 
            difference ||r_ij|| = sqrt(x^2 + y^2 + z^2) for each edge
    """
    num_nodes = atom_positions.shape[0]
    assert atom_positions.shape[0] == graph_indexes.shape[0] and atom_positions.shape[1] == 3

    # Relative difference of atom positions
    r = atom_positions - atom_positions.unsqueeze(1)
    assert r.shape == (num_nodes, num_nodes, 3)

    # L2 norm (Euclidian distance) of relative positions
    r_norm = torch.linalg.vector_norm(r, ord=2, dim=2)
    assert r_norm.shape == (num_nodes, num_nodes)

    # Adjancency matrix for our graph
    adjancency = (r_norm <= cutoff_dist) & (graph_indexes == graph_indexes.unsqueeze(1))
    assert adjancency.shape == (num_nodes, num_nodes)

    # Get edge indices from the mask
    edges_indices = adjancency.nonzero(as_tuple=False)
    assert edges_indices.shape[1] == 2

    # Remove self-loops (i.e., edges where source == target)
    edges_indices = edges_indices[edges_indices[:, 0] != edges_indices[:, 1]]
    num_edges = edges_indices.shape[0]
    assert edges_indices.shape == (num_edges, 2)

    # Extract edge difference and edge norm
    edges_diff = r[edges_indices[:, 0], edges_indices[:, 1], :]
    edges_norm = r_norm[edges_indices[:, 0], edges_indices[:, 1]]
    assert edges_diff.shape == (num_edges, 3) and edges_norm.shape == (num_edges,)

    return edges_indices, edges_diff, edges_norm


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.message_layers = nn.ModuleList([PaiNNMessage(num_features, num_rbf_features, cutoff_dist) for _ in range(num_message_passing_layers)])
        self.update_layers = nn.ModuleList([PaiNNUpdate(num_features) for _ in range(num_message_passing_layers)])

        self.embedding = nn.Embedding(num_unique_atoms, num_features)

        hidden_dimension = 64
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, num_outputs)
        )


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        num_nodes = len(atoms)

        # Create edge index matrice with associate data
        edges_indices, edges_diff, edges_norm = create_edge_data(atom_positions, graph_indexes, self.cutoff_dist)

        # Initialize scalar and vector nodes
        s = self.embedding(atoms)
        v = torch.zeros((num_nodes, self.num_features, 3), device=atom_positions.device)
        assert s.shape == (num_nodes, self.num_features) and v.shape == (num_nodes, self.num_features, 3)

        for i in range(self.num_message_passing_layers):
            # Message layer
            dv, ds = self.message_layers[i](v, s, edges_indices, edges_diff, edges_norm)
            v = v + dv
            s = s + ds

            # Update layer
            dv, ds = self.update_layers[i](v, s)
            v = v + dv
            s = s + ds

        # Pass through the MLP
        mlp_output = self.mlp(s)
        assert mlp_output.shape == (num_nodes, self.num_outputs)

        return mlp_output


def radial_basis_functions(edge_norm: torch.FloatTensor, num_rbf_features: int, cutoff_dist: float) -> torch.FloatTensor:
    """
    Sinc radial basis functions based on edge distance:

        sin(n * pi * ||r_ij|| / r_cut) / ||r_ij||
    
    Args:  
        edge_norm: torch.FloatTensor of size [num_edges] with ||r_ij|| for each edge
        num_rbf_features: float with number of RBF features which is usually 20
        cutoff_dist: float with cutoff value r_cut

    Returns:
        torch.FloatTensor of size [num_edges, num_rbf_features]
    """

    n = torch.arange(1, num_rbf_features+1, device=edge_norm.device) * torch.pi / cutoff_dist
    edge_norm = edge_norm.unsqueeze(-1)
    
    return torch.sin(edge_norm * n) / edge_norm


def cos_cutoff(edge_norm: torch.FloatTensor, cutoff_dist: float) -> torch.FloatTensor:
    """
    Computing the cutoff using the behler cosine function introduced on page 5 
    in source 39 of the PaiNN article:

    f(||r_ij||) = 0.5 * (cos(pi * ||r_ij|| / r_cut) + 1)  if ||r_ij|| ≤ r_cutoff 
                  0                                       otherwize

    Assuming that ||r_ij|| ≤ r_cutoff is given.

    Args:  
        edge_norm: torch.FloatTensor of size [num_edges] with ||r_ij|| for each edge
        cutoff_dist: float with cutoff value r_cut

    Returns:
        torch.FloatTensor of size [num_edges]
    """

    return 0.5 * (torch.cos(torch.pi * edge_norm / cutoff_dist) + 1)


class PaiNNMessage(nn.Module):
    """
    Message Component of Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(self, num_features: int, num_rbf_features: int, cutoff_dist: float) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.cutoff_dist = cutoff_dist

        super().__init__()

        self.phi_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, 3 * self.num_features)
        )

        self.rbf_layer = nn.Linear(self.num_rbf_features, 3 * self.num_features)


    def forward(
        self, 
        v: torch.FloatTensor, 
        s: torch.FloatTensor, 
        edges_indices: torch.LongTensor, 
        edges_diff: torch.FloatTensor, 
        edges_norm: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            v: torch.FloatTensor of size [num_nodes, num_features, 3] with vector nodes features
            s: torch.FloatTensor of size [num_nodes, num_features] with scalar nodes features
            edges_indices: torch.LongTensor of size [num_edges, 2] with edge indecies (i,j) for each edge
            edges_diff: torch.FloatTensor of size [num_edges, 3] with r_i - r_j for each edge
            edges_norm: torch.FloatTensor of size [num_edges] with ||r_ij|| for each edge

        Returns:
            dv: torch.FloatTensor of size [num_nodes, num_features, 3] with changes to the vector nodes features
            ds: torch.FloatTensor of size [num_nodes, num_features] with changes to the scalar nodes features
        """
        num_nodes = v.shape[0]
        num_edges = edges_indices.shape[0]
        assert v.shape[0] == s.shape[0] and v.shape[1] == s.shape[1] == self.num_features and v.shape[2] == 3
        assert edges_indices.shape[0] == edges_diff.shape[0] == edges_norm.shape[0] and edges_indices.shape[1] == 2 and edges_diff.shape[1] == 3

        # Compute MPL layer for the scalar nodes returning phi
        phi = self.phi_mlp(s)
        assert phi.shape == (num_nodes, 3 * self.num_features)
        
        # Compute sinc radial basis functions (RBF) based on edge distance
        rbf_layer_input = radial_basis_functions(edges_norm, self.num_rbf_features, self.cutoff_dist)
        assert rbf_layer_input.shape == (num_edges, self.num_rbf_features)

        # Compute linear layer in RBF part
        rbf_layer_output = self.rbf_layer(rbf_layer_input)
        assert rbf_layer_output.shape == (num_edges, 3 * self.num_features)

        # Compute and apply cosine cutoff 
        W = rbf_layer_output * cos_cutoff(edges_norm, self.cutoff_dist).unsqueeze(-1)
        assert W.shape == (num_edges, 3 * self.num_features)

        # Compute elementwise matrix multiplication
        split_input = phi[edges_indices[:, 1]] * W
        assert split_input.shape == (num_edges, 3 * self.num_features)

        # Split input
        s1, s2, s3 = torch.split(split_input, self.num_features, dim=1)
        assert s1.shape == s2.shape == s3.shape == (num_edges, self.num_features)

        # Make r_ij to a unit vectors i.e. vector of length 1
        edges_unit = edges_diff / edges_norm.unsqueeze(-1)
        assert edges_unit.shape == (num_edges, 3)

        # Compute product with s3 and edge difference unit vectors
        prod_s1 = s1.unsqueeze(-1) * v[edges_indices[:, 1]]
        assert prod_s1.shape == (num_edges, self.num_features, 3)

        # Compute product with s3 and edge difference unit vectors
        prod_s3 = s3.unsqueeze(-1) * edges_unit.unsqueeze(1)
        assert prod_s3.shape == (num_edges, self.num_features, 3)

        # Sum prod_s1 and prod_s3
        sum_s1_s3 = prod_s1 + prod_s3
        assert sum_s1_s3.shape == (num_edges, self.num_features, 3)

        # Sum over edges to change dimension 0 from num_edges to num_nodes to get dv
        dv = torch.zeros_like(v)
        dv.index_add_(0, edges_indices[:, 0], sum_s1_s3)
        assert dv.shape == (num_nodes, self.num_features, 3)

        # Sum over edges to change dimension 0 from num_edges to num_nodes to get ds
        ds = torch.zeros_like(s)
        ds.index_add_(0, edges_indices[:, 0], s2)
        assert ds.shape == (num_nodes, self.num_features)

        return dv, ds
    

class PaiNNUpdate(nn.Module):
    """
    Update Component of the Polarizable Atom Interaction Neural Network (PaiNN).
    """

    def __init__(self, num_features: int) -> None:
        """
        Args:
            num_features (int): Number of features for each atom.
        """
        super().__init__()
        self.num_features = num_features

        self.U = nn.Linear(num_features, num_features, bias=False)
        self.V = nn.Linear(num_features, num_features, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, 3 * num_features),
        )


    def forward(self, v: torch.FloatTensor, s: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            v: torch.FloatTensor of size [num_nodes, num_features, 3] with vector nodes features
            s: torch.FloatTensor of size [num_nodes, num_features] with scalar nodes features

        Returns:
            dv: torch.FloatTensor of size [num_nodes, num_features, 3] with changes to the vector nodes features
            ds: torch.FloatTensor of size [num_nodes, num_features] with changes to the scalar nodes features
        """
        num_nodes = v.shape[0]
        assert v.shape[0] == s.shape[0] and v.shape[1] == s.shape[1] == self.num_features and v.shape[2] == 3

        # Matrix vector product with U/V and v where v's last and second last dimension is swapped in the product
        permuted_v = v.permute(0, 2, 1)
        Uv = self.U(permuted_v).permute(0, 2, 1)
        Vv = self.V(permuted_v).permute(0, 2, 1)
        assert Uv.shape == (num_nodes, self.num_features, 3) and Vv.shape == (num_nodes, self.num_features, 3)

        # Compute the inner product between Uv and Vv
        inner_prod = torch.sum(Uv * Vv, dim=2) 
        assert inner_prod.shape == (num_nodes, self.num_features)

        # Compute the L2 norm of Vv of the last dimension
        norm = torch.linalg.vector_norm(Vv, ord=2, dim=2)
        assert norm.shape == (num_nodes, self.num_features)

        # Concatenate norm with scalar features
        mlp_input = torch.cat((norm, s), dim=1)
        assert mlp_input.shape == (num_nodes, 2 * self.num_features)
        
        # Pass through the MLP
        mlp_output = self.mlp(mlp_input)
        assert mlp_output.shape == (num_nodes, 3 * self.num_features)
        
        # Split the MLP output into three parts
        avv, asv, ass = torch.split(mlp_output, self.num_features, dim=1) 
        assert avv.shape == asv.shape == ass.shape == (num_nodes, self.num_features)

        # Compute the change to vector features and scalar features
        dv = Uv * avv.unsqueeze(-1)
        ds = ass + asv * inner_prod
        assert dv.shape == (num_nodes, self.num_features, 3) and ds.shape == (num_nodes, self.num_features)

        return dv, ds