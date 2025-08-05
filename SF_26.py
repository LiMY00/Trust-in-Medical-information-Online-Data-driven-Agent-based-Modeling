import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats

class Doctor:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Non_Doc:
    def __init__(self, NonDoc_id,  agent_type):
        self.agent_id = NonDoc_id
        lower, upper = 0, 1
        mu, sigma = 0.54, 0.29
        self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self.agent_type = agent_type

class Influencer:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Bot:
    def __init__(self, Bot_id, agent_type):
        self.agent_id = Bot_id
        self.opinion = 0
        self.agent_type = agent_type        

class SFNetwork:

    def __init__(self, num_Doc, num_NonDoc, num_Bot, num_Influencer, num_stub_influencer, num_titled_doc, num_titled_bot, random_state) -> None:
        # self.graph = nx.barabasi_albert_graph(num_Doc+num_NonDoc, 1, seed = random_state)
        self.graph = nx.DiGraph(nx.scale_free_graph(num_Doc+num_NonDoc + num_Bot, seed=random_state)) 
        # self.graph = self.generate_PBN()
        self.agents = {}
        self.num_of_nodes = num_Doc+num_NonDoc + num_Bot
        # self.num_of_nodes = len(self.graph.nodes)
        self.degree = self.graph.in_degree()
        # self.degree = self.graph.degree()
        self.nodes = []
        for node in self.graph.nodes():
            self.nodes.append(node)
        
        influencer = sorted(self.degree, key=lambda x: x[1], reverse=True)[:num_Influencer]
        self.influencer = [item[0] for item in influencer]
        low_degree = [x for x in self.nodes if x not in self.influencer]


        self.doc_nodes = random.sample(low_degree, num_Doc)
        self.doc_nodes_title = random.sample(self.doc_nodes, num_titled_doc)
        self.bot_nodes = random.sample([node for node in low_degree if node not in self.doc_nodes], num_Bot)
        self.bot_nodes_title = random.sample(self.bot_nodes, num_titled_bot)

        stub_influencer = sorted(self.degree, key=lambda x: x[1], reverse=True)[:num_stub_influencer]
        self.stub_influencer = [item[0] for item in stub_influencer]
        self.boost_edges = []
        self.boost_active = False

        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)
            if node in self.doc_nodes:
                if node in self.doc_nodes_title:
                    agent = Doctor(node, "Doctor")
                    self.agents[node] = agent
                else:
                    agent = Doctor(node, "Notitle_doctor")
                    self.agents[node] = agent
            elif node in self.bot_nodes:
                if node in self.bot_nodes_title:
                    agent = Bot(node, "Titled_Bot")
                else:
                    agent = Bot(node, "Bot")
                self.agents[node] = agent
            elif node in self.stub_influencer:
                agent = Influencer(node, "Influencer")
                self.agents[node] = agent
            else:
                agent = Non_Doc(node, "Non_Doc")
                self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix()
        self.previous_opinions = [10 for node in self.graph.nodes()]  
    
    def calculate_edge_weights(self):

        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            num_TitledNeighbors = 0
            num_NonTitledNeighbors = 0
            num_stub_Influencer = 0
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Doc" or "Titled_Bot" and neighbor != node1:
                    num_TitledNeighbors = num_TitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Non_Doc" and neighbor != node1:
                    num_TitledNeighbors = num_TitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Bot" and neighbor != node1:
                    num_TitledNeighbors = num_TitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Notitle_doctor" and neighbor != node1:
                    num_TitledNeighbors = num_TitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Influencer" and neighbor != node1:
                    num_stub_Influencer = num_stub_Influencer + 1  
            if agent_type1 == "Doctor" or agent_type1 == "Notitle_doctor" or agent_type1 == "Influencer" or agent_type1 == "Bot" or agent_type1 == "Titled_Bot":
                if node1 == node2:
                    weight = 1
                # elif agent_type2 == "Doctor" and num_TitledNeighbors !=0:
                #     weight = 0.4/num_TitledNeighbors
                else:
                    weight = 0
            else:  
            # Assign weights based on agent types
                if node1 != node2:
                    if agent_type2 == "Non_Doc" or agent_type2 == "Notitle_doctor" or agent_type2 == "Influencer" or agent_type2 == "Bot":
                        weight = 0.003/(0.183*num_TitledNeighbors + 0.003*(num_NonTitledNeighbors+num_stub_Influencer) + 0.906)
                    else:
                        weight = 0.183/(0.183*num_TitledNeighbors + 0.003*(num_NonTitledNeighbors+num_stub_Influencer) + 0.906)
                else:                         
                    weight = 0.906/(0.183*num_TitledNeighbors + 0.003*(num_NonTitledNeighbors+num_stub_Influencer) + 0.906)
        
            self.graph[node1][node2]['weight'] = weight
    
    def generate_weight_matrix(self):

        weight_matrix = np.zeros((self.num_of_nodes, self.num_of_nodes))

        for edge in self.graph.edges():
            node1, node2 = edge
            weight_matrix[node1][node2] = self.graph[node1][node2]['weight']

        return weight_matrix
    
    # def boost_doctor_visibility(self):
    #     """
    #     Add edges from all doctor agents to all other agents to boost their visibility.
    #     These edges will be removed after the first iteration.
    #     # """
    #     if self.boost_active:
    #         return  # Already boosted
        
    #     self.boost_edges = []
        
    #     # Get all doctor nodes (both titled and non-titled)
    #     all_doctor_nodes = self.doc_nodes
        
    #     # Add edges from each doctor to all other nodes
    #     for doctor_node in all_doctor_nodes:
    #         for target_node in self.graph.nodes():
    #             if doctor_node != target_node and not self.graph.has_edge(target_node, doctor_node):
    #                 self.graph.add_edge(target_node, doctor_node)
    #                 self.boost_edges.append((target_node, doctor_node))
        
    #     # Recalculate weights and weight matrix with the new edges
    #     self.calculate_edge_weights()
    #     self.weight_matrix = self.generate_weight_matrix()
    #     self.boost_active = True
    #     print(f"Added {len(self.boost_edges)} boost edges to enhance visibility of doctor.")

    def boost_doctor_visibility(self, doctor_node=None):
        """
        Add edges from one specific doctor agent to all other agents to boost their visibility.
        These edges will be removed after the first iteration.
        
        Args:
            doctor_node: Specific doctor node to boost. If None, selects a random doctor.
        """
        if self.boost_active:
            return  # Already boosted
        
        self.boost_edges = []
        
        # Select doctor to boost
        if doctor_node is None:
            # Select a random doctor from all doctor nodes
            # Use specified doctor if it exists in doctor nodes
            if doctor_node in self.doc_nodes:
                selected_doctor = doctor_node
            else:
                print(f"Warning: Node {doctor_node} is not a doctor. Selecting random doctor instead.")
                selected_doctor = random.choice(self.doc_nodes)
        
        # Add edges from the selected doctor to all other nodes
        for target_node in self.graph.nodes():
            if selected_doctor != target_node and not self.graph.has_edge(target_node, selected_doctor):
                self.graph.add_edge(target_node, selected_doctor)
                self.boost_edges.append((target_node, selected_doctor))
        
        # Recalculate weights and weight matrix with the new edges
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix()
        self.boost_active = True
        self.boosted_doctor = selected_doctor


    def remove_doctor_boost(self):
        """
        Remove the temporary boost edges that were added to enhance doctor visibility.
        """
        if not self.boost_active:
            return  # No boost to remove
        
        # Remove all boost edges
        for edge in self.boost_edges:
            if self.graph.has_edge(edge[0], edge[1]):
                self.graph.remove_edge(edge[0], edge[1])
        
        # Clear the boost edges list and reset boost status
        removed_count = len(self.boost_edges)
        self.boost_edges = []
        self.boost_active = False
        
        # Recalculate weights and weight matrix without the boost edges
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix()
        
        # print(f"Removed {removed_count} boost edges")

    def update_opinions(self):
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            
            weighted_sum = sum(self.weight_matrix[node][n] * self.agents[n].opinion for n in neighbors)
            new_opinion = weighted_sum / sum(self.weight_matrix[node][n] for n in neighbors)
            self.agents[node].opinion = new_opinion

    def consensus_reached(self, tolerance=0.001):
        opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        consensus_opinion = sum(opinions) / len(opinions)
        return all(abs(opinion - consensus_opinion) <= tolerance for opinion in opinions)
    
    def opinions_stabilized(self, tolerance=1e-4):
        current_opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        deltas = [abs(curr - prev) for curr, prev in zip(current_opinions, self.previous_opinions)]
        max_delta = max(deltas)
        self.previous_opinions = current_opinions
        return max_delta < tolerance
    
    def generate_PBN(self):
        gml_path = 'polblogs.gml'
        self.G = nx.read_gml(gml_path, label='id')

        if min(self.G.nodes()) == 1:
            mapping = {node: node - 1 for node in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)
        return self.G
    
    # def opinions_stabilized(self, tolerance=1e-4):
    #     current_opinions = [self.agents[node].opinion for node in self.graph.nodes()]
    #     deltas = [abs(curr - prev) for curr, prev in zip(current_opinions, self.previous_opinions)]
    #     max_delta = max(deltas)
    #     self.previous_opinions = current_opinions
    #     return max_delta < tolerance
